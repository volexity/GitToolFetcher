"""GitToolFetcher manages multiple versions of github-hosted projects."""

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

import multiprocess  # type: ignore[import-untyped]
import multiprocess.queues  # type: ignore[import-untyped]
import requests
from multiprocess import Process
from multiprocess.queues import Queue

from .models.tool_management_result import ToolManagementResult
from .models.tool_process_error import ToolProcessError
from .plaintext_stream_handler import PlainTextStreamHandler

logger: Final[logging.Logger] = logging.getLogger(__name__)


handler = PlainTextStreamHandler()
formatter = logging.Formatter("GitToolFetcher: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

if TYPE_CHECKING:
    ArchiveQueue: TypeAlias = Queue[tuple[str, Path]]  # only for static analysis
    OutputQueue: TypeAlias = multiprocess.queues.Queue[ToolManagementResult]
else:
    ArchiveQueue = Queue
    OutputQueue = Queue

archive_queue: ArchiveQueue
output_queue: OutputQueue

# Spawns fresh interpreter, avoids issues with fork() from global thread
with contextlib.suppress(RuntimeError):
    multiprocess.set_start_method("spawn")


class GitToolFetcher:
    """GitToolFetcher manages multiple versions of github-hosted projects."""

    def __init__(
        self,
        repo_name: str,
        storage_base: Path,
        *,
        bin_path: Path | None = None,
        version_encode_callback: Callable[[str], str | None] | None = None,
        version_decode_callback: Callable[[str], str | None] | None = None,
        install_callback: Callable[[str, Path, Path], Any] | None = None,
        uninstall_callback: Callable[[str, Path], Any] | None = None,
        display_progress: bool = False,
    ) -> None:
        """Initialize the GitToolFetcher context with default values.

        Args:
            repo_name (str) : Repository name the project is hosted on.
            storage_base (Path) : Top level directory allocated to GitToolFetcher.
            bin_path (Path | None) : Relative path with the installed project where the binaries are
                                     meant to be executed.
            version_encode_callback (Callable[[str], str | None] | None): Callback to encode a version string
                                                                          into the project's tags.
            version_decode_callback (Callable[[str], str | None] | None): Callback to decode the project's tags
                                                                          into a version string.
            install_callback (Callable[[str, Path, Path], Any] | None): Custom logic for the install process.
            uninstall_callback (Callable[[str, Path], Any] | None): Custom logic for the uninstall process.
            display_progress (bool): Weather to output progress updates to the console.
        """
        super().__init__()

        self.__display_progress: Final[bool] = display_progress

        # Partition the storage base with each github repo name.
        storage_base = storage_base / repo_name

        self.__repo_name: Final[str] = repo_name
        self.__bin_path: Final[Path] = GitToolFetcher.__sanitize(bin_path) if bin_path else Path()

        # Setup callbacks
        self.__version_encode_callback: Final[Callable[[str], str | None]] = (
            version_encode_callback if version_encode_callback else self.__default_version_callback
        )
        self.__version_decode_callback: Final[Callable[[str], str | None]] = (
            version_decode_callback if version_decode_callback else self.__default_version_callback
        )
        self.__install_callback: Final[Callable[[str, Path, Path], Any]] = (
            install_callback if install_callback else self.__default_install_callback
        )
        self.__uninstall_callback: Final[Callable[[str, Path], Any]] = (
            uninstall_callback if uninstall_callback else self.__default_uninstall_callback
        )

        logger.debug(f"Using storage base: {storage_base}")
        self.__install_path: Final[Path] = storage_base / "installed"
        self.__download_path: Final[Path] = storage_base / "downloads"
        self.__tmp_path: Final[Path] = storage_base / "tmp"

        logger.debug("Creating directories to manage project versions...")
        self.__install_path.mkdir(parents=True, exist_ok=True)
        self.__download_path.mkdir(parents=True, exist_ok=True)
        self.__tmp_path.mkdir(parents=True, exist_ok=True)

        self.__available_cache_path: Final[Path] = storage_base / "available_cache.json"
        logger.debug("Initializing available versions")
        if not self.__available_cache_path.exists():
            self.refresh()

    @staticmethod
    def __default_version_callback(version: str) -> str:
        """Default version encode/decode callback, simply returns the version string without alteration.

        Args:
            version (str): Version string to encode/decode.

        Returns:
            str: Encoded/Decoded version string.
        """
        return version

    @staticmethod
    def __default_install_callback(_version: str, archive_data_path: Path, install_path: Path) -> None:
        """Default install callback, simply moves the content of the archive to the install directory.

        Args:
            version (str): The version of the project being installed.
            archive_data_path (Path): The path to the content of the project's archive.
            install_path (Path): The path to the temporary install directory.
        """
        archive_data_path.rename(install_path)

    @staticmethod
    def __default_uninstall_callback(_version: str, _install_path: Path) -> None:
        """Default uninstall callback, performs no action.

        Args:
            version (str): The version of the project being installed.
            install_path (Path): The path to the temporary install directory.
        """
        return

    @staticmethod
    def __sanitize(path: Path) -> Path:
        """Sanitize the user input path as to prevent dir walks.

        Args:
            path (Path) : The path to sanitize.

        Returns:
            Path:  The sanitized path.
        """
        return Path(os.path.relpath(os.path.join("/", path), "/"))  # noqa: PTH118

    def refresh(self) -> None:
        """Refresh the list of available project versions from Github's API."""
        logger.debug("Refreshing available project versions ...")
        available_versions: dict[str, str] = {}
        tags_per_page: Final[int] = 100
        tags_url: str = f"https://api.github.com/repos/{self.__repo_name}/tags?per_page={tags_per_page}"

        while True:
            response: requests.Response = requests.get(tags_url, allow_redirects=True, timeout=60)
            if response.status_code != 200:  # noqa: PLR2004
                logger.error(f"Got bad status code from github: {response.status_code}")
                logger.error(f"Response: {response.content!r}")
                raise ConnectionError(response.content)

            tags_data: bytes | Any = response.content
            tags: Any = json.loads(tags_data)

            for tag in tags:
                if (version := self.__version_decode_callback(tag["name"])) is not None:
                    available_versions[version] = tag["tarball_url"]

            if response.links.get("next") is None:
                break
            tags_url = response.links["next"]["url"]

        with self.__available_cache_path.open("wb") as file:
            file.write(json.dumps(available_versions).encode("utf-8"))

        logger.info(f"Refreshed available {self.__repo_name} versions.")

    def _list_available_paths(self, *, refresh: bool = False) -> dict[str, str]:
        """Get the available project versions locally, with remote paths.

        Args:
            refresh (bool, optional): Whether to refresh the available versions. Defaults to False.

        Returns:
            dict[str, str]: Available project versions as {version: path}.
        """
        if refresh:
            self.refresh()
        with self.__available_cache_path.open() as file:
            return json.load(file)

    def _list_available(self, *, refresh: bool = False) -> Iterable[str]:
        """Get the available project versions locally.

        Args:
            *
            refresh (bool, optional): Whether to refresh the available versions. Defaults to False.

        Returns:
            Iterable[str]: The list of available project versions.
        """
        return (v for v in self._list_available_paths(refresh=refresh))

    def _list_installed(self) -> Iterable[str]:
        """Get the locally installed Project versions.

        Returns:
            Iterable[str]: Containing keys (project versions) whose values are install paths.
        """
        return (d.name for d in self.__install_path.iterdir() if d.is_dir())

    def _download_task(
        self,
        archive_queue: ArchiveQueue,
        version: str,
        target_dir: Path,
        *,
        force: bool = False,
    ) -> bool:
        """Download a project's version from github.

        Args:
            archive_queue (Queue[tuple[str, Path]]): Outgoing queue of versions and paths.
            version (str): The project's version to download.
            target_dir (Path): Where to save the download to.
            *
            force (bool): Wether to overwrite existing files.

        Raise:
            Exception: If the project's version to download is not available.

        Returns:
            bool: Whether the download was successful
        """
        logger.info(f'Downloading "{self.__repo_name}" version {version}...')

        archive_path: Path = target_dir / version
        if not force and archive_path.exists():
            logger.info(f"\033[32m✔\033[0m {version} already downloaded.")
            archive_queue.put((version, archive_path))
            return False

        try:
            response: Final[requests.Response] = requests.get(
                self._list_available_paths(refresh=False)[version], allow_redirects=True, timeout=60, stream=True
            )
        except KeyError:
            # Version not valid, missing from the JSON.
            logger.info(f"\033[31m✘\033[0m {version} not found, skipping.")
            return False

        with TemporaryDirectory(dir=self.__tmp_path, prefix=f"download_{version}_") as tmpdir:
            tmparchive_path: Final[Path] = self.__tmp_path / tmpdir / version

            with tmparchive_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1048576):
                    file.write(chunk)
                    file.flush()

            tmparchive_path.rename(archive_path)

            logger.info(f"\033[32m✔\033[0m {version} downloaded succesfully.")
            archive_queue.put((version, archive_path))
            return True
        return False

    def download(self, *versions: str, target_dir: Path, force: bool = False) -> list[tuple[str, Path]]:
        """Downloads all desired project versions from Github.

        Args:
            versions: Versions to download.
            target_dir (Path): Where to save the download to.
            force: Whether to overwrite existing files.

        Raise:
            TODO

        Returns:
            list[Path]: The paths of all downloaded archives.
        """
        logger.info("\033[1mDOWNLOAD\033[0m")

        archive_queue: multiprocess.queues.Queue[tuple[str, Path]] = cast(
            "multiprocess.queues.Queue[tuple[str, Path]]", multiprocess.Queue()
        )
        process_pool: Final[list[Process]] = [
            Process(
                target=self._download_task,
                kwargs={"archive_queue": archive_queue, "version": version, "target_dir": target_dir, "force": force},
            )
            for version in versions
        ]

        for process in process_pool:
            process.start()

        for process in process_pool:
            process.join()

        archive_list: Final[list[tuple[str, Path]]] = []
        while not archive_queue.empty():
            archive_list.append(archive_queue.get())

        return archive_list

    def list_available(self, *, refresh: bool = False) -> Iterable[str]:
        """Get the available project versions locally.

        Args:
            *
            refresh (bool, optional): Whether to refresh the available versions. Defaults to False.

        Returns:
            Iterable[str]: The list of available project versions.
        """
        available_versions: Final[Iterable[str]] = self._list_available(refresh=refresh)
        return sorted({ver for ver in map(self.__version_decode_callback, available_versions) if ver is not None})

    def list_installed(self) -> Iterable[str]:
        """Get the locally installed Project versions.

        Returns:
            Iterable[str]: Containing keys (project versions) whose values are install paths.
        """
        result: Final[Iterable[str]] = self._list_installed()
        return sorted({ver for ver in map(self.__version_decode_callback, result) if ver is not None})

    def _install_task(
        self,
        output_queue: OutputQueue,
        version: str,
        tar_path: Path,
        *,
        force: bool = False,
    ) -> None:
        """Installs the specified project version.

        This method checks if the specified project version is already installed.
        If it is, it logs an error. If the project version is not installed, it checks
        if it is available for installation.
        If the version is available, it proceeds to download, build, and install it.
        The function will populate a queue of installations if it succeeds.
        If any error occurs during the installation process, it logs the error and raises
        an exception.

        The output_queue contains the result of each installation.

        Args:
            output_queue (Queue[InstallationResult]): Outgoing queue of installations results.
            version (str): The project version to install.
            tar_path (Path): Path to tarball for specified version.
            *
            force (bool): Whether to force installation or not.

        Raises:
            Exception: If an error occurs during the installation process.
        """
        # Checks for previously installed versions
        installed: Final[Iterable[str]] = self._list_installed()
        if not force and version in installed:
            logger.info(f"\033[32m✔\033[0m {self.__repo_name} version {version} is already installed.")
            output_queue.put(ToolManagementResult(version, status=True, result=None))
            return

        # Checks for available versions
        available: Iterable[str] = self._list_available()
        if version not in available:
            # Attempts to refresh the cache
            available = self._list_available(refresh=True)
            if version not in available:
                logger.info(f"\033[31m✘\033[0m {self.__repo_name} version {version} is not available.")
                output_queue.put(ToolManagementResult(version, status=False, result=None))
                return

        result: dict[str, Any] | None = None
        # Install the new version
        try:
            # STEP.1 : Install
            with (
                TemporaryDirectory(
                    prefix=f"build_{version}_",
                    dir=self.__tmp_path,
                    # this is redundant (would get cleaned up automatically) but making
                    # explicit to support future debugging functionality that disables
                    # auto-delete for failed builds
                    delete=True,
                ) as build_tmpdir,
                tarfile.open(tar_path, "r") as tar_file,
            ):
                toplevelname: Final[Path] = GitToolFetcher.__sanitize(Path(tar_file.getnames()[0]))
                archive_data_path: Final[Path] = Path(build_tmpdir) / toplevelname
                tmp_install_path: Final[Path] = Path(build_tmpdir) / version
                install_path: Final[Path] = self.__install_path / version

                # Unpack the tarball pulled from Github
                logger.info(f'Extracting "{self.__repo_name}" version {tar_path.name}')

                for file in tar_file.getmembers():
                    tar_file.extract(file, build_tmpdir)
                logger.info(f"\033[32m✔\033[0m {version} extracted.")

                if install_path.exists():
                    shutil.rmtree(install_path)

                # Call the install function provided to the constructor
                logger.info(f"Installing {self.__repo_name} version {version} (this may take a while)...")
                result = self.__install_callback(version, archive_data_path, tmp_install_path)
                tmp_install_path.rename(install_path)

                logger.info(f"\033[32m✔\033[0m {self.__repo_name} version {version} installed at {install_path}.")
                output_queue.put(ToolManagementResult(version, status=True, result=result))
                return

            # STEP.3 : Cleanup
            tar_path.unlink()
        except Exception as e:
            # General exception logic
            logger.error(f"{self.__repo_name} version {version} installation failed")  # noqa: TRY400
            if isinstance(e, FileNotFoundError):
                pass
            elif isinstance(e, ToolProcessError):
                result = e.result
            else:
                # Raise what we can't handle
                raise
            output_queue.put(ToolManagementResult(version, status=False, result=result))
            return
        output_queue.put(ToolManagementResult(version, status=True, result=result))
        return

    def install(self, *versions: str, force: bool = False) -> list[ToolManagementResult]:
        """Installs all desired project versions.

        This method first ensures all versions are downloaded.
        It then creates a process pool of installations to execute in parallel.
        These processes will populate a queue with successful installations.
        Once the pool finishes, the method builds a list from the queue and returns.

        Args:
            *versions (str): The project versions to install.
            force (bool): Whether to force installations or not.

        Returns:
            list[tuple[bool, Any | None]]: List of installation results.
        """
        result_list: list[ToolManagementResult] = []
        # Download all version tarballs from Github
        tar_paths: Final[list[tuple[str, Path]]] = self.download(
            *versions, target_dir=self.__download_path, force=force
        )

        logger.info("\033[1mINSTALL\033[0m")

        # Execute the installations in parallel
        output_queue: Final[multiprocess.queues.Queue[ToolManagementResult]] = cast(
            "multiprocess.queues.Queue[ToolManagementResult]", multiprocess.Queue()
        )
        process_pool: Final[list[Process]] = [
            Process(
                target=self._install_task,
                kwargs={"output_queue": output_queue, "version": version, "tar_path": tar_path, "force": force},
            )
            for version, tar_path in tar_paths
        ]

        for process in process_pool:
            process.start()

        for process in process_pool:
            process.join()

        # Build a list of successful installations to return
        while not output_queue.empty():
            result_list.append(output_queue.get())
        return result_list

    def uninstall(self, *versions: str) -> list[ToolManagementResult]:
        """Uninstall a locally installed version.

        Args:
            *versions (str): The project's versions to uninstall.

        Return:
            bool : True if the uninstall succeded.
            Any | None: Custom data returned by the install callback.
                        Returns None if the command didn't run.
        """
        result_list: list[ToolManagementResult] = []
        for version in versions:
            result: dict[str, Any] | None = None
            logger.info(f"Uninstalling version {version}...")
            installed_versions: Iterable[str] = self._list_installed()

            if version not in installed_versions:
                logger.error(f"\033[31m✘\033[0m {self.__repo_name} version {version} not installed.")
                result_list.append(ToolManagementResult(version, status=False, result=result))
                continue

            install_dir: Path = self.__install_path / version
            result = self.__uninstall_callback(version, install_dir)

            with TemporaryDirectory(prefix=f"remove_{version}_", dir=self.__tmp_path) as remove_tmpdir:
                install_dir.rename(Path(remove_tmpdir) / version)
                logger.info(f"\033[32m✔\033[0m {self.__repo_name} version {version} has been uninstalled.")
            result_list.append(ToolManagementResult(version, status=True, result=result))
        return result_list

    def get_tool_path(self, version: str) -> Path | None:
        """Returns the absolute install path of the specified project version.

        Args:
            version (str) : The version of the tool to retrive the path too.

        Returns:
            Path : The path to the specified project version (if any).
        """
        if version in self._list_installed():
            return (self.__install_path / version).resolve()
        return None

    def run(
        self,
        version: str,
        command: str,
        *vargs: str,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        input_data: bytes | None = None,
        stdin: int | None = None,
        stdout: int | None = None,
        stderr: int | None = None,
    ) -> int:
        """Runs a command from a specified version of the project.

        Args:
            version (str) : The version of the tool to select.
            command (str) : The command to run.
            *vargs (str) : List of arguments to pass to the command.
            env (dict[str, str], optional) : Additional environement variables to set as key-values pairs.
            cwd (Path, optional) : Work directory overide (if any).
            input_data (bytes | None) : Data to be piped into stdin (if any).
            stdin (int | None) : File descriptor to redirect stdin to (if any).
            stdout (int | None) : File descriptor to redirect stdout to (if any).
            stderr (int | None) : File descriptor to redirect stderr to (if any).
        """
        if tool_path := self.get_tool_path(version):
            workdir: Path = tool_path / self.__bin_path
            full_command: Final[Path] = workdir / GitToolFetcher.__sanitize(Path(command))

            if cwd:
                workdir = cwd

            env_vars: dict[str, str] = os.environ.copy()
            if env:
                env_vars.update(env)

            status = subprocess.run(  # noqa: S603
                [full_command, *vargs],
                env=env_vars,
                cwd=workdir,
                check=False,
                input=input_data,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
            )
            return status.returncode
        msg: str = f"Tool version {version} unavailable !"
        raise ValueError(msg)

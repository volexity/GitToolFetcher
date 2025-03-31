"""GitToolFetcher manages multiple versions of github-hosted projects."""

import json
import logging
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final

import requests
from tqdm import tqdm
from yaspin import yaspin

logger: Final[logging.Logger] = logging.getLogger(__name__)

PROGRESS_BAR_FORMAT: Final[str] = (
    "[{elapsed} - {remaining_s:.0f}s] {desc} |{bar}| [{n_fmt}/{total_fmt} - {rate_fmt}] ({percentage:3.0f} %)"
)


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
        install_callback: Callable[[str, Path, Path], None] | None = None,
        uninstall_callback: Callable[[str, Path], None] | None = None,
        display_progress: bool = False,
    ) -> None:
        """Initialize the GitToolFetcher context with default values.

        Args:
            repo_name (str) : Repository name the project is hosted on.
            storage_base (Path) : Top level directory allocated to GitToolFetcher.
            *
            bin_path (Path | None) : Relative path with the installed project where the binaries are
                                     meant to be executed.
            version_encode_callback (Callable[[str], str | None] | None): Callback to encode a version string
                                                                          into the project's tags.
            version_decode_callback (Callable[[str], str | None] | None): Callback to decode the project's tags
                                                                          into a version string.
            install_callback (Callable[[str, Path, Path], None] | None): Custom logic for the install process (if any).
            uninstall_callback (Callable[[str, Path], None] | None): Custom logic for the uninstall process (if any).
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
        self.__install_callback: Final[Callable[[str, Path, Path], None]] = (
            install_callback if install_callback else self.__default_install_callback
        )
        self.__uninstall_callback: Final[Callable[[str, Path], None]] = (
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
        """."""
        return version

    @staticmethod
    def __default_install_callback(version: str, archive_data_path: Path, install_path: Path) -> None:  # noqa: ARG004
        """Default install callback, simply moves the content of the archive to the install directory.

        Args:
            version (str): The version of the project being installed.
            archive_data_path (Path): The path to the content of the project's archive.
            install_path (Path): The path to the temporary install directory.
        """
        archive_data_path.rename(install_path)

    @staticmethod
    def __default_uninstall_callback(version: str, install_path: Path) -> None:
        """Default uninstall callback, performs no action.

        Args:
            version (str): The version of the project being installed.
            install_path (Path): The path to the temporary install directory.
        """

    @staticmethod
    def __sanitize(path: Path) -> Path:
        """Sanitize the user input path as to prevent dir walks.

        Args:
            path (Path) : The path to sanitize.
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
                available_versions[tag["name"]] = tag["tarball_url"]

            if response.links.get("next"):
                tags_url = response.links["next"]["url"]
            else:
                break

        with self.__available_cache_path.open("wb") as file:
            file.write(json.dumps(available_versions).encode("utf-8"))

        logger.info(f"Refreshed available {self.__repo_name} versions.")

    def _list_available_paths(self, *, refresh: bool = False) -> dict[str, str]:
        """Get the available project versions locally, with remote paths.

        Args:
            *
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

    def _download(self, version: str, target_dir: Path, *, force: bool = False) -> Path:
        """Download a project's version from github.

        Args:
            version (str): The project's version to download.
            target_dir (Path): Where to save the download to.
            *
            force (bool): Wether to overwrite existing files.

        Raise:
            Exception: If the project's version to download is not available.

        Returns:
            Path: The path of the downloaded archive.
        """
        archive_path: Path = target_dir / version
        if not force and archive_path.exists():
            return archive_path

        response: Final[requests.Response] = requests.get(
            self._list_available_paths(refresh=False)[version], allow_redirects=True, timeout=60, stream=True
        )

        tmpdir: Final[TemporaryDirectory] = TemporaryDirectory(dir=self.__tmp_path, prefix=f"download_{version}_")
        tmparchive_path: Final[Path] = self.__tmp_path / tmpdir.name / version

        with tmparchive_path.open("wb") as file:
            content_size: int = int(response.headers.get("content-length", 0))
            description: str = f'Downloading "{self.__repo_name}" version {version}'
            for chunk in tqdm(
                response.iter_content(chunk_size=1048576),
                bar_format=PROGRESS_BAR_FORMAT,
                desc=description,
                total=content_size,
                unit="B",
                unit_scale=True,
                colour="yellow",
                leave=False,
                disable=(not self.__display_progress),
            ):
                file.write(chunk)
                file.flush()
        tmparchive_path.rename(archive_path)

        return archive_path

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

    def download(self, version: str, target_dir: Path, *, force: bool = False) -> Path:
        """Download a project's version from github.

        Args:
            version (str): The project's version to download.
            target_dir (Path): Where to save the download to.
            *
            force (bool): Wether to overwrite existing files.

        Raise:
            Exception: If the project's version to download is not available.

        Returns:
            Path: The path of the downloaded archive.
        """
        if enc_version := self.__version_encode_callback(version):
            return self._download(enc_version, target_dir, force=force)
        msg: Final[str] = f"{enc_version} is not a valid {self.__repo_name} version."
        raise ValueError(msg)

    def install(self, version: str, *, force: bool = False) -> bool:
        """Installs the specified project version.

        This method checks if the specified project version is already installed.
        If it is, it logs an error. If the project version is not installed, it checks
        if it is available for installation.
        If the version is available, it proceeds to download, build, and install it.
        If any error occurs during the installation process, it logs the error and raises
        an exception.

        Args:
            version (str): The project version to install.
            *
            force (bool): Wether to force installation or not.

        Returns:
            bool: True if the install succeded.

        Raises:
            Exception: If an error occurs during the installation process.
        """
        if enc_version := self.__version_encode_callback(version):
            # Checks for previously installed versions
            installed: Final[Iterable[str]] = self._list_installed()
            if not force and enc_version in installed:
                logger.info(f"{self.__repo_name} version {enc_version} already installed")
                return True

            # Checks for available versions
            available: Iterable[str] = self._list_available()
            if enc_version not in available:
                # Attempts to refresh the cache
                available = self._list_available(refresh=True)
                if enc_version not in available:
                    logger.error(f"{self.__repo_name} version {enc_version} is not available")
                    return False

            # Install the new version
            try:
                # STEP.1 : Download
                tar_path: Final[Path] = self._download(enc_version, self.__download_path, force=force)

                # STEP.2 : Install
                build_tmpdir: Final[TemporaryDirectory] = TemporaryDirectory(
                    prefix=f"build_{enc_version}_", dir=self.__tmp_path
                )
                with tarfile.open(tar_path, "r") as tar_file:
                    toplevelname: Final[Path] = GitToolFetcher.__sanitize(Path(tar_file.getnames()[0]))
                    archive_data_path: Final[Path] = Path(build_tmpdir.name) / toplevelname
                    tmp_install_path: Final[Path] = Path(build_tmpdir.name) / enc_version
                    install_path: Final[Path] = self.__install_path / enc_version

                    description: Final[str] = f'Extracting "{self.__repo_name}" version {tar_path.name}'
                    for file in tqdm(
                        tar_file.getmembers(),
                        bar_format=PROGRESS_BAR_FORMAT,
                        desc=description,
                        total=len(tar_file.getmembers()),
                        unit="file",
                        colour="yellow",
                        leave=False,
                        disable=(not self.__display_progress),
                    ):
                        tar_file.extract(file, build_tmpdir.name)

                    if install_path.exists():
                        shutil.rmtree(install_path)

                    if self.__display_progress:
                        with yaspin() as spinner:
                            spinner.color = "green"
                            spinner.text = f'Installing "{self.__repo_name}" version {enc_version} ...'
                            self.__install_callback(version, archive_data_path, tmp_install_path)
                    else:
                        self.__install_callback(version, archive_data_path, tmp_install_path)

                    tmp_install_path.rename(install_path)

                    logger.info(f"{self.__repo_name} version {enc_version} installed at {install_path}")

                    # this is redundant (would get cleaned up automatically) but making
                    # explicit to support future debugging functionality that disables
                    # auto-delete for failed builds
                    build_tmpdir.cleanup()

                # STEP.3 : Cleanup
                tar_path.unlink()
            except FileNotFoundError:
                logger.exception(f"{self.__repo_name} version {enc_version} installation failed")
            except Exception:  # TODO: Presice
                logger.exception(f"{self.__repo_name} version {enc_version} installation failed")
            else:
                return True
        return False

    def uninstall(self, version: str) -> bool:
        """Uninstall a locally installed version.

        Args:
            version (str): The project's version to uninstall.

        Return:
            bool : True if the uninstall succeded.
        """
        if enc_version := self.__version_encode_callback(version):
            logger.debug(f"Uninstall version: {enc_version}")
            installed_versions: Final[Iterable[str]] = self._list_installed()

            if enc_version not in installed_versions:
                logger.error(f"{self.__repo_name} version {enc_version} not installed")
                return False

            install_dir: Final[Path] = self.__install_path / enc_version
            self.__uninstall_callback(version, install_dir)

            with TemporaryDirectory(prefix=f"remove_{enc_version}_", dir=self.__tmp_path) as remove_tmpdir:
                install_dir.rename(Path(remove_tmpdir) / enc_version)
                logger.info(f"{self.__repo_name} version {enc_version} has been uninstalled")
            return True
        return False

    def get_tool_path(self, version: str) -> Path | None:
        """Returns the absolute install path of the specified project version.

        Args:
            version (str) : The version of the tool to retrive the path too.

        Returns:
            Path : The path to the specified project version (if any).
        """
        if (enc_version := self.__version_encode_callback(version)) and enc_version in self._list_installed():
            return (self.__install_path / enc_version).resolve()
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
        return 0

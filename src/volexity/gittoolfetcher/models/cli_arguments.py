"""CLI Arguments data model."""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Final


class CLIArguments:
    """CLI Arguments data model."""

    def __init__(self, argv: list[str]) -> None:
        """Initialize a new instance of the CLI Arguments data model.

        Args:
            argv (list[str]) : Raw CLI arguments.
        """
        parser: Final[ArgumentParser] = ArgumentParser(prog=Path(argv[0]).name)

        parser.add_argument("repo", metavar="REPO_NAME", help="Name of the GitHub repo to manage.")

        parser.add_argument("-r", "--refresh", action="store_true", help="Refresh the available version cache.")
        parser.add_argument("-s", "--show", action="store_true", help="Show available project versions.")
        parser.add_argument(
            "-f", "--force", action="store_true", help="Force the installation, even if already installed."
        )

        parser.add_argument("-i", "--install", nargs="*", help="Install one or more project versions.")
        parser.add_argument("-u", "--uninstall", nargs="*", help="Uninstall one or more project versions.")

        parser.add_argument("-v", "--version", nargs="*", metavar="VERSION", help="Select a version of the project.")
        parser.add_argument("-l", "--list", action="store_true", help="List installed project versions.")
        parser.add_argument(
            "-e",
            "--execute",
            nargs="*",
            metavar="COMMAND",
            help="Execute a command with the selected version of the project.",
        )

        parsed_args: Final[Namespace] = parser.parse_args(argv[1:])

        if len(argv) <= 1:
            parser.print_usage()
            sys.exit()

        self.__repo: Final[str] = parsed_args.repo
        self.__refresh: Final[bool] = parsed_args.refresh
        self.__show: Final[bool] = parsed_args.show
        self.__force: Final[bool] = parsed_args.force

        self.__install: Final[list[str]] = (
            [ver for row in (vers.split(",") for vers in parsed_args.install) for ver in row]
            if parsed_args.install
            else []
        )

        self.__uninstall: Final[list[str]] = (
            [ver for row in (vers.split(",") for vers in parsed_args.uninstall) for ver in row]
            if parsed_args.uninstall
            else []
        )

        self.__version: Final[list[str]] = (
            [ver for row in (vers.split(",") for vers in parsed_args.version) for ver in row]
            if parsed_args.version
            else []
        )

        self.__list: Final[bool] = parsed_args.list
        self.__execute_command: Final[str | None] = parsed_args.execute[0] if parsed_args.execute else None
        self.__execute_args: Final[list[str]] = parsed_args.execute[1:] if parsed_args.execute else []

    @property
    def repo(self) -> str:
        """Returns the target repository name.

        Returns:
            str : Target GitHub repo name.
        """
        return self.__repo

    @property
    def refresh(self) -> bool:
        """Returns wether the available version cache should be refreshed.

        Returns:
            bool : Wether the available version cache should be refreshed.
        """
        return self.__refresh

    @property
    def show(self) -> bool:
        """Returns wether to list the available project versions.

        Returns:
            bool : Wether to list the available project versions.
        """
        return self.__show

    @property
    def force(self) -> bool:
        """Returns wether to force installation or not.

        Returns:
            bool : Wether to force installation or not.
        """
        return self.__force

    @property
    def install(self) -> list[str]:
        """Returns the list of project versions to install.

        Returns:
            list[str] : List of project versions to install.
        """
        return self.__install.copy()

    @property
    def uninstall(self) -> list[str]:
        """Returns the list of project versions to uninstall.

        Returns:
            list[str] : List of project versions to uninstall.
        """
        return self.__uninstall.copy()

    @property
    def version(self) -> list[str]:
        """Returns the selected project version.

        Returns:
            list[str] : List of selected project versions.
        """
        return self.__version.copy()

    @property
    def execute_command(self) -> str | None:
        """Returns the command to execute with the selected project version.

        Returns:
            str | None : Command to execute.
        """
        return self.__execute_command

    @property
    def execute_args(self) -> list[str]:
        """Returns the arguments of the command to execute.

        Returns:
            list[str] : List of cli arguments.
        """
        return self.__execute_args

    @property
    def list(self) -> bool:
        """Returns wether to list the installed project versions.

        Returns:
            bool : Wether to list the installed project versions.
        """
        return self.__list

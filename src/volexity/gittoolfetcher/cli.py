"""Implements the GitToolFetcher command line interface."""

import logging
import shutil
import sys
from cmd import Cmd
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .git_tool_fetcher import GitToolFetcher
from .models.cli_arguments import CLIArguments

if TYPE_CHECKING:
    from collections.abc import Iterable

logging.basicConfig()
logging.getLogger(__name__.rsplit(".", 1)[0]).setLevel(logging.INFO)

logger: Final[logging.Logger] = logging.getLogger(__name__)


def run_cli() -> None:
    """Implements the GitToolFetcher command line interface."""
    storage_base: Final[Path] = Path("storage")
    bin_path: Final[Path] = Path("bin")

    cmd: Final[Cmd] = Cmd()
    displaywidth: Final[int] = shutil.get_terminal_size().columns

    args: Final[CLIArguments] = CLIArguments(sys.argv)
    manager: Final[GitToolFetcher] = GitToolFetcher(args.repo, storage_base, bin_path=bin_path, display_progress=True)

    versions: Final[Iterable[str]] = args.version if len(args.version) else manager.list_installed()

    if args.refresh:
        manager.refresh()
    if args.show:
        cmd.columnize([*manager.list_available()], displaywidth)
    if args.list:
        cmd.columnize([*manager.list_installed()], displaywidth)
    if len(args.install):
        for version in args.install:
            manager.install(version, force=args.force)
    if len(args.uninstall):
        for version in args.uninstall:
            manager.uninstall(version)
    if args.execute_command:
        for version in versions:
            manager.run(version, args.execute_command, *args.execute_args)

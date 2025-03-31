from pathlib import Path
from typing import Final

from volexity.gittoolfetcher.git_tool_fetcher import GitToolFetcher


def test_version_listing() -> None:
    """Test that GitToolFetcher can successfuly retrieve the version list of a GitHub repository."""
    manager: Final[GitToolFetcher] = GitToolFetcher("golang/go", Path("storage"))
    version_list: Final[list[str]] = [*manager.list_available()]
    assert len(version_list) > 0

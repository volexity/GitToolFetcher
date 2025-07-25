"""Exception raised on Yara install/uninstall failure."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolProcessError(Exception):
    """Exception raised on Yara install/uninstall failure."""

    msg: str
    result: Any

"""."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolManagementResult:
    """.

    Members:
        version (str): .
        status (bool): .
        result (Any | None): The result component contains the custom data returned by the install callback.
                             It is None if the version was already installed.
    """

    version: str
    status: bool
    result: Any | None = None

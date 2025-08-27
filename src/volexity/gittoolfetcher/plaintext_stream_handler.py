"""PlainTextStreamHandler removes unwanted characters from the log output when not redirected to a tty."""

import re
from logging import LogRecord, StreamHandler
from typing import TextIO, cast


class PlainTextStreamHandler(StreamHandler):  # type: ignore[reportMissingTypeArgument]
    """PlainTextStreamHandler removes unwanted characters from the log output when not redirected to a tty."""

    def emit(self, record: LogRecord) -> None:
        """Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        msg: str
        try:
            msg = self.format(record)
            stream: TextIO = cast("TextIO", self.stream)  # type: ignore[reportUnknownMemberType]
            if not stream.isatty():
                msg = self._strip_special_chars(str(msg))
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def _strip_special_chars(self, msg: str) -> str:
        """Remove unwanted color & control characters from the message log.

        Args:
            msg (str): Message log to cleanup.

        Returns:
            str: The cleaned up log message.
        """
        no_color: str = re.sub(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]", "", msg)
        no_ctrl: str = re.sub(r"[^\x20-\x7E]", "", no_color)
        return no_ctrl

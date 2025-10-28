import sys
import traceback
from typing import Optional, cast


class DocumentPortalException(Exception):
    """
    Custom exception class for the Document Portal system.

    This class enhances standard exceptions by capturing detailed context,
    including the file, line number, and a full traceback string,
    making it easier for developers to debug issues.

    It supports wrapping existing exceptions or capturing the current
    exception context (`sys.exc_info`).
    """
    def __init__(self, error_message, error_details: Optional[object] = None):
        """
        Initializes a DocumentPortalException.

        Args:
            error_message (Union[str, BaseException]): The main error message or an
                existing exception object to wrap.
            error_details (Optional[object]): Optional object to provide exception context.
                This can be:
                - An instance of `sys` (e.g., to pass a module, though `None` is standard).
                - A `BaseException` instance (to explicitly wrap an exception).
                - `None` (default), which captures the exception currently being handled
                  via `sys.exc_info()`.
        """
        # Normalize message
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # Resolve exc_info (supports: sys module, Exception object, or current context)
        exc_type = exc_value = exc_tb = None
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            if hasattr(error_details, "exc_info"):  # e.g., sys module
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk to the last frame to report the most relevant location (where the error occurred)
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        # Store location details
        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Full pretty traceback (if available)
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        """
        Returns a compact, logger-friendly string representation of the exception.

        This includes the file, line number, message, and the full traceback if available.

        Returns:
            str: The string representation.
        """
        # Compact, logger-friendly message (no leading spaces)
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        """
        Returns an official string representation of the exception instance.

        Returns:
            str: The unambiguous string representation.
        """
        return f"DocumentPortalException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"
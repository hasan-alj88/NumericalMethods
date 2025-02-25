import logging


class LogAndReraise:
    """
    Context manager that logs an exception and re-raises it with an additional message.

    Example usage:
        with LogAndReraise("Could not symbolically differentiate", 
                           ValueError, "Please provide a derivative function manually"):
            f_sym = self.function(x_sym)
            derivative_sym = sympy.diff(f_sym, x_sym)
    """

    def __init__(self, log_message, exception_type=None, message=None, logger=None):
        self.log_message = log_message
        self.exception_type = exception_type
        self.message = message
        self.logger = logger or logging.getLogger()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Log the original exception with additional message
            self.logger.error(f"{self.log_message}: {exc_val}")

            # If new exception type is provided, raise that instead
            if self.exception_type is not None:
                raise self.exception_type(self.message) from exc_val

            # Otherwise, just let the original exception propagate
            return False
        return False
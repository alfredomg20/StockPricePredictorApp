class AppException(Exception):
    """Base exception class for all application-specific errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)
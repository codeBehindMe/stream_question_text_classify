class InvalidModelException(Exception):
    """Raises when the expected model object from the file system is not the type that was expected by the algorithm"""

    def __init__(self, message, *args):
        self.message = message

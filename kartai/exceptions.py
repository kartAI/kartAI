class CheckpointNotFoundException(Exception):
    """Exception for not finding a checkpoint file."""
    pass

class InvalidCheckpointException(Exception):
    """Exception for not beeing able to load a checkpoint."""
    pass

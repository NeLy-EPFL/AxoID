# Load subpackages in current namespace
from . import detection
from . import tracking
from . import utils

# Define list of importable objects under '*'
__all__ = ["detection", "tracking", "utils"]

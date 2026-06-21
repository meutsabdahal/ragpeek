from importlib.metadata import PackageNotFoundError, version

from ragpeek.decorators import trace
from ragpeek.logging import log_retrieval, log_generation, link_retrieval_to_generation
from ragpeek.config import TracerConfig
from ragpeek.serialization import serialize_trace

try:
    __version__ = version("ragpeek")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "trace",
    "log_retrieval",
    "log_generation",
    "link_retrieval_to_generation",
    "serialize_trace",
    "TracerConfig",
    "__version__",
]

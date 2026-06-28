"""Project-wide immutable constants and enumerations.

Physical/mathematical constants and enums are allowed in code (per the
submission guidelines); configurable values live in ``config/`` instead.
"""

from enum import Enum


class RunMode(str, Enum):
    """How a generation run is executed."""

    BASELINE = "baseline"
    AIRLLM = "airllm"


class QuantLevel(str, Enum):
    """Supported quantization levels (decreasing bit-width)."""

    FP16 = "fp16"
    Q8 = "q8"
    Q4 = "q4"
    Q2 = "q2"


# Unit conversions used by metrics/energy math.
BYTES_PER_MB: int = 1024 * 1024
SECONDS_PER_HOUR: int = 3600

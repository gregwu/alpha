"""Feature engineering: per-ticker time-series features plus
cross-sectional (relative-strength, sector, rank) features.

Every feature uses only information available at its own date.
"""

from .build import build_features, FEATURE_GROUPS  # noqa: F401

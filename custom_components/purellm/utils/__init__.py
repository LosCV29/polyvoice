"""Utility modules for PolyVoice."""
from .fuzzy_matching import (
    normalize_cover_query,
    find_entity_by_name,
    STOPWORDS,
    DEVICE_SYNONYMS,
)
from .parsing import (
    parse_entity_config,
    parse_list_config,
)
from .helpers import (
    get_friendly_name,
    format_human_readable_state,
    calculate_distance_miles,
    get_nested,
)

__all__ = [
    # Fuzzy matching
    "normalize_cover_query",
    "find_entity_by_name",
    "STOPWORDS",
    "DEVICE_SYNONYMS",
    # Parsing
    "parse_entity_config",
    "parse_list_config",
    # Helpers
    "get_friendly_name",
    "format_human_readable_state",
    "calculate_distance_miles",
    "get_nested",
]

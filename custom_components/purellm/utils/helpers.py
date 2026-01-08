"""General helper utilities for PolyVoice."""
from __future__ import annotations

from math import radians, cos, sin, asin, sqrt
from typing import Any


def get_friendly_name(entity_id: str, state) -> str:
    """Get the friendly name for an entity."""
    return state.attributes.get("friendly_name", entity_id.split(".")[-1])


def format_human_readable_state(entity_id: str, state: str) -> str:
    """Convert entity state to human-readable format.

    Examples:
        - binary_sensor.door: "on" -> "OPEN", "off" -> "CLOSED"
        - lock.front: "locked" -> "LOCKED"
        - light.kitchen: "on" -> "ON"
    """
    domain = entity_id.split(".")[0]

    if domain == "binary_sensor":
        if "door" in entity_id or "gate" in entity_id or "mailbox" in entity_id:
            return "OPEN" if state == "on" else "CLOSED"
        return "detected" if state == "on" else "clear"
    elif domain == "lock":
        return "LOCKED" if state == "locked" else "UNLOCKED"
    elif domain == "cover":
        return state.upper()
    elif domain in ("light", "switch", "fan"):
        return "ON" if state == "on" else "OFF"
    elif domain == "vacuum":
        return state.upper()
    else:
        return state.upper()


def calculate_distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in miles using Haversine formula."""
    lat1_r, lon1_r = radians(lat1), radians(lon1)
    lat2_r, lon2_r = radians(lat2), radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return 3956 * c  # Earth's radius in miles


def get_nested(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely get nested dict values.

    Example:
        data = {"content_urls": {"desktop": {"page": "http://..."}}}
        get_nested(data, "content_urls", "desktop", "page")  # -> "http://..."
        get_nested(data, "missing", "path", default="N/A")   # -> "N/A"
    """
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, {})
        else:
            return default
    return obj if obj != {} else default

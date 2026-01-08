"""Parsing utilities for PolyVoice configuration strings."""
from __future__ import annotations


def parse_entity_config(config_string: str) -> dict[str, str]:
    """Parse a config string like 'room:entity_id' into a dict.

    Example:
        Input: "living room:media_player.living_room\\nkitchen:media_player.kitchen"
        Output: {"living room": "media_player.living_room", "kitchen": "media_player.kitchen"}
    """
    result = {}
    if not config_string:
        return result
    for line in config_string.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip().lower()] = value.strip()
    return result


def parse_list_config(config_string: str) -> list[str]:
    """Parse a config string with one item per line into a list.

    Example:
        Input: "calendar.family\\ncalendar.work"
        Output: ["calendar.family", "calendar.work"]
    """
    if not config_string:
        return []
    return [line.strip() for line in config_string.strip().split("\n") if line.strip()]

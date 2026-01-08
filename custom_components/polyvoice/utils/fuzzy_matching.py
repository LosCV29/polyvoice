"""Fuzzy matching utilities for PolyVoice.

This module contains the KILLER FEATURE that makes PolyVoice superior to native HA intents:
- Synonym expansion (blind/shade/curtain/cover are interchangeable)
- Stopword removal
- Room abbreviation expansion
- Multi-pass entity search with priority queue
"""
from __future__ import annotations

import difflib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from homeassistant.helpers import entity_registry as er

_LOGGER = logging.getLogger(__name__)

# Stopwords to remove from queries (articles, possessives, prepositions)
STOPWORDS = frozenset([
    "the", "my", "a", "an", "in", "on", "at", "to", "for", "of",
    "please", "can", "you", "could", "would"
])

# Room abbreviations - expand these to full names
ROOM_ABBREVIATIONS = {
    "lr": "living room",
    "br": "bedroom",
    "mbr": "master bedroom",
    "mb": "master bedroom",
    "dr": "dining room",
    "fr": "family room",
    "kr": "kitchen",
    "ba": "bathroom",
    "bthrm": "bathroom",
    "bth": "bathroom",
    "gar": "garage",
    "ofc": "office",
    "lndry": "laundry",
    "bsmt": "basement",
    "atc": "attic",
}

# Synonym groups for fuzzy entity matching
# When searching for entities, these synonyms are treated as equivalent
COVER_SYNONYMS = frozenset([
    "blind", "blinds", "shade", "shades", "curtain", "curtains",
    "cover", "covers", "drape", "drapes", "roller", "rollers",
    "blackout", "blackouts", "sheer", "sheers"
])

# Extended synonyms for all device types - includes BOTH singular and plural
DEVICE_SYNONYMS = {
    # Cover synonyms (blind/shade/curtain/cover are interchangeable)
    "blind": ["shade", "curtain", "cover", "drape", "roller", "blinds", "shades"],
    "shade": ["blind", "curtain", "cover", "drape", "roller", "blinds", "shades"],
    "curtain": ["blind", "shade", "cover", "drape", "curtains", "blinds", "shades"],
    "cover": ["blind", "shade", "curtain", "drape", "covers", "blinds", "shades"],
    "drape": ["blind", "shade", "curtain", "cover", "drapes", "blinds", "shades"],
    "roller": ["blind", "shade", "curtain", "cover", "rollers", "blinds", "shades"],
    # Plural forms
    "blinds": ["shades", "curtains", "covers", "drapes", "rollers", "blind", "shade"],
    "shades": ["blinds", "curtains", "covers", "drapes", "rollers", "blind", "shade"],
    "curtains": ["blinds", "shades", "covers", "drapes", "curtain", "blind", "shade"],
    "covers": ["blinds", "shades", "curtains", "drapes", "cover", "blind", "shade"],
    "drapes": ["blinds", "shades", "curtains", "covers", "drape", "blind", "shade"],
    "rollers": ["blinds", "shades", "curtains", "covers", "roller", "blind", "shade"],
    # Light synonyms
    "light": ["lamp", "bulb", "fixture", "lights", "lamps"],
    "lights": ["lamps", "bulbs", "fixtures", "light", "lamp"],
    "lamp": ["light", "bulb", "lamps", "lights"],
    "lamps": ["lights", "bulbs", "lamp", "light"],
    "bulb": ["light", "lamp", "bulbs"],
    "bulbs": ["lights", "lamps", "bulb"],
    # Lock synonyms
    "lock": ["deadbolt", "latch", "locks"],
    "locks": ["deadbolts", "latches", "lock"],
    # Climate synonyms
    "thermostat": ["climate", "hvac", "ac", "heater", "temp", "temperature"],
    "ac": ["air conditioner", "air conditioning", "climate", "thermostat", "cooling"],
    "air conditioner": ["ac", "climate", "thermostat"],
    "heater": ["heating", "thermostat", "climate", "heat"],
    "heat": ["heater", "heating", "thermostat"],
    # Fan synonyms
    "fan": ["ceiling fan", "exhaust fan", "fans"],
    "fans": ["ceiling fans", "fan"],
    "ceiling fan": ["fan"],
    # Door synonyms - door and gate are NOT synonyms (different devices)
    "door": ["doors"],
    "doors": ["door"],
    "gate": ["gates"],
    "gates": ["gate"],
    "garage": ["garage door"],
    "garage door": ["garage"],
    # TV/Media synonyms
    "tv": ["television", "telly", "screen"],
    "television": ["tv", "telly"],
    "speaker": ["media player", "sonos", "echo", "speakers"],
    "speakers": ["media players", "speaker"],
    # Switch synonyms
    "switch": ["outlet", "plug", "switches"],
    "switches": ["outlets", "plugs", "switch"],
    "outlet": ["switch", "plug", "outlets"],
    "plug": ["outlet", "switch", "plugs"],
}


def _strip_stopwords(query: str) -> str:
    """Remove stopwords from query for better matching."""
    words = query.lower().split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered) if filtered else query.lower()


# Device type words - these are the ONLY words that should be fuzzy matched
# Location words must match exactly to prevent cross-room confusion
DEVICE_TYPE_WORDS = frozenset([
    # Covers
    "blind", "blinds", "shade", "shades", "curtain", "curtains", "cover", "covers",
    "drape", "drapes", "roller", "rollers", "blackout", "sheer",
    # Lights
    "light", "lights", "lamp", "lamps", "bulb", "bulbs", "fixture", "fixtures",
    "chandelier", "sconce", "spotlight", "strip",
    # Doors/Locks
    "door", "doors", "gate", "gates", "lock", "locks", "deadbolt", "latch",
    # Climate
    "thermostat", "climate", "hvac", "ac", "heater", "heating", "cooling",
    # Fans
    "fan", "fans", "ceiling", "exhaust",
    # Media
    "tv", "television", "speaker", "speakers", "media", "player", "sonos", "echo",
    # Switches/Outlets
    "switch", "switches", "outlet", "outlets", "plug", "plugs",
    # Sensors
    "sensor", "sensors", "motion", "contact", "temperature", "humidity",
    # Other
    "vacuum", "robot", "camera", "doorbell", "siren", "alarm",
])

# Location words - must match EXACTLY, never fuzzy
LOCATION_WORDS = frozenset([
    "living", "dining", "bed", "bedroom", "bath", "bathroom", "kitchen",
    "front", "back", "side", "rear", "main", "upper", "lower", "upstairs", "downstairs",
    "master", "guest", "kids", "kid", "baby", "nursery", "child", "children",
    "office", "study", "den", "library", "home",
    "garage", "carport", "driveway", "yard", "garden", "patio", "porch", "deck", "balcony",
    "backyard", "frontyard", "sideyard",  # Compound yard words
    "hallway", "hall", "foyer", "entry", "entryway", "mudroom", "laundry", "utility",
    "basement", "attic", "closet", "pantry", "storage",
    "pool", "spa", "gym", "theater", "media", "game", "play",
    "north", "south", "east", "west", "left", "right", "center", "middle",
    "first", "second", "third", "1st", "2nd", "3rd",
    "room",  # Generic room word
])


def _extract_device_type(words: set[str]) -> set[str]:
    """Extract device type words from a set of words."""
    return words & DEVICE_TYPE_WORDS


def _extract_location(words: set[str]) -> set[str]:
    """Extract location words from a set of words."""
    return words & LOCATION_WORDS


def _is_word_match(query: str, target: str) -> bool:
    """Check if query and target match with EXACT location and fuzzy device type.

    STRICT MATCHING RULES:
    1. Location words (living, kitchen, front, back, etc.) must match EXACTLY
    2. Device type words (light, door, shade, etc.) can match via synonyms
    3. This prevents "living room shade" from matching "dining room light"

    Args:
        query: User's search query
        target: Entity name/alias to check

    Returns:
        True if locations match exactly AND device types are compatible
    """
    query_words = set(query.lower().split())
    target_words = set(target.lower().split())

    # Remove stopwords
    stopwords = {"the", "a", "an", "my", "in", "on", "at", "to", "for", "of", "is", "are"}
    query_words -= stopwords
    target_words -= stopwords

    # Extract location and device type
    query_locations = _extract_location(query_words)
    target_locations = _extract_location(target_words)
    query_devices = _extract_device_type(query_words)
    target_devices = _extract_device_type(target_words)

    # RULE 1: If query has location words, target MUST have the SAME locations
    if query_locations:
        if query_locations != target_locations:
            return False

    # RULE 2: Device types must match (exact or synonym)
    if query_devices and target_devices:
        # Check for exact device match
        if query_devices & target_devices:
            return True
        # Check for synonym match
        for q_device in query_devices:
            if q_device in DEVICE_SYNONYMS:
                synonyms = set(DEVICE_SYNONYMS[q_device])
                if synonyms & target_devices:
                    return True
        return False

    # RULE 3: If no device types extracted, check for remaining word overlap
    query_other = query_words - query_locations - query_devices
    target_other = target_words - target_locations - target_devices

    # Need some meaningful overlap
    if query_other and target_other:
        return bool(query_other & target_other)

    # If query has locations that match and no device type specified, allow
    if query_locations and query_locations == target_locations:
        return True

    return False


def _expand_abbreviations(query: str) -> str:
    """Expand room abbreviations like 'lr' -> 'living room'."""
    words = query.lower().split()
    expanded = []
    for word in words:
        if word in ROOM_ABBREVIATIONS:
            expanded.append(ROOM_ABBREVIATIONS[word])
        else:
            expanded.append(word)
    return " ".join(expanded)


def normalize_cover_query(query: str) -> list[str]:
    """Generate query variations for comprehensive fuzzy matching.

    This is the KILLER FEATURE that makes PolyVoice superior to native HA intents.

    For "the living room blinds" generates:
    - "the living room blinds" (original)
    - "living room blinds" (stopwords stripped)
    - "living room shades" (synonym substitution)
    - "living room shade" (singular form)
    - "living room curtains", "living room covers", etc.

    Works for all device types: blinds/shades, lights/lamps, locks, etc.
    """
    query_lower = query.lower().strip()
    variations = set()  # Use set to avoid duplicates

    # Start with original query
    variations.add(query_lower)

    # Strip stopwords version
    stripped = _strip_stopwords(query_lower)
    variations.add(stripped)

    # Expand abbreviations
    expanded = _expand_abbreviations(query_lower)
    variations.add(expanded)
    expanded_stripped = _strip_stopwords(expanded)
    variations.add(expanded_stripped)

    # Generate synonym variations for all base queries
    base_queries = list(variations)
    for base_query in base_queries:
        words = base_query.split()
        for i, word in enumerate(words):
            if word in DEVICE_SYNONYMS:
                # Generate variations with each synonym
                for replacement in DEVICE_SYNONYMS[word]:
                    new_words = words.copy()
                    new_words[i] = replacement
                    variation = " ".join(new_words)
                    variations.add(variation)

    # Return as list, with stripped versions first (more likely to match)
    result = list(variations)
    # Prioritize shorter queries (stopwords stripped) as they're more likely to match
    result.sort(key=len)
    return result


def find_entity_by_name(
    hass: HomeAssistant,
    query: str,
    device_aliases: dict[str, str]
) -> tuple[str | None, str | None]:
    """Search for entity using device aliases first, then fall back to HA entity registry aliases.

    Returns (entity_id, friendly_name) or (None, None) if not found.

    OPTIMIZED: Single-pass search with priority queue instead of 6 separate passes.
    ENHANCED: Supports cover synonyms (blind/shade/curtain/cover are interchangeable)
    """
    # ALWAYS try the exact original query first before any variations
    result = _find_entity_by_query(hass, query, device_aliases)
    if result[0] is not None:
        return result

    # Generate synonym variations for cover-related queries
    query_variations = normalize_cover_query(query)

    for query_var in query_variations:
        # Skip the original query since we already tried it
        if query_var.lower() == query.lower():
            continue
        result = _find_entity_by_query(hass, query_var, device_aliases)
        if result[0] is not None:
            return result

    return (None, None)


def _find_entity_by_query(
    hass: HomeAssistant,
    query: str,
    device_aliases: dict[str, str]
) -> tuple[str | None, str | None]:
    """Internal entity search for a single query string."""
    query_lower = query.lower().strip()

    # PRIORITY 1: Exact match in configured device aliases (O(1) dict lookup)
    if query_lower in device_aliases:
        entity_id = device_aliases[query_lower]
        state = hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", query) if state else query
        return (entity_id, friendly_name)

    # Collect partial matches with priorities (lower = better)
    partial_matches: list[tuple[int, str, str]] = []  # (priority, entity_id, name)

    # PRIORITY 2: Partial match in device aliases (word boundary check)
    for alias, entity_id in device_aliases.items():
        if _is_word_match(query_lower, alias):
            state = hass.states.get(entity_id)
            friendly_name = state.attributes.get("friendly_name", alias) if state else alias
            return (entity_id, friendly_name)  # Return immediately for device aliases

    # Single pass through entity registry for aliases + friendly names
    ent_reg = er.async_get(hass)
    all_states = {s.entity_id: s for s in hass.states.async_all()}  # Cache states lookup

    for entity_entry in ent_reg.entities.values():
        state = all_states.get(entity_entry.entity_id)
        friendly_name = state.attributes.get("friendly_name", "") if state else ""

        # Check entity registry aliases
        if entity_entry.aliases:
            for alias in entity_entry.aliases:
                alias_lower = alias.lower()
                if alias_lower == query_lower:
                    return (entity_entry.entity_id, friendly_name or alias)  # PRIORITY 3: Exact alias
                if _is_word_match(query_lower, alias_lower):
                    partial_matches.append((4, entity_entry.entity_id, friendly_name or alias))

        # Check friendly name (word-boundary partial matching)
        if friendly_name:
            fn_lower = friendly_name.lower()
            if fn_lower == query_lower:
                partial_matches.append((5, entity_entry.entity_id, friendly_name))  # PRIORITY 5: Exact friendly
            elif _is_word_match(query_lower, fn_lower):
                partial_matches.append((6, entity_entry.entity_id, friendly_name))  # PRIORITY 6: Partial friendly

    # Check states not in entity registry (rare but possible)
    for entity_id, state in all_states.items():
        if entity_id not in {e.entity_id for e in ent_reg.entities.values()}:
            friendly_name = state.attributes.get("friendly_name", "")
            if friendly_name:
                fn_lower = friendly_name.lower()
                if fn_lower == query_lower:
                    partial_matches.append((5, entity_id, friendly_name))
                elif _is_word_match(query_lower, fn_lower):
                    partial_matches.append((6, entity_id, friendly_name))

    # Return best match by priority
    if partial_matches:
        partial_matches.sort(key=lambda x: x[0])
        return (partial_matches[0][1], partial_matches[0][2])

    # PRIORITY 7: Location-aware fuzzy matching (catches typos in device type only)
    # Only match entities that have the SAME location words
    query_words = set(query_lower.split()) - {"the", "a", "an", "my", "in", "on", "at", "to", "for", "of"}
    query_locations = _extract_location(query_words)
    query_devices = _extract_device_type(query_words)

    # Only try difflib if we have a device type to match
    if query_devices:
        candidates = []
        for entity_id, state in all_states.items():
            friendly_name = state.attributes.get("friendly_name", "")
            if not friendly_name:
                continue

            fn_lower = friendly_name.lower()
            fn_words = set(fn_lower.split()) - {"the", "a", "an", "my"}
            fn_locations = _extract_location(fn_words)
            fn_devices = _extract_device_type(fn_words)

            # Location must match exactly (or both have no location)
            if query_locations != fn_locations:
                continue

            # Must have a device type to compare
            if not fn_devices:
                continue

            # Check if device types are compatible (exact or synonym)
            device_match = False
            if query_devices & fn_devices:
                device_match = True
            else:
                for q_device in query_devices:
                    if q_device in DEVICE_SYNONYMS:
                        if set(DEVICE_SYNONYMS[q_device]) & fn_devices:
                            device_match = True
                            break

            if device_match:
                candidates.append((entity_id, friendly_name, fn_lower))

        # If we have location-matched candidates, use difflib on those only
        if candidates:
            candidate_names = [c[2] for c in candidates]
            close_matches = difflib.get_close_matches(query_lower, candidate_names, n=1, cutoff=0.6)
            if close_matches:
                matched_name = close_matches[0]
                for entity_id, friendly_name, fn_lower in candidates:
                    if fn_lower == matched_name:
                        return (entity_id, friendly_name)

    return (None, None)

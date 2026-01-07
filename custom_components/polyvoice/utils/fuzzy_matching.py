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


def _is_word_match(query: str, target: str) -> bool:
    """Check if query and target share meaningful words (not just substrings).

    Prevents false positives like:
    - "back door" matching "ac" (substring in word)
    - "living room shade" matching "dining room light" (only "room" overlaps)

    Args:
        query: User's search query
        target: Entity name/alias to check

    Returns:
        True if there's a meaningful word-level match
    """
    query_words = set(query.lower().split())
    target_words = set(target.lower().split())

    # Common words that shouldn't trigger matches by themselves
    common_words = {"room", "light", "lights", "the", "a", "an", "my", "in", "on", "at", "to", "for", "of"}

    # Get meaningful words (exclude common words)
    query_meaningful = query_words - common_words
    target_meaningful = target_words - common_words

    # Check for meaningful word overlap (not just "room" or "light")
    meaningful_overlap = query_meaningful & target_meaningful
    if meaningful_overlap:
        return True

    # Check if multi-word target is contained in query or vice versa
    query_lower = query.lower()
    target_lower = target.lower()

    # Require minimum length for substring matching to avoid false positives
    # like "ac" in "back" or "at" in "thermostat"
    min_substr_len = 4

    if len(target_lower) >= min_substr_len and target_lower in query_lower:
        return True
    if len(query_lower) >= min_substr_len and query_lower in target_lower:
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
    # Generate synonym variations for cover-related queries
    query_variations = normalize_cover_query(query)

    for query_var in query_variations:
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

    _LOGGER.debug("Fuzzy search: query='%s'", query_lower)

    # PRIORITY 1: Exact match in configured device aliases (O(1) dict lookup)
    if query_lower in device_aliases:
        entity_id = device_aliases[query_lower]
        state = hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", query) if state else query
        _LOGGER.debug("Fuzzy match P1: exact alias -> %s", entity_id)
        return (entity_id, friendly_name)

    # Collect partial matches with priorities (lower = better)
    partial_matches: list[tuple[int, str, str]] = []  # (priority, entity_id, name)

    # PRIORITY 2: Partial match in device aliases (word boundary check)
    for alias, entity_id in device_aliases.items():
        if _is_word_match(query_lower, alias):
            state = hass.states.get(entity_id)
            friendly_name = state.attributes.get("friendly_name", alias) if state else alias
            _LOGGER.debug("Fuzzy match P2: partial alias '%s' -> %s", alias, entity_id)
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
                    _LOGGER.debug("Fuzzy match P3: exact registry alias '%s' -> %s", alias, entity_entry.entity_id)
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
        _LOGGER.debug("Fuzzy match: best partial (P%d) '%s' -> %s", partial_matches[0][0], partial_matches[0][2], partial_matches[0][1])
        return (partial_matches[0][1], partial_matches[0][2])

    # PRIORITY 7: Generic fuzzy matching using difflib (catches typos and close matches)
    # Only if no other matches found - this is the last resort
    all_friendly_names = []
    name_to_entity: dict[str, tuple[str, str]] = {}
    for entity_id, state in all_states.items():
        friendly_name = state.attributes.get("friendly_name", "")
        if friendly_name:
            fn_lower = friendly_name.lower()
            all_friendly_names.append(fn_lower)
            name_to_entity[fn_lower] = (entity_id, friendly_name)

    # Find close matches with 60% similarity threshold
    close_matches = difflib.get_close_matches(query_lower, all_friendly_names, n=1, cutoff=0.6)
    if close_matches:
        matched_name = close_matches[0]
        entity_id, friendly_name = name_to_entity[matched_name]
        _LOGGER.debug("Fuzzy match P7: difflib '%s' -> '%s' (%s)", query_lower, friendly_name, entity_id)
        return (entity_id, friendly_name)

    _LOGGER.debug("Fuzzy match: no match for '%s'", query_lower)
    return (None, None)

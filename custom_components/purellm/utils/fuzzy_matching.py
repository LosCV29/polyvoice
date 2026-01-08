"""Fuzzy matching utilities for PolyVoice.

This module handles device name matching with:
- Synonym expansion (blind/shade/curtain/cover are interchangeable)
- Stopword removal
- Direct entity matching (NO room fuzzy logic - causes cross-room confusion)
- LRU caching for name→entity_id resolution (states fetched fresh each time)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from homeassistant.helpers import entity_registry as er

_LOGGER = logging.getLogger(__name__)

# Module-level cache for entity lookups (name→entity_id only, NOT states)
# This caches the resolution of device names to entity IDs
# States are always fetched fresh via hass.states.get()
_entity_cache: dict[str, tuple[str | None, str | None]] = {}
_cache_aliases_hash: int | None = None

# Stopwords to remove from queries (articles, possessives, prepositions)
STOPWORDS = frozenset([
    "the", "my", "a", "an", "in", "on", "at", "to", "for", "of",
    "please", "can", "you", "could", "would"
])

# Synonym groups for fuzzy entity matching
# When searching for entities, these synonyms are treated as equivalent
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
    # Door synonyms
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


def normalize_cover_query(query: str) -> list[str]:
    """Generate query variations with device synonyms.

    For "living room blinds" generates:
    - "living room blinds" (original)
    - "living room shades" (synonym substitution)
    - "living room curtains", "living room covers", etc.
    """
    query_lower = query.lower().strip()
    variations = set()

    # Start with original and stopwords-stripped
    variations.add(query_lower)
    stripped = _strip_stopwords(query_lower)
    variations.add(stripped)

    # Generate synonym variations
    for base_query in [query_lower, stripped]:
        words = base_query.split()
        for i, word in enumerate(words):
            if word in DEVICE_SYNONYMS:
                for replacement in DEVICE_SYNONYMS[word]:
                    new_words = words.copy()
                    new_words[i] = replacement
                    variations.add(" ".join(new_words))

    # Return shorter (stopwords stripped) first
    result = list(variations)
    result.sort(key=len)
    return result


def _words_match(query: str, target: str) -> bool:
    """Check if all query words appear in target (simple containment check)."""
    query_words = set(query.lower().split()) - STOPWORDS
    target_words = set(target.lower().split()) - STOPWORDS

    if not query_words:
        return False

    # All query words must be in target
    return query_words <= target_words


def clear_entity_cache() -> None:
    """Clear the entity lookup cache. Call when entities/aliases change."""
    global _entity_cache, _cache_aliases_hash
    _entity_cache.clear()
    _cache_aliases_hash = None
    _LOGGER.debug("Entity lookup cache cleared")


def find_entity_by_name(
    hass: HomeAssistant,
    query: str,
    device_aliases: dict[str, str]
) -> tuple[str | None, str | None]:
    """Search for entity by name - with caching for name→entity_id resolution.

    Returns (entity_id, friendly_name) or (None, None) if not found.
    Note: Only caches entity_id resolution, states are always fetched fresh.
    """
    global _entity_cache, _cache_aliases_hash

    # Invalidate cache if aliases changed
    aliases_hash = hash(tuple(sorted(device_aliases.items()))) if device_aliases else 0
    if _cache_aliases_hash != aliases_hash:
        _entity_cache.clear()
        _cache_aliases_hash = aliases_hash

    # Check cache first
    cache_key = query.lower().strip()
    if cache_key in _entity_cache:
        return _entity_cache[cache_key]

    # Try original query first
    result = _find_entity_by_query(hass, query, device_aliases)
    if result[0] is not None:
        _entity_cache[cache_key] = result
        return result

    # Try synonym variations
    for query_var in normalize_cover_query(query):
        if query_var.lower() == query.lower():
            continue
        result = _find_entity_by_query(hass, query_var, device_aliases)
        if result[0] is not None:
            _entity_cache[cache_key] = result
            return result

    # Cache negative result too (avoids repeated failed lookups)
    _entity_cache[cache_key] = (None, None)
    return (None, None)


def _find_entity_by_query(
    hass: HomeAssistant,
    query: str,
    device_aliases: dict[str, str]
) -> tuple[str | None, str | None]:
    """Internal entity search - direct matching only, no fuzzy logic."""
    query_lower = query.lower().strip()

    # PRIORITY 1: Exact match in configured device aliases
    if query_lower in device_aliases:
        entity_id = device_aliases[query_lower]
        state = hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", query) if state else query
        return (entity_id, friendly_name)

    # PRIORITY 2: Partial match in device aliases (all words present)
    for alias, entity_id in device_aliases.items():
        if _words_match(query_lower, alias) or _words_match(alias, query_lower):
            state = hass.states.get(entity_id)
            friendly_name = state.attributes.get("friendly_name", alias) if state else alias
            return (entity_id, friendly_name)

    # Single pass through entity registry
    ent_reg = er.async_get(hass)
    all_states = {s.entity_id: s for s in hass.states.async_all()}

    partial_matches: list[tuple[int, str, str]] = []

    for entity_entry in ent_reg.entities.values():
        state = all_states.get(entity_entry.entity_id)
        friendly_name = state.attributes.get("friendly_name", "") if state else ""

        # PRIORITY 3: Exact match on entity registry alias
        if entity_entry.aliases:
            for alias in entity_entry.aliases:
                if alias.lower() == query_lower:
                    return (entity_entry.entity_id, friendly_name or alias)
                if _words_match(query_lower, alias.lower()):
                    partial_matches.append((4, entity_entry.entity_id, friendly_name or alias))

        # PRIORITY 5: Exact match on friendly name
        if friendly_name:
            fn_lower = friendly_name.lower()
            if fn_lower == query_lower:
                partial_matches.append((5, entity_entry.entity_id, friendly_name))
            elif _words_match(query_lower, fn_lower):
                partial_matches.append((6, entity_entry.entity_id, friendly_name))

    # Check states not in entity registry
    for entity_id, state in all_states.items():
        if entity_id not in {e.entity_id for e in ent_reg.entities.values()}:
            friendly_name = state.attributes.get("friendly_name", "")
            if friendly_name:
                fn_lower = friendly_name.lower()
                if fn_lower == query_lower:
                    partial_matches.append((5, entity_id, friendly_name))
                elif _words_match(query_lower, fn_lower):
                    partial_matches.append((6, entity_id, friendly_name))

    # Return best match
    if partial_matches:
        partial_matches.sort(key=lambda x: x[0])
        return (partial_matches[0][1], partial_matches[0][2])

    return (None, None)

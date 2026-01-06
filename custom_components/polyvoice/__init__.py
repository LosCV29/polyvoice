"""The PolyVoice integration."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION, Platform.UPDATE]


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the PolyVoice component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry to current version."""
    _LOGGER.info("Migrating PolyVoice from version %s", entry.version)
    
    if entry.version < 2:
        hass.config_entries.async_update_entry(entry, version=2)
        _LOGGER.info("Migration to version 2 successful")
    
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up PolyVoice from a config entry."""
    import time
    import traceback

    hass.data.setdefault(DOMAIN, {})

    config = {**entry.data, **entry.options}
    hass.data[DOMAIN][entry.entry_id] = {"config": config}

    # ========== GLOBAL MEDIA_NEXT_TRACK MONITOR ==========
    # This catches ALL calls to media_next_track from ANY source
    async def monitor_service_calls(call):
        """Monitor all service calls to catch double-skip source."""
        if call.domain == "media_player" and call.service in ("media_next_track", "media_previous_track"):
            stack = ''.join(traceback.format_stack()[-8:-1])
            _LOGGER.warning("╔══════════════════════════════════════════════════════════╗")
            _LOGGER.warning("║  🔍 GLOBAL MONITOR: %s.%s CALLED  ║", call.domain, call.service)
            _LOGGER.warning("╚══════════════════════════════════════════════════════════╝")
            _LOGGER.warning("MONITOR: Timestamp: %.3f", time.time())
            _LOGGER.warning("MONITOR: Target: %s", call.data.get("entity_id", "unknown"))
            _LOGGER.warning("MONITOR: Call stack:\n%s", stack)

    # Register listener for ALL service calls
    hass.bus.async_listen("call_service", monitor_service_calls)
    _LOGGER.warning("🔍 PolyVoice: GLOBAL media_next_track monitor ACTIVE - will catch ALL skip calls")
    # ========================================================

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(
        entry.add_update_listener(_async_update_listener)
    )

    _LOGGER.info("PolyVoice setup complete")
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle config entry updates - reload the integration."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
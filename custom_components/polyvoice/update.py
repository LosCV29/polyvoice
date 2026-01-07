"""Update entity for PolyVoice integration."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import aiohttp

from homeassistant.components.update import (
    UpdateDeviceClass,
    UpdateEntity,
    UpdateEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers import device_registry as dr

from .const import DOMAIN, get_version

_LOGGER = logging.getLogger(__name__)

GITHUB_REPO = "LosCV29/polyvoice"
SCAN_INTERVAL = timedelta(hours=4)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up PolyVoice update entity."""
    async_add_entities([PolyVoiceUpdateEntity(hass, entry)])


class PolyVoiceUpdateEntity(UpdateEntity):
    """Update entity for PolyVoice."""

    _attr_has_entity_name = True
    _attr_name = "Update"
    _attr_device_class = UpdateDeviceClass.FIRMWARE
    _attr_supported_features = UpdateEntityFeature.INSTALL

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the update entity."""
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_update"
        self._installed_version: str | None = get_version()
        self._latest_version: str | None = None
        self._release_url: str | None = None
        self._release_notes: str | None = None

    @property
    def installed_version(self) -> str | None:
        """Return the installed version."""
        return self._installed_version

    @property
    def latest_version(self) -> str | None:
        """Return the latest version."""
        return self._latest_version

    @property
    def release_url(self) -> str | None:
        """Return the release URL."""
        return self._release_url

    @property
    def release_summary(self) -> str | None:
        """Return the release notes."""
        return self._release_notes

    async def async_added_to_hass(self) -> None:
        """Update device registry sw_version when entity is added."""
        await super().async_added_to_hass()
        # Force update device registry with current version
        # This ensures sw_version stays in sync after updates
        current_version = get_version()
        device_registry = dr.async_get(self.hass)
        device = device_registry.async_get_device(
            identifiers={(DOMAIN, self._entry.entry_id)}
        )
        if device and device.sw_version != current_version:
            device_registry.async_update_device(
                device.id, sw_version=current_version
            )
            _LOGGER.info(
                "Updated PolyVoice device version from %s to %s",
                device.sw_version,
                current_version,
            )

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "PolyVoice",
            "manufacturer": "LosCV29",
            "model": "Voice Assistant",
            "sw_version": get_version(),
        }

    async def async_update(self) -> None:
        """Check GitHub for the latest release."""
        try:
            session = async_get_clientsession(self.hass)
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    tag = data.get("tag_name", "")
                    # Remove 'v' prefix if present
                    self._latest_version = tag.lstrip("v")
                    self._release_url = data.get("html_url")

                    # Get release notes (truncate if too long)
                    body = data.get("body", "")
                    if len(body) > 500:
                        self._release_notes = body[:497] + "..."
                    else:
                        self._release_notes = body

                    _LOGGER.debug(
                        "PolyVoice update check: installed=%s, latest=%s",
                        self._installed_version,
                        self._latest_version
                    )
                else:
                    _LOGGER.warning("GitHub API returned status %s", response.status)
        except Exception as err:
            _LOGGER.error("Failed to check for updates: %s", err)

    async def async_install(
        self, version: str | None, backup: bool, **kwargs: Any
    ) -> None:
        """Install the update via HACS."""
        # Trigger HACS update if available
        try:
            # Try to call HACS download service
            await self.hass.services.async_call(
                "hacs",
                "download",
                {
                    "repository": GITHUB_REPO,
                },
                blocking=True,
            )
            _LOGGER.info("PolyVoice update triggered via HACS")

            # Show notification to restart
            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "PolyVoice Updated",
                    "message": f"PolyVoice has been updated to version {version}. Please restart Home Assistant to complete the update.",
                    "notification_id": "polyvoice_update",
                },
            )
        except Exception as err:
            _LOGGER.error("Failed to trigger HACS update: %s", err)
            # Fallback: show notification with instructions
            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "PolyVoice Update Available",
                    "message": f"Please update PolyVoice to version {version} via HACS:\n\n1. Go to HACS → Integrations\n2. Find PolyVoice\n3. Click Update\n4. Restart Home Assistant",
                    "notification_id": "polyvoice_update",
                },
            )

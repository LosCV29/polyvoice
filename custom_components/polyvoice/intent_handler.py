"""Intent handler for PolyVoice.

This module intercepts native Home Assistant intents and routes them
through PolyVoice for LLM-controlled processing.
"""
from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

from homeassistant.helpers.intent import IntentHandler, IntentResponse
from homeassistant.exceptions import HomeAssistantError

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.intent import Intent

_LOGGER = logging.getLogger(__name__)

# Intents that should be handled by PolyVoice instead of native HA
INTERCEPTED_INTENTS = frozenset([
    "HassTurnOn",
    "HassTurnOff",
    "HassToggle",
    "HassLightSet",
    "HassOpenCover",
    "HassCloseCover",
    "HassSetPosition",
])

# Patterns to extract device names from intent slot structures
SLOT_PATTERNS = [
    # Nested value structure: {"value": {"text": "device name"}}
    r'"text"\s*:\s*"([^"]+)"',
    # Simple value structure: {"value": "device name"}
    r'"value"\s*:\s*"([^"]+)"',
]


class PolyVoiceIntentHandler(IntentHandler):
    """Handler that intercepts native HA intents and routes them to PolyVoice."""

    def __init__(
        self,
        intent_type: str,
        hass: "HomeAssistant",
        conversation_entity_id: str,
        original_handler: IntentHandler | None = None,
    ):
        """Initialize the intent handler.

        Args:
            intent_type: The intent type to handle (e.g., "HassTurnOn")
            hass: Home Assistant instance
            conversation_entity_id: Entity ID of the PolyVoice conversation entity
            original_handler: The original handler to fall back to if needed
        """
        self._intent_type = intent_type
        self._hass = hass
        self._conversation_entity_id = conversation_entity_id
        self._original_handler = original_handler

    @property
    def intent_type(self) -> str:
        """Return the intent type this handler handles."""
        return self._intent_type

    async def async_handle(self, intent: "Intent") -> IntentResponse:
        """Handle the intent by routing to PolyVoice.

        Args:
            intent: The intent to handle

        Returns:
            IntentResponse with the result
        """
        _LOGGER.info(
            "PolyVoice intercepted %s intent - routing to LLM for intelligent handling",
            self._intent_type
        )

        # Extract the command from intent slots
        command = self._extract_command_from_slots(intent.slots)

        if not command:
            # Can't extract command - fall back to original handler if available
            _LOGGER.warning("Could not extract command from %s slots, falling back", self._intent_type)
            if self._original_handler:
                return await self._original_handler.async_handle(intent)
            raise HomeAssistantError(f"Could not process {self._intent_type} intent")

        _LOGGER.info("PolyVoice processing command: '%s'", command)

        try:
            # Get the conversation entity
            entity = self._hass.data.get("conversation", {}).get("entities", {}).get(self._conversation_entity_id)

            if entity and hasattr(entity, "async_process_intercepted"):
                # Use dedicated method for intercepted intents
                result = await entity.async_process_intercepted(command)
                response = intent.create_response()
                response.async_set_speech(result)
                return response
            else:
                _LOGGER.warning("PolyVoice entity not found or doesn't support interception")
                if self._original_handler:
                    return await self._original_handler.async_handle(intent)
                raise HomeAssistantError("PolyVoice entity not available")

        except Exception as err:
            _LOGGER.error("Error processing intercepted intent: %s", err, exc_info=True)
            if self._original_handler:
                return await self._original_handler.async_handle(intent)
            raise HomeAssistantError(f"Error processing intent: {err}")

    def _extract_command_from_slots(self, slots: dict[str, Any]) -> str | None:
        """Extract a natural language command from intent slots.

        The slot structure can vary:
        - {"name": {"value": "device name"}}
        - {"name": {"value": {"text": "device name", ...}}}

        Args:
            slots: Intent slots dictionary

        Returns:
            Extracted command string or None
        """
        if not slots:
            return None

        device_name = None

        # Try to extract from 'name' slot
        name_slot = slots.get("name", {})
        if isinstance(name_slot, dict):
            value = name_slot.get("value")
            if isinstance(value, dict):
                # Nested structure
                device_name = value.get("text") or value.get("name")
            elif isinstance(value, str):
                device_name = value

        # Fall back to checking all slots
        if not device_name:
            for slot_name, slot_data in slots.items():
                if isinstance(slot_data, dict):
                    value = slot_data.get("value")
                    if isinstance(value, dict):
                        device_name = value.get("text") or value.get("name")
                    elif isinstance(value, str):
                        device_name = value
                    if device_name:
                        break

        # Last resort: try regex on the string representation
        if not device_name:
            slots_str = str(slots)
            for pattern in SLOT_PATTERNS:
                match = re.search(pattern, slots_str)
                if match:
                    device_name = match.group(1)
                    break

        if not device_name:
            return None

        # Reconstruct a natural language command
        action_map = {
            "HassTurnOn": "turn on",
            "HassTurnOff": "turn off",
            "HassToggle": "toggle",
            "HassLightSet": "set",
            "HassOpenCover": "open",
            "HassCloseCover": "close",
            "HassSetPosition": "set position for",
        }

        action = action_map.get(self._intent_type, self._intent_type.lower().replace("hass", ""))

        return f"{action} the {device_name}"


def register_intent_handlers(
    hass: "HomeAssistant",
    conversation_entity_id: str,
    excluded_intents: set[str],
) -> dict[str, IntentHandler]:
    """Register PolyVoice handlers for excluded intents.

    Args:
        hass: Home Assistant instance
        conversation_entity_id: Entity ID of the PolyVoice conversation entity
        excluded_intents: Set of intent types to intercept

    Returns:
        Dict of intent_type -> original_handler for later restoration
    """
    from homeassistant.helpers import intent as intent_helpers

    original_handlers: dict[str, IntentHandler] = {}

    for intent_type in excluded_intents:
        if intent_type not in INTERCEPTED_INTENTS:
            continue

        # Save original handler
        original = intent_helpers.async_get_handler(hass, intent_type)
        if original:
            original_handlers[intent_type] = original
            _LOGGER.debug("Saved original handler for %s", intent_type)

        # Register our handler
        handler = PolyVoiceIntentHandler(
            intent_type=intent_type,
            hass=hass,
            conversation_entity_id=conversation_entity_id,
            original_handler=original,
        )
        intent_helpers.async_register(hass, handler)
        _LOGGER.info("Registered PolyVoice handler for %s intent", intent_type)

    return original_handlers


def restore_intent_handlers(
    hass: "HomeAssistant",
    original_handlers: dict[str, IntentHandler],
) -> None:
    """Restore original intent handlers.

    Args:
        hass: Home Assistant instance
        original_handlers: Dict of intent_type -> original_handler
    """
    from homeassistant.helpers import intent as intent_helpers

    for intent_type, handler in original_handlers.items():
        intent_helpers.async_register(hass, handler)
        _LOGGER.info("Restored original handler for %s intent", intent_type)

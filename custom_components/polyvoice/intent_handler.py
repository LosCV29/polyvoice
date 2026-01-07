"""Intent handler for PolyVoice.

This module intercepts native Home Assistant intents and routes them
through PolyVoice for LLM-controlled processing.

Interception logic:
1. If intent type is in excluded_intents → intercept to LLM
2. If target entity (or alias) matches a Smart Device → intercept to LLM
3. Otherwise → use native HA
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

# Regex patterns to extract device names from slot strings as last resort
SLOT_PATTERNS = [
    r"'text':\s*'([^']+)'",
    r"'name':\s*'([^']+)'",
    r"'value':\s*'([^']+)'",
]

# ALL intents that CAN be intercepted by PolyVoice
# This comprehensive list allows users to exclude any native HA intent
INTERCEPTABLE_INTENTS = frozenset([
    # Basic device control
    "HassTurnOn",
    "HassTurnOff",
    "HassToggle",
    # Lights
    "HassLightSet",
    # Covers (blinds, shades, curtains, garage doors)
    "HassOpenCover",
    "HassCloseCover",
    "HassSetPosition",
    # Climate
    "HassClimateGetTemperature",
    "HassClimateSetTemperature",
    # State queries
    "HassGetState",
    # Fan control
    "HassFanSetSpeed",
    # Humidifier
    "HassHumidifierMode",
    "HassHumidifierSetpoint",
    # Media player
    "HassMediaNext",
    "HassMediaPause",
    "HassMediaPlayerMute",
    "HassMediaPlayerUnmute",
    "HassMediaPrevious",
    "HassMediaSearchAndPlay",
    "HassMediaUnpause",
    "HassSetVolume",
    "HassSetVolumeRelative",
    # Vacuum/Lawn mower
    "HassVacuumReturnToBase",
    "HassVacuumStart",
    "HassLawnMowerDock",
    "HassLawnMowerStartMowing",
    # Timers
    "HassStartTimer",
    "HassCancelTimer",
    "HassCancelAllTimers",
    "HassIncreaseTimer",
    "HassDecreaseTimer",
    "HassPauseTimer",
    "HassUnpauseTimer",
    "HassTimerStatus",
    # Lists
    "HassListAddItem",
    "HassListCompleteItem",
    "HassShoppingListAddItem",
    "HassShoppingListCompleteItem",
    # Misc
    "HassBroadcast",
    "HassGetCurrentDate",
    "HassGetCurrentTime",
    "HassGetWeather",
    "HassNevermind",
    "HassRespond",
])


class PolyVoiceIntentHandler(IntentHandler):
    """Handler that intercepts native HA intents and routes them to PolyVoice."""

    def __init__(
        self,
        intent_type: str,
        hass: "HomeAssistant",
        conversation_entity_id: str,
        excluded_intents: set[str],
        llm_controlled_entities: set[str],
        device_aliases: dict[str, str],
        original_handler: IntentHandler | None = None,
    ):
        """Initialize the intent handler.

        Args:
            intent_type: The intent type to handle (e.g., "HassTurnOn")
            hass: Home Assistant instance
            conversation_entity_id: Entity ID of the PolyVoice conversation entity
            excluded_intents: Intent types to always intercept
            llm_controlled_entities: Entity IDs to always route to LLM (Smart Devices)
            device_aliases: Mapping of alias names to entity IDs for Smart Devices
            original_handler: The original handler to fall back to if needed
        """
        self._intent_type = intent_type
        self._hass = hass
        self._conversation_entity_id = conversation_entity_id
        self._excluded_intents = excluded_intents
        self._llm_controlled_entities = llm_controlled_entities
        self._device_aliases = device_aliases
        self._original_handler = original_handler

    @property
    def intent_type(self) -> str:
        """Return the intent type this handler handles."""
        return self._intent_type

    def _get_target_entity_id(self, intent: "Intent") -> str | None:
        """Extract the target entity ID from intent slots using fuzzy matching."""
        from .utils.fuzzy_matching import find_entity_by_name

        if not hasattr(intent, 'slots') or not intent.slots:
            _LOGGER.debug("No slots in intent")
            return None

        _LOGGER.debug("Intent slots: %s", intent.slots)

        # Get the device name from slots
        name_slot = intent.slots.get("name", {})
        device_name = None

        if isinstance(name_slot, dict):
            value = name_slot.get("value")
            if isinstance(value, dict):
                device_name = value.get("text") or value.get("name")
            elif isinstance(value, str):
                device_name = value

        if not device_name:
            _LOGGER.debug("Could not extract device_name from slots")
            return None

        _LOGGER.info("Extracted device_name='%s' from intent slots", device_name)

        # Build aliases dict for fuzzy matching:
        # 1. Friendly names of all Smart Devices (llm_controlled_entities)
        # 2. User-defined aliases (device_aliases) that point to Smart Devices
        entity_aliases = {}

        # Add friendly names of Smart Devices
        for entity_id in self._llm_controlled_entities:
            state = self._hass.states.get(entity_id)
            if state:
                friendly = state.attributes.get("friendly_name", "").lower()
                if friendly:
                    entity_aliases[friendly] = entity_id

        # Add user-defined aliases (format: {alias_name: entity_id})
        # Only include aliases that point to Smart Devices
        for alias_name, entity_id in self._device_aliases.items():
            if entity_id in self._llm_controlled_entities:
                entity_aliases[alias_name.lower()] = entity_id

        _LOGGER.info("Smart Device aliases for matching: %s", entity_aliases)

        # Use PolyVoice's fuzzy matching (handles synonyms like blind/shade)
        matched_id, matched_name = find_entity_by_name(
            self._hass, device_name, entity_aliases
        )

        _LOGGER.info("Fuzzy match result: matched_id=%s, matched_name=%s", matched_id, matched_name)

        # Only return if matched entity is a Smart Device
        if matched_id and matched_id in self._llm_controlled_entities:
            _LOGGER.info("Fuzzy matched '%s' to Smart Device %s", device_name, matched_id)
            return matched_id

        return None

    def _should_intercept(self, intent: "Intent") -> bool:
        """Determine if this intent should be intercepted.

        Returns True if:
        1. The intent type is in excluded_intents, OR
        2. The target entity (or alias) matches a Smart Device
        """
        # Always intercept if intent type is excluded
        if self._intent_type in self._excluded_intents:
            _LOGGER.debug("Intercepting %s - intent type is in excluded_intents", self._intent_type)
            return True

        # Check if target entity matches a Smart Device (via name or alias)
        target_entity = self._get_target_entity_id(intent)
        if target_entity:
            _LOGGER.info(
                "Intercepting %s for entity %s - matched Smart Device",
                self._intent_type, target_entity
            )
            return True

        return False

    async def async_handle(self, intent: "Intent") -> IntentResponse:
        """Handle the intent by routing to PolyVoice if appropriate.

        Args:
            intent: The intent to handle

        Returns:
            IntentResponse with the result
        """
        # Check if we should intercept this intent
        if not self._should_intercept(intent):
            _LOGGER.debug("Not intercepting %s - using native handler", self._intent_type)
            if self._original_handler:
                return await self._original_handler.async_handle(intent)
            raise HomeAssistantError(f"No handler for {self._intent_type}")

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
            # Get the conversation entity via EntityComponent
            entity_component = self._hass.data.get("conversation")
            entity = None
            if entity_component and hasattr(entity_component, "get_entity"):
                entity = entity_component.get_entity(self._conversation_entity_id)

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
            # Basic control
            "HassTurnOn": "turn on",
            "HassTurnOff": "turn off",
            "HassToggle": "toggle",
            # Lights
            "HassLightSet": "set",
            # Covers
            "HassOpenCover": "open",
            "HassCloseCover": "close",
            "HassSetPosition": "set position for",
            # Climate
            "HassClimateGetTemperature": "get temperature for",
            "HassClimateSetTemperature": "set temperature for",
            # State
            "HassGetState": "get state of",
            # Fan
            "HassFanSetSpeed": "set fan speed for",
            # Humidifier
            "HassHumidifierMode": "set humidifier mode for",
            "HassHumidifierSetpoint": "set humidity for",
            # Media
            "HassMediaNext": "skip to next on",
            "HassMediaPause": "pause",
            "HassMediaPlayerMute": "mute",
            "HassMediaPlayerUnmute": "unmute",
            "HassMediaPrevious": "previous track on",
            "HassMediaSearchAndPlay": "play music on",
            "HassMediaUnpause": "resume",
            "HassSetVolume": "set volume for",
            "HassSetVolumeRelative": "adjust volume for",
            # Vacuum/Lawn mower
            "HassVacuumReturnToBase": "send vacuum home",
            "HassVacuumStart": "start vacuuming with",
            "HassLawnMowerDock": "dock the lawn mower",
            "HassLawnMowerStartMowing": "start mowing with",
            # Timers
            "HassStartTimer": "start a timer for",
            "HassCancelTimer": "cancel timer",
            "HassCancelAllTimers": "cancel all timers",
            "HassIncreaseTimer": "add time to timer",
            "HassDecreaseTimer": "reduce time on timer",
            "HassPauseTimer": "pause timer",
            "HassUnpauseTimer": "resume timer",
            "HassTimerStatus": "check timer status",
            # Lists
            "HassListAddItem": "add to list",
            "HassListCompleteItem": "complete item on list",
            "HassShoppingListAddItem": "add to shopping list",
            "HassShoppingListCompleteItem": "complete shopping list item",
            # Misc
            "HassBroadcast": "broadcast",
            "HassGetCurrentDate": "get current date",
            "HassGetCurrentTime": "get current time",
            "HassGetWeather": "get weather",
            "HassNevermind": "cancel",
            "HassRespond": "respond",
        }

        action = action_map.get(self._intent_type, self._intent_type.lower().replace("hass", ""))

        return f"{action} the {device_name}"


def register_intent_handlers(
    hass: "HomeAssistant",
    conversation_entity_id: str,
    excluded_intents: set[str],
    llm_controlled_entities: set[str],
    device_aliases: dict[str, str],
) -> dict[str, IntentHandler]:
    """Register PolyVoice handlers for interceptable intents.

    Handlers are registered when:
    - excluded_intents is set: registers handlers for those specific intent types
    - llm_controlled_entities is set: registers handlers for ALL interceptable
      intents to enable per-entity (Smart Device) routing

    Args:
        hass: Home Assistant instance
        conversation_entity_id: Entity ID of the PolyVoice conversation entity
        excluded_intents: Set of intent types to always intercept
        llm_controlled_entities: Set of entity IDs to always route to LLM (Smart Devices)
        device_aliases: Mapping of alias names to entity IDs for Smart Devices

    Returns:
        Dict of intent_type -> original_handler for later restoration
    """
    from homeassistant.helpers import intent as intent_helpers

    original_handlers: dict[str, IntentHandler] = {}

    # Build set of intents to register:
    # - All excluded_intents (user explicitly wants these intercepted)
    # - All INTERCEPTABLE_INTENTS if we have Smart Devices configured
    intents_to_register: set[str] = set()

    # Always register excluded_intents (if they're interceptable)
    for intent_type in excluded_intents:
        if intent_type in INTERCEPTABLE_INTENTS:
            intents_to_register.add(intent_type)
        else:
            _LOGGER.warning(
                "Intent %s is not interceptable by PolyVoice - skipping", intent_type
            )

    # If we have Smart Devices configured, register ALL interceptable intents
    if llm_controlled_entities:
        intents_to_register.update(INTERCEPTABLE_INTENTS)

    for intent_type in intents_to_register:
        # Get original handler if it exists
        # Handlers are stored in hass.data["intent"] as a dict
        intent_handlers = hass.data.get("intent", {})
        original = intent_handlers.get(intent_type) if isinstance(intent_handlers, dict) else None

        # Register our handler (it will override any existing one)
        handler = PolyVoiceIntentHandler(
            intent_type=intent_type,
            hass=hass,
            conversation_entity_id=conversation_entity_id,
            excluded_intents=excluded_intents,
            llm_controlled_entities=llm_controlled_entities,
            device_aliases=device_aliases,
            original_handler=original,
        )
        intent_helpers.async_register(hass, handler)
        _LOGGER.info("Registered PolyVoice handler for %s intent", intent_type)

        if original:
            original_handlers[intent_type] = original

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

"""Conversation platform for PolyVoice - Multi-Provider Support."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
from typing import Any, Literal

import aiohttp
import voluptuous as vol
from openai import AsyncOpenAI, AsyncAzureOpenAI

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.components.camera import async_get_image
from homeassistant.helpers import intent, entity_registry as er
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import ulid, dt as dt_util

from .const import (
    DOMAIN,
    # Provider settings
    CONF_PROVIDER,
    CONF_BASE_URL,
    CONF_API_KEY,
    CONF_MODEL,
    CONF_TEMPERATURE,
    CONF_MAX_TOKENS,
    CONF_TOP_P,
    CONF_REASONING_EFFORT,
    CONF_USE_RESPONSES_API,
    # Provider constants
    PROVIDER_LM_STUDIO,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
    PROVIDER_AZURE,
    PROVIDER_OLLAMA,
    PROVIDER_BASE_URLS,
    OPENAI_COMPATIBLE_PROVIDERS,
    DEFAULT_PROVIDER,
    DEFAULT_API_KEY,
    # Native intents
    CONF_USE_NATIVE_INTENTS,
    CONF_EXCLUDED_INTENTS,
    CONF_CUSTOM_EXCLUDED_INTENTS,
    CONF_SYSTEM_PROMPT,
    CONF_ENABLE_ASSIST,
    CONF_LLM_HASS_API,
    CONF_CUSTOM_LATITUDE,
    CONF_CUSTOM_LONGITUDE,
    # External API keys
    CONF_OPENWEATHERMAP_API_KEY,
    CONF_GOOGLE_PLACES_API_KEY,
    CONF_YELP_API_KEY,
    CONF_NEWSAPI_KEY,
    # Feature toggles
    CONF_ENABLE_WEATHER,
    CONF_ENABLE_CALENDAR,
    CONF_ENABLE_CAMERAS,
    CONF_ENABLE_SPORTS,
    CONF_ENABLE_STOCKS,
    CONF_ENABLE_NEWS,
    CONF_ENABLE_PLACES,
    CONF_ENABLE_RESTAURANTS,
    CONF_ENABLE_THERMOSTAT,
    CONF_ENABLE_DEVICE_STATUS,
    CONF_ENABLE_WIKIPEDIA,
    CONF_ENABLE_MUSIC,
    # Entity config
    CONF_THERMOSTAT_ENTITY,
    CONF_ROOM_PLAYER_MAPPING,
    CONF_LAST_ACTIVE_SPEAKER,
    CONF_CALENDAR_ENTITIES,
    CONF_DEVICE_ALIASES,
    CONF_NOTIFICATION_SERVICE,
    CONF_CAMERA_ENTITIES,
    # Defaults
    DEFAULT_EXCLUDED_INTENTS,
    DEFAULT_CUSTOM_EXCLUDED_INTENTS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_ENABLE_ASSIST,
    DEFAULT_LLM_HASS_API,
    DEFAULT_ENABLE_WEATHER,
    DEFAULT_ENABLE_CALENDAR,
    DEFAULT_ENABLE_CAMERAS,
    DEFAULT_ENABLE_SPORTS,
    DEFAULT_ENABLE_STOCKS,
    DEFAULT_ENABLE_NEWS,
    DEFAULT_ENABLE_PLACES,
    DEFAULT_ENABLE_RESTAURANTS,
    DEFAULT_ENABLE_THERMOSTAT,
    DEFAULT_ENABLE_DEVICE_STATUS,
    DEFAULT_ENABLE_WIKIPEDIA,
    DEFAULT_ENABLE_MUSIC,
    DEFAULT_ROOM_PLAYER_MAPPING,
    DEFAULT_LAST_ACTIVE_SPEAKER,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_USE_RESPONSES_API,
    CAMERA_FRIENDLY_NAMES,
    ALL_NATIVE_INTENTS,
    # Thermostat settings
    CONF_THERMOSTAT_MIN_TEMP,
    CONF_THERMOSTAT_MAX_TEMP,
    CONF_THERMOSTAT_TEMP_STEP,
    CONF_THERMOSTAT_USE_CELSIUS,
    DEFAULT_THERMOSTAT_MIN_TEMP,
    DEFAULT_THERMOSTAT_MAX_TEMP,
    DEFAULT_THERMOSTAT_TEMP_STEP,
    DEFAULT_THERMOSTAT_USE_CELSIUS,
    DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS,
    # Event names
    CONF_FACIAL_RECOGNITION_EVENT,
    DEFAULT_FACIAL_RECOGNITION_EVENT,
)

_LOGGER = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CONSTANTS - Now loaded from config, with fallback defaults
# =============================================================================

# Default location (used when custom location not set and HA location unavailable)
DEFAULT_LATITUDE = 0.0
DEFAULT_LONGITUDE = 0.0


# SPEED OPTIMIZATION PATTERNS
SIMPLE_QUERY_PATTERNS = [
    "turn on", "turn off", "switch on", "switch off",
    "lock", "unlock", "open", "close",
    "what time", "what's the time", "current time",
    "what temperature", "what's the temperature",
    "is the", "are the", "status of",
    "start", "stop", "pause", "resume",
]

# SAFETY: Allowlisted domains for HA service calls (prevents dangerous calls)
ALLOWED_SERVICE_DOMAINS = {
    "light", "switch", "cover", "lock", "climate", "media_player", 
    "fan", "vacuum", "scene", "script", "input_boolean", "input_number",
    "input_select", "input_text", "automation", "timer", "counter",
    "number", "select", "button", "siren", "humidifier", "notify",
}

# API timeout in seconds for external calls
API_TIMEOUT = 15

# CAMERA_FRIENDLY_NAMES is now imported from const.py

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_entity_config(config_string: str) -> dict[str, str]:
    """Parse a config string like 'room:entity_id' into a dict."""
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
    """Parse a config string with one item per line into a list."""
    if not config_string:
        return []
    return [line.strip() for line in config_string.strip().split("\n") if line.strip()]


def get_friendly_name(entity_id: str, state) -> str:
    """Get the friendly name for an entity."""
    return state.attributes.get("friendly_name", entity_id.split(".")[-1])


def format_human_readable_state(entity_id: str, state: str) -> str:
    """Convert entity state to human-readable format."""
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
    
    a = sin(dlat/2)**2 + cos(lat1_r) * cos(lat2_r) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return 3956 * c  # Earth's radius in miles


async def fetch_wikidata_birthdate(session: aiohttp.ClientSession, wikibase_item: str) -> dict | None:
    """
    Fetch birthdate from Wikidata for a given Wikibase item ID.
    Returns dict with 'birthdate' and 'age' or None if not found.
    """
    try:
        headers = {"User-Agent": "HomeAssistant-PolyVoice/1.0"}
        wikidata_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikibase_item}.json"

        async with session.get(wikidata_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            wd_data = await resp.json()

        entity = wd_data.get("entities", {}).get(wikibase_item, {})
        claims = entity.get("claims", {})

        # P569 is birth date property in Wikidata
        if "P569" not in claims:
            return None

        birth_claim = claims["P569"][0]
        time_value = birth_claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "")

        if not time_value:
            return None

        # Parse Wikidata date format: +YYYY-MM-DDTHH:MM:SSZ
        match = re.match(r'\+(\d{4})-(\d{2})-(\d{2})', time_value)
        if not match:
            return None

        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        birthdate = datetime(year, month, day)

        today = datetime.now()
        age = today.year - birthdate.year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            age -= 1

        return {
            "birthdate": birthdate,
            "birthdate_formatted": birthdate.strftime("%B %d, %Y"),
            "age": age,
        }
    except Exception as e:
        _LOGGER.warning("Wikidata fetch error: %s", e)
        return None


def find_entity_by_name(hass: HomeAssistant, query: str, device_aliases: dict) -> tuple[str | None, str | None]:
    """
    Search for entity using device aliases first, then fall back to HA entity registry aliases.
    Returns (entity_id, friendly_name) or (None, None) if not found.
    """
    query_lower = query.lower().strip()
    
    # FIRST: Check configured device aliases (exact match)
    if query_lower in device_aliases:
        entity_id = device_aliases[query_lower]
        state = hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", query) if state else query
        return (entity_id, friendly_name)
    
    # SECOND: Check configured device aliases (partial match)
    for alias, entity_id in device_aliases.items():
        if query_lower in alias or alias in query_lower:
            state = hass.states.get(entity_id)
            friendly_name = state.attributes.get("friendly_name", alias) if state else alias
            return (entity_id, friendly_name)
    
    # THIRD: Check HA entity registry aliases
    ent_reg = er.async_get(hass)
    for entity_entry in ent_reg.entities.values():
        if entity_entry.aliases:
            for alias in entity_entry.aliases:
                if alias.lower() == query_lower:
                    state = hass.states.get(entity_entry.entity_id)
                    friendly_name = state.attributes.get("friendly_name", alias) if state else alias
                    return (entity_entry.entity_id, friendly_name)
    
    # FOURTH: Check HA entity registry aliases (partial match)
    for entity_entry in ent_reg.entities.values():
        if entity_entry.aliases:
            for alias in entity_entry.aliases:
                if query_lower in alias.lower() or alias.lower() in query_lower:
                    state = hass.states.get(entity_entry.entity_id)
                    friendly_name = state.attributes.get("friendly_name", alias) if state else alias
                    return (entity_entry.entity_id, friendly_name)
    
    # FIFTH: Check friendly names (exact)
    for state in hass.states.async_all():
        friendly_name = state.attributes.get("friendly_name", "")
        if friendly_name and friendly_name.lower() == query_lower:
            return (state.entity_id, friendly_name)
    
    # SIXTH: Check friendly names (partial)
    for state in hass.states.async_all():
        friendly_name = state.attributes.get("friendly_name", "")
        if friendly_name and query_lower in friendly_name.lower():
            return (state.entity_id, friendly_name)
    
    return (None, None)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entity."""
    agent = LMStudioConversationEntity(config_entry)
    async_add_entities([agent])
    
    # Store agent reference for service calls
    hass.data.setdefault("polyvoice", {})
    hass.data["polyvoice"][config_entry.entry_id] = agent
    
    # Register facial recognition service
    async def handle_facial_recognition(call):
        """Handle facial recognition service call."""
        camera = call.data.get("camera", "porch")
        notify = call.data.get("notify", True)
        
        _LOGGER.info("Facial recognition service called for camera: %s", camera)
        
        # Find the agent
        for entry_id, stored_agent in hass.data.get("polyvoice", {}).items():
            if hasattr(stored_agent, "_check_single_camera"):
                # Ensure session is available
                if stored_agent._session is None:
                    stored_agent._session = async_get_clientsession(hass)
                
                result = await stored_agent._check_single_camera(camera, "")
                
                if notify and "analysis" in result:
                    # Get identified people
                    identified = result.get("identified_people", [])
                    if identified:
                        names = [p["name"] for p in identified]
                        title = f"🏠 {', '.join(names)} at {camera}"
                    else:
                        title = f"📷 Person at {camera}"

                    # Send notification using configured service
                    notification_service = getattr(stored_agent, 'notification_service', '')
                    if notification_service:
                        try:
                            await hass.services.async_call(
                                "notify", notification_service,
                                {
                                    "title": title,
                                    "message": result["analysis"],
                                    "data": {
                                        "image": f"/api/camera_proxy/camera.{camera}_camera"
                                    }
                                }
                            )
                        except Exception as e:
                            _LOGGER.warning("Could not send notification: %s", e)
                    else:
                        _LOGGER.debug("No notification service configured")
                
                # Fire event for other automations (use configurable event name)
                event_name = getattr(stored_agent, 'facial_recognition_event', DEFAULT_FACIAL_RECOGNITION_EVENT)
                hass.bus.async_fire(event_name, {
                    "camera": camera,
                    "identified_people": result.get("identified_people", []),
                    "analysis": result.get("analysis", ""),
                    "is_known_person": len(result.get("identified_people", [])) > 0
                })
                
                _LOGGER.info("Facial recognition complete for %s: %s", camera, result)
                return result
        
        _LOGGER.warning("No LM Studio agent found for facial recognition")
        return {"error": "Agent not found"}
    
    # Register the service
    hass.services.async_register(
        "polyvoice",
        "facial_recognition",
        handle_facial_recognition,
        schema=vol.Schema({
            vol.Optional("camera", default="porch"): str,
            vol.Optional("notify", default=True): bool,
        })
    )
    
    _LOGGER.info("Registered polyvoice.facial_recognition service")


class LMStudioConversationEntity(ConversationEntity):
    """LM Studio conversation agent entity - SPEED OPTIMIZED."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = config_entry
        self._attr_unique_id = config_entry.entry_id
        self._session = None  # Shared aiohttp session - set in async_added_to_hass

        # Usage tracking for HA dashboard sensors
        self._api_calls = {
            "weather": 0, "places": 0, "restaurants": 0, "news": 0,
            "sports": 0, "wikipedia": 0, "llm": 0,
        }
        self._tokens_used = {"input": 0, "output": 0}

        # Tools cache (built once, reused for all requests)
        self._tools = None

        # Initialize config
        self._update_from_config({**config_entry.data, **config_entry.options})

    def _update_from_config(self, config: dict[str, Any]) -> None:
        """Update configuration with multi-provider support."""
        # Provider selection
        self.provider = config.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        self.api_key = config.get(CONF_API_KEY, DEFAULT_API_KEY)
        self.model = config.get(CONF_MODEL, "")
        self.temperature = config.get(CONF_TEMPERATURE, 0.7)
        self.max_tokens = config.get(CONF_MAX_TOKENS, 2000)
        self.top_p = config.get(CONF_TOP_P, 0.95)

        # GPT-5 specific settings
        self.reasoning_effort = config.get(CONF_REASONING_EFFORT, DEFAULT_REASONING_EFFORT)
        self.use_responses_api = config.get(CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API)

        # Check if model is GPT-5
        self.is_gpt5 = self.model.lower().startswith("gpt-5") or "gpt-5" in self.model.lower()

        # Get base URL (use provider default if not specified)
        base_url = config.get(CONF_BASE_URL)
        if not base_url:
            base_url = PROVIDER_BASE_URLS.get(self.provider, "http://localhost:1234/v1")
        self.base_url = base_url

        # Create client for OpenAI-compatible providers
        if self.provider == PROVIDER_AZURE:
            # Azure OpenAI uses a different client with specific configuration
            # Extract the azure endpoint from the base_url
            # Expected format: https://{resource}.openai.azure.com/openai/deployments/{deployment}
            # or just: https://{resource}.openai.azure.com
            azure_endpoint = self.base_url
            if "/openai/deployments/" in azure_endpoint:
                azure_endpoint = azure_endpoint.split("/openai/deployments/")[0]
            self.client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=self.api_key,
                api_version="2024-02-01",
            )
        elif self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            # Standard OpenAI-compatible providers (LM Studio, OpenAI, Groq, OpenRouter, Ollama)
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key if self.api_key else "ollama",  # Ollama doesn't require auth
            )
        else:
            # For Anthropic and Google, we'll use aiohttp directly
            self.client = None
        
        self.use_native_intents = config.get(CONF_USE_NATIVE_INTENTS, True)
        
        self.enable_assist = config.get(CONF_ENABLE_ASSIST, DEFAULT_ENABLE_ASSIST)
        if self.enable_assist:
            self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL
        else:
            self._attr_supported_features = 0
        
        llm_api_config = config.get(CONF_LLM_HASS_API, [DEFAULT_LLM_HASS_API])
        if isinstance(llm_api_config, str):
            self.llm_api_ids = [api.strip() for api in llm_api_config.split(",") if api.strip()]
        elif isinstance(llm_api_config, list):
            self.llm_api_ids = llm_api_config
        else:
            self.llm_api_ids = [DEFAULT_LLM_HASS_API]
        
        # ALWAYS use default excluded intents (hardcoded) - UI can only ADD more, not remove
        self.excluded_intents = set(DEFAULT_EXCLUDED_INTENTS)

        # Add any custom excluded intents from UI config
        custom_excluded = config.get(CONF_CUSTOM_EXCLUDED_INTENTS, DEFAULT_CUSTOM_EXCLUDED_INTENTS)
        if custom_excluded:
            custom_list = [i.strip() for i in custom_excluded.split(",") if i.strip()]
            self.excluded_intents.update(custom_list)

        _LOGGER.warning("=== EXCLUDED INTENTS (hardcoded + custom) === %s", self.excluded_intents)

        self.system_prompt = config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
        
        # Custom location for external APIs (falls back to HA location if not set)
        custom_lat = config.get(CONF_CUSTOM_LATITUDE)
        custom_lon = config.get(CONF_CUSTOM_LONGITUDE)
        
        try:
            lat = float(custom_lat) if custom_lat else 0.0
            self.custom_latitude = lat if lat != 0.0 else None
        except (ValueError, TypeError):
            self.custom_latitude = None
            
        try:
            lon = float(custom_lon) if custom_lon else 0.0
            self.custom_longitude = lon if lon != 0.0 else None
        except (ValueError, TypeError):
            self.custom_longitude = None
        
        # External API keys from config
        self.openweathermap_api_key = config.get(CONF_OPENWEATHERMAP_API_KEY, "")
        self.google_places_api_key = config.get(CONF_GOOGLE_PLACES_API_KEY, "")
        self.yelp_api_key = config.get(CONF_YELP_API_KEY, "")
        self.newsapi_key = config.get(CONF_NEWSAPI_KEY, "")
        
        # Feature toggles
        self.enable_weather = config.get(CONF_ENABLE_WEATHER, DEFAULT_ENABLE_WEATHER)
        self.enable_calendar = config.get(CONF_ENABLE_CALENDAR, DEFAULT_ENABLE_CALENDAR)
        self.enable_cameras = config.get(CONF_ENABLE_CAMERAS, DEFAULT_ENABLE_CAMERAS)
        self.enable_sports = config.get(CONF_ENABLE_SPORTS, DEFAULT_ENABLE_SPORTS)
        self.enable_stocks = config.get(CONF_ENABLE_STOCKS, DEFAULT_ENABLE_STOCKS)
        self.enable_news = config.get(CONF_ENABLE_NEWS, DEFAULT_ENABLE_NEWS)
        self.enable_places = config.get(CONF_ENABLE_PLACES, DEFAULT_ENABLE_PLACES)
        self.enable_restaurants = config.get(CONF_ENABLE_RESTAURANTS, DEFAULT_ENABLE_RESTAURANTS)
        self.enable_thermostat = config.get(CONF_ENABLE_THERMOSTAT, DEFAULT_ENABLE_THERMOSTAT)
        self.enable_device_status = config.get(CONF_ENABLE_DEVICE_STATUS, DEFAULT_ENABLE_DEVICE_STATUS)
        self.enable_wikipedia = config.get(CONF_ENABLE_WIKIPEDIA, DEFAULT_ENABLE_WIKIPEDIA)
        self.enable_music = config.get(CONF_ENABLE_MUSIC, DEFAULT_ENABLE_MUSIC)

        # Music configuration
        raw_mapping = config.get(CONF_ROOM_PLAYER_MAPPING, DEFAULT_ROOM_PLAYER_MAPPING)
        self.room_player_mapping = parse_entity_config(raw_mapping)
        self.last_active_speaker = config.get(CONF_LAST_ACTIVE_SPEAKER, DEFAULT_LAST_ACTIVE_SPEAKER)
        _LOGGER.info("Music config loaded: enable_music=%s, raw_mapping='%s', parsed=%s",
                     self.enable_music, raw_mapping, self.room_player_mapping)

        # Entity configuration from UI
        self.thermostat_entity = config.get(CONF_THERMOSTAT_ENTITY, "")
        self.calendar_entities = parse_list_config(config.get(CONF_CALENDAR_ENTITIES, ""))
        self.camera_entities = parse_list_config(config.get(CONF_CAMERA_ENTITIES, ""))
        self.device_aliases = parse_entity_config(config.get(CONF_DEVICE_ALIASES, ""))
        self.notification_service = config.get(CONF_NOTIFICATION_SERVICE, "")

        # Thermostat settings (user-configurable limits and step)
        # First determine unit preference to select appropriate defaults
        self.thermostat_use_celsius = config.get(CONF_THERMOSTAT_USE_CELSIUS, DEFAULT_THERMOSTAT_USE_CELSIUS)

        # Choose defaults based on unit preference
        if self.thermostat_use_celsius:
            default_min = DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS
            default_max = DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS
        else:
            default_min = DEFAULT_THERMOSTAT_MIN_TEMP
            default_max = DEFAULT_THERMOSTAT_MAX_TEMP
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP

        # Load configured values, falling back to unit-appropriate defaults
        configured_min = config.get(CONF_THERMOSTAT_MIN_TEMP)
        configured_max = config.get(CONF_THERMOSTAT_MAX_TEMP)
        configured_step = config.get(CONF_THERMOSTAT_TEMP_STEP)

        # Detect and fix values that appear to be in the wrong unit
        # (e.g., 60-85 stored when user switches to Celsius)
        if self.thermostat_use_celsius:
            if configured_min is not None and configured_min > 40:  # Likely Fahrenheit value
                configured_min = None  # Reset to Celsius default
            if configured_max is not None and configured_max > 50:  # Likely Fahrenheit value
                configured_max = None  # Reset to Celsius default
        else:
            if configured_min is not None and configured_min < 32:  # Likely Celsius value
                configured_min = None  # Reset to Fahrenheit default
            if configured_max is not None and configured_max < 50:  # Likely Celsius value
                configured_max = None  # Reset to Fahrenheit default

        self.thermostat_min_temp = int(configured_min if configured_min is not None else default_min)
        self.thermostat_max_temp = int(configured_max if configured_max is not None else default_max)
        self.thermostat_temp_step = int(configured_step if configured_step is not None else default_step)

        # Event names (user-configurable)
        self.facial_recognition_event = config.get(CONF_FACIAL_RECOGNITION_EVENT, DEFAULT_FACIAL_RECOGNITION_EVENT)

        # Build camera friendly names mapping from configured camera entities
        # camera.front_porch -> key: "front_porch", friendly: "Front Porch"
        self.camera_friendly_names = {}
        for entity_id in self.camera_entities:
            # Extract camera key from entity_id (e.g., camera.front_porch -> front_porch)
            if entity_id.startswith("camera."):
                camera_key = entity_id.replace("camera.", "").replace("_camera", "")
            else:
                camera_key = entity_id.replace("_camera", "")
            # Get friendly name from defaults or generate from key
            friendly_name = CAMERA_FRIENDLY_NAMES.get(
                camera_key,
                camera_key.replace("_", " ").title()
            )
            self.camera_friendly_names[camera_key] = friendly_name

        # Build tools list ONCE (major performance boost!)
        self._tools = self._build_tools()
        
        _LOGGER.info(
            "Config updated - Provider: %s, Model: %s, Assist: %s, Tools: %d",
            self.provider, self.model, self.enable_assist, len(self._tools)
        )
        _LOGGER.info("Excluded intents: %s", self.excluded_intents)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @property
    def device_info(self):
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self.entry.entry_id)},
            "name": self.entry.data.get(CONF_NAME, "PolyVoice"),
            "manufacturer": "LM Studio",
            "model": "Local LLM",
            "entry_type": "service",
        }

    @property
    def temp_unit(self) -> str:
        """Return the temperature unit symbol based on user preference."""
        return "°C" if self.thermostat_use_celsius else "°F"

    def format_temp(self, temp: float | int) -> str:
        """Format a temperature value with the appropriate unit."""
        return f"{int(temp)}{self.temp_unit}"

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_updated)
        )
        
        # Get shared aiohttp session (HUGE perf boost - reuses TCP connections!)
        self._session = async_get_clientsession(self.hass)
        
        # Register usage tracking sensors
        self._update_usage_sensors()

    async def _async_entry_updated(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle config entry update."""
        self._update_from_config({**entry.data, **entry.options})
        self.async_write_ha_state()

    def _update_usage_sensors(self) -> None:
        """Update usage tracking sensors in Home Assistant."""
        if not self.hass:
            return
        self.hass.states.async_set(
            f"sensor.{DOMAIN}_api_calls",
            sum(self._api_calls.values()),
            {"friendly_name": "LM Studio API Calls", "icon": "mdi:api", **self._api_calls}
        )
        self.hass.states.async_set(
            f"sensor.{DOMAIN}_tokens_used",
            self._tokens_used["input"] + self._tokens_used["output"],
            {
                "friendly_name": "LM Studio Tokens Used", "icon": "mdi:counter",
                "input_tokens": self._tokens_used["input"],
                "output_tokens": self._tokens_used["output"],
                "unit_of_measurement": "tokens",
            }
        )

    def _track_api_call(self, api_name: str) -> None:
        """Track an API call for usage statistics."""
        if api_name in self._api_calls:
            self._api_calls[api_name] += 1
            self._update_usage_sensors()

    def _track_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Track token usage for statistics."""
        self._tokens_used["input"] += input_tokens
        self._tokens_used["output"] += output_tokens
        self._update_usage_sensors()

    def _build_tools(self) -> list[dict]:
        """Build the tools list based on enabled features."""
        tools = []
        
        # ===== CORE TOOLS (always enabled) =====
        tools.append({
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get the current date and time. Use for 'what day is it', 'what's the date', 'what time is it', or any time/date questions.",
                "parameters": {"type": "object", "properties": {}}
            }
        })
        
        # ===== WEATHER (if enabled and API key available) =====
        if self.enable_weather and self.openweathermap_api_key:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get current weather AND forecast. ALWAYS call this for weather questions: 'what's the weather', 'will it rain', 'temperature', 'forecast'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name (optional, defaults to configured location)"},
                            "forecast_type": {"type": "string", "enum": ["current", "weekly", "both"], "description": "Type: 'current' for now, 'weekly' for 5-day, 'both' for all (default: both)"}
                        }
                    }
                }
            })
        
        # ===== PLACES (if enabled and API key available) =====
        if self.enable_places and self.google_places_api_key:
            tools.append({
                "type": "function",
                "function": {
                    "name": "find_nearby_places",
                    "description": "Find nearby places for DIRECTIONS. Use for 'nearest', 'closest', 'where is', 'find a', 'directions to'. NOT for food recommendations (use get_restaurant_recommendations instead).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for (e.g., 'Publix', 'gas station', 'pharmacy', 'CVS')"},
                            "max_results": {"type": "integer", "description": "Max results (default: 5, max: 20)"}
                        },
                        "required": ["query"]
                    }
                }
            })
        
        # ===== THERMOSTAT (if enabled and entity configured) =====
        if self.enable_thermostat and self.thermostat_entity:
            temp_unit = self.temp_unit
            step = self.thermostat_temp_step
            tools.append({
                "type": "function",
                "function": {
                    "name": "control_thermostat",
                    "description": f"Control or check the thermostat/AC/air. Use for: 'raise/lower the AC' (±{step}{temp_unit}), 'set AC to 72', 'what is the AC set to', 'what's the temp inside'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["raise", "lower", "set", "check"], "description": f"'raise' = +{step}{temp_unit}, 'lower' = -{step}{temp_unit}, 'set' = specific temp, 'check' = get current status"},
                            "temperature": {"type": "number", "description": f"Target temperature in {temp_unit} (only for 'set' action)"}
                        },
                        "required": ["action"]
                    }
                }
            })
        
        # ===== WIKIPEDIA/AGE (if enabled) =====
        if self.enable_wikipedia:
            tools.append({
                "type": "function",
                "function": {
                    "name": "calculate_age",
                    "description": "REQUIRED for 'how old is [person]' questions. Looks up birthdate from Wikidata and calculates current age. NEVER guess ages - ALWAYS use this tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "person_name": {"type": "string", "description": "The person's name (e.g., 'LeBron James', 'Taylor Swift')"}
                        },
                        "required": ["person_name"]
                    }
                }
            })
            
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_wikipedia_summary",
                    "description": "Get information from Wikipedia. Use for 'who is', 'what is', 'tell me about' questions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "The topic to look up (e.g., 'Albert Einstein', 'World War II', 'photosynthesis')"}
                        },
                        "required": ["topic"]
                    }
                }
            })
        
        # ===== SPORTS (if enabled) =====
        if self.enable_sports:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_sports_info",
                    "description": "MANDATORY: You MUST call this tool for ANY sports question. NEVER answer sports questions from memory - scores and schedules change constantly. Use for: 'did [team] win', 'when is the next [team] game', '[team] score', 'how did [team] do'. Returns real-time scores and schedules.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "team_name": {"type": "string", "description": "Team name. IMPORTANT: If user mentions a specific league (Champions League, UCL, Premier League, etc.), include it! Examples: 'Liverpool Champions League', 'Man City UCL', 'Real Madrid Champions League', 'Miami Heat', 'Duke Blue Devils'"},
                            "query_type": {"type": "string", "enum": ["last_game", "next_game", "standings", "both"], "description": "What info to get: 'last_game' for recent result, 'next_game' for upcoming, 'standings' for league position, 'both' for last and next games (default)"}
                        },
                        "required": ["team_name"]
                    }
                }
            })
            # UFC/MMA tool (event-based, not team-based)
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_ufc_info",
                    "description": "Get UFC/MMA fight information. Use for: 'next UFC event', 'when is UFC', 'upcoming UFC fights', 'UFC schedule'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_type": {"type": "string", "enum": ["next_event", "upcoming"], "description": "What info to get: 'next_event' for the next UFC event, 'upcoming' for list of upcoming events"}
                        }
                    }
                }
            })

        # ===== STOCKS (if enabled - free API, no key required) =====
        if self.enable_stocks:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Get current stock price and daily change. Use for: 'Apple stock', 'what's Tesla at', 'AAPL price', 'how is NVDA doing'. Works with stock symbols (AAPL, TSLA, GOOGL) or company names.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol (e.g., 'AAPL', 'TSLA', 'GOOGL', 'MSFT', 'NVDA', 'AMZN') or company name (e.g., 'Apple', 'Tesla')"}
                        },
                        "required": ["symbol"]
                    }
                }
            })

        # ===== NEWS (if enabled and API key available) =====
        if self.enable_news and self.newsapi_key:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_news",
                    "description": "Get latest news headlines. Use for 'what's in the news', 'latest headlines', 'news about X', 'tech news', 'sports news', etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "enum": ["general", "science", "sports", "business", "health", "entertainment", "tech", "politics", "food", "travel"], "description": "News category (optional)"},
                            "topic": {"type": "string", "description": "Specific topic to search for (e.g., 'Tesla', 'AI', 'climate change')"},
                            "max_results": {"type": "integer", "description": "Number of articles to return (default: 5, max: 10)"}
                        }
                    }
                }
            })
        
        # ===== CALENDAR (if enabled and entities configured) =====
        if self.enable_calendar and self.calendar_entities:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_calendar_events",
                    "description": "Get upcoming calendar events. Use for 'what's on my calendar', 'any events today', 'my schedule', 'upcoming appointments'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days_ahead": {"type": "integer", "description": "Number of days to look ahead (default: 7, max: 30)"}
                        }
                    }
                }
            })

        # ===== RESTAURANT RECOMMENDATIONS (if enabled and API key available) =====
        if self.enable_restaurants and self.yelp_api_key:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_restaurant_recommendations",
                    "description": "Get restaurant recommendations and reviews using Yelp. Use for 'best tacos near me', 'good sushi restaurants', 'where should I eat', food recommendations. Do NOT use for directions - use find_nearby_places for that.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What type of food or restaurant to search for (e.g., 'tacos', 'sushi', 'Italian', 'breakfast', 'coffee')"},
                            "max_results": {"type": "integer", "description": "Number of results to return (default: 5, max: 10)"}
                        },
                        "required": ["query"]
                    }
                }
            })

        # ===== CAMERA CHECKS (if enabled) =====
        # Uses ha_video_vision integration for AI analysis
        if self.enable_cameras:
            # Unified camera check with detail level
            tools.append({
                "type": "function",
                "function": {
                    "name": "check_camera",
                    "description": "Check a camera with AI vision analysis. Use for: 'check the [location] camera', 'what's happening in [location]', 'who's at the [location]', 'is anyone at [location]', 'is someone in [location]'. Works with any camera location: garage, kitchen, nursery, driveway, porch, backyard, living room, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The camera location to check. Examples: 'garage', 'kitchen', 'nursery', 'living room', 'driveway', 'porch', 'backyard', 'front door', 'doorbell'"
                            },
                            "detail_level": {
                                "type": "string",
                                "enum": ["quick", "detailed"],
                                "description": "Level of detail: 'quick' for fast presence check (is anyone there?), 'detailed' for full scene description + facial recognition. Default: 'quick' for presence questions, 'detailed' for scene/activity questions."
                            },
                            "query": {
                                "type": "string",
                                "description": "Optional specific question about what to look for (e.g., 'is the baby sleeping', 'is there a package', 'is the car there'). If provided, implies 'detailed' mode."
                            }
                        },
                        "required": ["location"]
                    }
                }
            })

        # ===== DEVICE STATUS (if enabled) =====
        if self.enable_device_status:
            tools.append({
                "type": "function",
                "function": {
                    "name": "check_device_status",
                    "description": "Check the current status of any device, sensor, door, lock, light, switch, or cover. IMPORTANT: Pass the COMPLETE device name exactly as the user said it - do not shorten or abbreviate. Examples: 'nursery door' not 'nursery', 'front door' not 'front', 'garage door' not 'garage'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device": {"type": "string", "description": "The COMPLETE device name exactly as spoken by the user. Include all words like 'door', 'light', 'sensor', etc. Examples: 'nursery door', 'front door', 'back door', 'garage door', 'kitchen light'"}
                        },
                        "required": ["device"]
                    }
                }
            })

        # ===== MUSIC CONTROL (if enabled and room mapping configured) =====
        _LOGGER.info("Music check: enable_music=%s, room_player_mapping=%s", self.enable_music, self.room_player_mapping)
        if self.enable_music and self.room_player_mapping:
            rooms_list = ", ".join(self.room_player_mapping.keys())
            _LOGGER.info("MUSIC TOOL ENABLED with rooms: %s", rooms_list)
            tools.append({
                "type": "function",
                "function": {
                    "name": "control_music",
                    "description": f"Control music playback via Music Assistant. Rooms: {rooms_list}. Actions: play, pause, resume, stop, skip_next, skip_previous, what_playing, transfer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["play", "pause", "resume", "stop", "skip_next", "skip_previous", "what_playing", "transfer"],
                                "description": "The music action to perform"
                            },
                            "query": {"type": "string", "description": "What to play (artist, album, track, playlist, or genre)"},
                            "room": {"type": "string", "description": f"Target room: {rooms_list}"},
                            "media_type": {
                                "type": "string",
                                "enum": ["artist", "album", "track", "playlist", "genre"],
                                "description": "Type of media (default: artist)"
                            },
                            "shuffle": {"type": "boolean", "description": "Enable shuffle mode"}
                        },
                        "required": ["action"]
                    }
                }
            })

        return tools

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        conversation_id = user_input.conversation_id or ulid.ulid_now()
        
        # Store original query for tools to access (for reliable device name extraction)
        self._current_user_query = user_input.text

        _LOGGER.info("=== Incoming request: '%s' (conv_id: %s) ===", user_input.text, conversation_id[:8])

        # Try native intents first if enabled
        if self.use_native_intents:
            native_result = await self._try_native_intent(user_input, conversation_id)
            if native_result is not None:
                return native_result

        # Use pre-built tools (cached at config load for speed!)
        tools = self._tools

        # SPEED OPTIMIZATION #2: Dynamic max_tokens
        dynamic_max_tokens = self._get_dynamic_max_tokens(user_input.text, tools)

        try:
            response = await self._call_llm_streaming(conversation_id, tools, user_input, dynamic_max_tokens)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response)
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )
        except Exception as err:
            _LOGGER.error("Error processing conversation: %s", err, exc_info=True)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Error: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

    def _get_dynamic_max_tokens(self, query: str, tools: list) -> int:
        """SPEED OPTIMIZATION #2: Reduce max_tokens for simple queries."""
        query_lower = query.lower()
        
        is_simple = any(pattern in query_lower for pattern in SIMPLE_QUERY_PATTERNS)
        
        if is_simple and not tools:
            return min(150, self.max_tokens)
        elif is_simple:
            return min(300, self.max_tokens)
        else:
            return self.max_tokens

    async def _try_native_intent(
        self, user_input: conversation.ConversationInput, conversation_id: str
    ) -> conversation.ConversationResult | None:
        """Try to handle with native intent system using HA's built-in conversation agent."""
        try:
            # Use HA's default conversation agent to parse and handle intent
            result = await conversation.async_converse(
                hass=self.hass,
                text=user_input.text,
                conversation_id=None,  # Fresh conversation
                context=user_input.context,
                language=user_input.language,
                agent_id="conversation.home_assistant",  # Full entity ID
            )
            
            _LOGGER.debug("Native converse response_type: %s", result.response.response_type)

            # Check if we got an intent result
            if hasattr(result.response, 'intent') and result.response.intent is not None:
                intent_type = result.response.intent.intent_type
                _LOGGER.warning("=== NATIVE INTENT MATCHED === type='%s' for: '%s'", intent_type, user_input.text[:80])

                # Check if this intent is in our excluded list
                if intent_type in self.excluded_intents:
                    _LOGGER.warning("=== INTENT EXCLUDED === '%s' - sending to LLM", intent_type)
                    return None
                else:
                    _LOGGER.warning("=== INTENT NOT EXCLUDED === '%s' - will be handled natively", intent_type)

            # ACTION_DONE = command executed (turn on light, etc)
            if result.response.response_type == intent.IntentResponseType.ACTION_DONE:
                # Check for nonsense responses that indicate mismatched intent
                speech = ""
                if result.response.speech:
                    if isinstance(result.response.speech, dict):
                        speech = result.response.speech.get("plain", {}).get("speech", "")
                    else:
                        speech = str(result.response.speech)
                
                # Filter out bad matches - these indicate HA misunderstood
                bad_responses = ["no timers", "no timer", "don't understand", "sorry"]
                if any(bad in speech.lower() for bad in bad_responses):
                    _LOGGER.debug("Filtering bad native response: %s", speech[:50])
                    return None
                    
                _LOGGER.info("Native intent ACTION_DONE for: %s", user_input.text[:50])
                return result
            
            # QUERY_ANSWER = question answered (what's the temperature, etc)  
            if result.response.response_type == intent.IntentResponseType.QUERY_ANSWER:
                _LOGGER.info("Native intent QUERY_ANSWER for: %s", user_input.text[:50])
                return result
                        
        except Exception as err:
            _LOGGER.debug("Native intent exception: %s", err)
        
        return None

    async def _call_llm_streaming(
        self,
        conversation_id: str,
        tools: list[dict],
        user_input: conversation.ConversationInput,
        max_tokens: int,
    ) -> str:
        """Call LLM with streaming and tool support."""
        # Route to appropriate provider
        if self.provider == PROVIDER_ANTHROPIC:
            return await self._call_anthropic(tools, user_input, max_tokens)
        elif self.provider == PROVIDER_GOOGLE:
            return await self._call_google(tools, user_input, max_tokens)
        else:
            # OpenAI-compatible providers (LM Studio, OpenAI, Groq, OpenRouter, Azure, Ollama)
            # Use Responses API for GPT-5 models when enabled (recommended by OpenAI)
            if self.is_gpt5 and self.use_responses_api and self.provider in [PROVIDER_OPENAI, PROVIDER_AZURE, PROVIDER_OPENROUTER]:
                return await self._call_openai_responses_api(tools, user_input, max_tokens)
            return await self._call_openai_compatible(tools, user_input, max_tokens)

    async def _call_anthropic(
        self,
        tools: list[dict],
        user_input: conversation.ConversationInput,
        max_tokens: int,
    ) -> str:
        """Call Anthropic Claude API."""
        if not self._session:
            self._session = async_get_clientsession(self.hass)
        
        # Build system prompt with date
        system_prompt = self.system_prompt or ""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        system_prompt = system_prompt.replace(
            "[CURRENT_DATE_WILL_BE_INJECTED_HERE]",
            f"TODAY'S DATE: {current_date}"
        )
        
        messages = [{"role": "user", "content": user_input.text}]
        
        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
        
        max_iterations = 5
        full_response = ""
        
        for iteration in range(max_iterations):
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
            }
            if anthropic_tools:
                payload["tools"] = anthropic_tools
            
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            
            try:
                self._track_api_call("llm")
                async with self._session.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        _LOGGER.error("Anthropic API error: %s", error)
                        return "Sorry, I couldn't process that request."
                    
                    result = await response.json()
                    
                    # Process response
                    tool_uses = []
                    text_content = ""
                    
                    for block in result.get("content", []):
                        if block.get("type") == "text":
                            text_content += block.get("text", "")
                        elif block.get("type") == "tool_use":
                            tool_uses.append(block)
                    
                    if tool_uses:
                        # Handle tool calls
                        messages.append({"role": "assistant", "content": result.get("content", [])})
                        
                        tool_results = []
                        for tool_use in tool_uses:
                            tool_name = tool_use.get("name")
                            tool_input = tool_use.get("input", {})
                            _LOGGER.info("Tool call: %s(%s)", tool_name, tool_input)
                            
                            result_data = await self._execute_tool(tool_name, tool_input, user_input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use.get("id"),
                                "content": json.dumps(result_data)
                            })
                        
                        messages.append({"role": "user", "content": tool_results})
                        continue
                    
                    if text_content:
                        full_response += text_content
                        return full_response
                    
            except Exception as e:
                _LOGGER.error("Anthropic API exception: %s", e)
                return "Sorry, there was an error processing your request."
        
        return full_response if full_response else "I couldn't complete that request."

    async def _call_google(
        self,
        tools: list[dict],
        user_input: conversation.ConversationInput,
        max_tokens: int,
    ) -> str:
        """Call Google Gemini API."""
        if not self._session:
            self._session = async_get_clientsession(self.hass)
        
        # Build system prompt with date
        system_prompt = self.system_prompt or ""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        system_prompt = system_prompt.replace(
            "[CURRENT_DATE_WILL_BE_INJECTED_HERE]",
            f"TODAY'S DATE: {current_date}"
        )
        
        # Convert tools to Gemini format
        gemini_tools = []
        if tools:
            function_declarations = []
            for tool in tools:
                func = tool.get("function", {})
                function_declarations.append({
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}})
                })
            gemini_tools = [{"functionDeclarations": function_declarations}]
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": user_input.text}]})
        
        max_iterations = 5
        full_response = ""
        
        for iteration in range(max_iterations):
            payload = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": self.temperature,
                }
            }
            if gemini_tools:
                payload["tools"] = gemini_tools
            
            url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
            
            try:
                self._track_api_call("llm")
                async with self._session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        _LOGGER.error("Google API error: %s", error)
                        return "Sorry, I couldn't process that request."
                    
                    result = await response.json()
                    
                    candidates = result.get("candidates", [])
                    if not candidates:
                        return "No response from Gemini."
                    
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    text_content = ""
                    function_calls = []
                    
                    for part in parts:
                        if "text" in part:
                            text_content += part["text"]
                        elif "functionCall" in part:
                            function_calls.append(part["functionCall"])
                    
                    if function_calls:
                        # Handle function calls
                        contents.append({"role": "model", "parts": parts})
                        
                        function_responses = []
                        for fc in function_calls:
                            tool_name = fc.get("name")
                            tool_args = fc.get("args", {})
                            _LOGGER.info("Tool call: %s(%s)", tool_name, tool_args)
                            
                            result_data = await self._execute_tool(tool_name, tool_args, user_input)
                            function_responses.append({
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": result_data
                                }
                            })
                        
                        contents.append({"role": "user", "parts": function_responses})
                        continue
                    
                    if text_content:
                        full_response += text_content
                        return full_response
                    
            except Exception as e:
                _LOGGER.error("Google API exception: %s", e)
                return "Sorry, there was an error processing your request."
        
        return full_response if full_response else "I couldn't complete that request."

    async def _call_openai_compatible(
        self,
        tools: list[dict],
        user_input: conversation.ConversationInput,
        max_tokens: int,
    ) -> str:
        """Call OpenAI-compatible API (LM Studio, OpenAI, Groq, OpenRouter, Azure, Ollama)."""
        messages = []
        
        if self.system_prompt:
            # Inject current date into system prompt
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            system_prompt_with_date = self.system_prompt.replace(
                "[CURRENT_DATE_WILL_BE_INJECTED_HERE]",
                f"TODAY'S DATE: {current_date}"
            )
            messages.append({
                "role": "system",
                "content": system_prompt_with_date
            })
        
        # Add user message directly (STATELESS - no history)
        messages.append({
            "role": "user",
            "content": user_input.text,
        })
        
        max_iterations = 5
        full_response = ""
        called_tools = set()  # Track tool calls to prevent duplicates
        
        for iteration in range(max_iterations):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "top_p": self.top_p,
                "stream": True,
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            accumulated_content = ""
            tool_calls_buffer = []
            
            self._track_api_call("llm")
            stream = await self.client.chat.completions.create(**kwargs)
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                if delta.content:
                    accumulated_content += delta.content
                    full_response += delta.content
                
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.index is not None:
                            while len(tool_calls_buffer) <= tc_delta.index:
                                tool_calls_buffer.append({
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            current = tool_calls_buffer[tc_delta.index]
                            
                            if tc_delta.id:
                                current["id"] = tc_delta.id
                            
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    current["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    current["function"]["arguments"] += tc_delta.function.arguments
            
            valid_tool_calls = [tc for tc in tool_calls_buffer if tc.get("id") and tc.get("function", {}).get("name")]
            
            # Filter out duplicate tool calls (same function + same arguments)
            unique_tool_calls = []
            for tc in valid_tool_calls:
                tool_key = f"{tc['function']['name']}:{tc['function']['arguments']}"
                if tool_key not in called_tools:
                    called_tools.add(tool_key)
                    unique_tool_calls.append(tc)
                else:
                    _LOGGER.debug("Skipping duplicate tool call: %s", tc['function']['name'])
            
            valid_tool_calls = unique_tool_calls

            if valid_tool_calls:
                _LOGGER.info("Processing %d tool call(s)", len(valid_tool_calls))
                
                messages.append({
                    "role": "assistant",
                    "content": accumulated_content if accumulated_content else None,
                    "tool_calls": valid_tool_calls
                })
                
                # SPEED OPTIMIZATION #1: Execute ALL tools in PARALLEL!
                tool_tasks = []
                for tool_call in valid_tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    _LOGGER.info("Tool call: %s(%s)", tool_name, arguments)
                    tool_tasks.append(self._execute_tool(tool_name, arguments, user_input))
                
                # Execute all tools simultaneously
                tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)
                
                # Add results to messages
                for tool_call, result in zip(valid_tool_calls, tool_results):
                    if isinstance(result, Exception):
                        _LOGGER.error("Tool %s failed: %s", tool_call["function"]["name"], result)
                        result = {"error": str(result)}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result),
                    })

                    _LOGGER.debug("Tool %s returned: %s", tool_call["function"]["name"], result)

                continue
            
            if accumulated_content:
                return full_response
            
            break
        
        return full_response if full_response else "I apologize, but I couldn't complete that request."

    async def _call_openai_responses_api(
        self,
        tools: list[dict],
        user_input: conversation.ConversationInput,
        max_tokens: int,
    ) -> str:
        """Call OpenAI Responses API for GPT-5 models (recommended for better performance)."""
        # Build input messages for Responses API format
        input_messages = []

        # Add user message
        input_messages.append({
            "role": "user",
            "content": user_input.text,
        })

        # Prepare instructions (system prompt) for Responses API
        instructions = ""
        if self.system_prompt:
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            instructions = self.system_prompt.replace(
                "[CURRENT_DATE_WILL_BE_INJECTED_HERE]",
                f"TODAY'S DATE: {current_date}"
            )

        max_iterations = 5
        full_response = ""
        called_tools = set()

        for iteration in range(max_iterations):
            # Build the Responses API request
            request_data = {
                "model": self.model,
                "input": input_messages,
                "max_output_tokens": max_tokens,
            }

            # Add instructions (system prompt)
            if instructions:
                request_data["instructions"] = instructions

            # Add reasoning configuration for GPT-5
            if self.reasoning_effort and self.reasoning_effort != "none":
                request_data["reasoning"] = {"effort": self.reasoning_effort}

            # Add tools if available
            if tools:
                request_data["tools"] = tools

            self._track_api_call("llm")

            try:
                # Call the Responses API endpoint
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                async with self._session.post(
                    f"{self.base_url.rstrip('/v1')}/v1/responses",
                    json=request_data,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _LOGGER.error("Responses API error %d: %s", resp.status, error_text)
                        # Fall back to Chat Completions if Responses API fails
                        _LOGGER.info("Falling back to Chat Completions API")
                        return await self._call_openai_compatible(tools, user_input, max_tokens)

                    response_data = await resp.json()

                # Extract the response
                output = response_data.get("output", [])
                output_text = response_data.get("output_text", "")

                # Check for tool calls in the response
                tool_calls = []
                text_content = ""

                for item in output:
                    if item.get("type") == "function_call":
                        tool_calls.append(item)
                    elif item.get("type") == "message":
                        content = item.get("content", [])
                        for c in content:
                            if c.get("type") == "output_text":
                                text_content += c.get("text", "")

                # If no structured output, use output_text
                if not text_content and output_text:
                    text_content = output_text

                if tool_calls:
                    # Process tool calls
                    _LOGGER.info("Processing %d tool call(s) from Responses API", len(tool_calls))

                    # Filter duplicates
                    unique_tool_calls = []
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = json.dumps(tc.get("arguments", {}), sort_keys=True)
                        tool_key = f"{tool_name}:{tool_args}"
                        if tool_key not in called_tools:
                            called_tools.add(tool_key)
                            unique_tool_calls.append(tc)

                    # Execute tools in parallel
                    tool_tasks = []
                    for tc in unique_tool_calls:
                        tool_name = tc.get("name", "")
                        arguments = tc.get("arguments", {})
                        _LOGGER.info("Tool call: %s(%s)", tool_name, arguments)
                        tool_tasks.append(self._execute_tool(tool_name, arguments, user_input))

                    tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                    # Add tool results to messages for next iteration
                    for i, tc in enumerate(unique_tool_calls):
                        result = tool_results[i]
                        if isinstance(result, Exception):
                            result_data = {"error": str(result)}
                        else:
                            result_data = result

                        # Add function call and result to input
                        input_messages.append({
                            "type": "function_call",
                            "call_id": tc.get("call_id", f"call_{i}"),
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments", {}),
                        })
                        input_messages.append({
                            "type": "function_call_output",
                            "call_id": tc.get("call_id", f"call_{i}"),
                            "output": json.dumps(result_data) if isinstance(result_data, dict) else str(result_data),
                        })

                    continue  # Continue to next iteration to get final response

                # No tool calls, we have the final response
                if text_content:
                    full_response += text_content
                    return full_response

            except Exception as e:
                _LOGGER.error("Responses API exception: %s", e, exc_info=True)
                # Fall back to Chat Completions
                _LOGGER.info("Falling back to Chat Completions API due to error")
                return await self._call_openai_compatible(tools, user_input, max_tokens)

            break

        return full_response if full_response else "I apologize, but I couldn't complete that request."

    async def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any], user_input: conversation.ConversationInput
    ) -> dict[str, Any]:
        """Execute a tool."""
        try:
            # Try HA services directly (domain.service format)
            if "." in tool_name:
                domain, service = tool_name.split(".", 1)
                
                # SAFETY: Check if domain is in allowlist
                if domain not in ALLOWED_SERVICE_DOMAINS:
                    _LOGGER.warning("Blocked service call to non-allowlisted domain: %s.%s", domain, service)
                    return {"error": f"Service domain '{domain}' is not allowed for safety reasons. Allowed domains: {', '.join(sorted(ALLOWED_SERVICE_DOMAINS))}"}
                
                service_response = await self.hass.services.async_call(
                    domain, service, arguments, blocking=True, return_response=True
                )
                
                if service_response:
                    return {"success": True, "result": service_response}
                
                return {"success": True, "message": f"Successfully executed {tool_name}"}
            
            # Try custom tool handler (scripts, etc.)
            return await self._custom_tool_handler(tool_name, arguments, user_input.text)
                
        except Exception as err:
            _LOGGER.error("Error executing tool %s: %s", tool_name, err, exc_info=True)
            return {"success": False, "error": str(err)}

    async def _custom_tool_handler(self, tool_name: str, arguments: dict[str, Any], user_query: str = "") -> dict[str, Any]:
        """Handle custom tools."""
        if tool_name == "get_current_datetime":
            now = datetime.now()
            return {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%I:%M %p"),
                "day_of_week": now.strftime("%A"),
                "full_datetime": now.strftime("%A, %B %d, %Y at %I:%M %p"),
                "timestamp": now.isoformat()
            }
        
        elif tool_name == "get_weather_forecast":
            forecast_type = arguments.get("forecast_type", "both")
            
            # API key from config - required
            api_key = self.openweathermap_api_key
            if not api_key:
                return {"error": "OpenWeatherMap API key not configured. Add it in Settings → PolyVoice → API Keys."}
            
            # Use custom location if set, otherwise fall back to defaults
            latitude = self.custom_latitude or self.hass.config.latitude
            longitude = self.custom_longitude or self.hass.config.longitude
            
            try:
                result = {}
                self._track_api_call("weather")
                
                async with asyncio.timeout(API_TIMEOUT):
                    # Get current weather using lat/lon for accuracy
                    current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=imperial"
                    
                    async with self._session.get(current_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            result["current"] = {
                                "temperature": round(data["main"]["temp"]),
                                "feels_like": round(data["main"]["feels_like"]),
                                "humidity": data["main"]["humidity"],
                                "conditions": data["weather"][0]["description"].title(),
                                "wind_speed": round(data["wind"]["speed"]),
                                "location": data["name"]
                            }
                            
                            # Add rain if present
                            if "rain" in data:
                                result["current"]["rain_1h"] = data["rain"].get("1h", 0)
                            
                            _LOGGER.info("Current weather: %s", result["current"])
                        else:
                            _LOGGER.error("Weather API error: %s", response.status)
                            return {"error": "Could not get current weather"}
                    
                    # Get forecast data (for today's high/low AND weekly forecast)
                    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={api_key}&units=imperial"
                    
                    async with self._session.get(forecast_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Get NEXT HOUR rain chance from first forecast entry (3-hour forecast)
                            next_hour_rain = 0
                            if data["list"] and len(data["list"]) > 0:
                                next_hour_rain = round(data["list"][0].get("pop", 0) * 100)
                            result["current"]["rain_chance_next_hour"] = next_hour_rain
                            
                            # Calculate AVERAGE rain chance for next 8 hours (first 3 forecast entries = ~9 hours)
                            rain_chances_8hr = []
                            for i, item in enumerate(data["list"][:3]):  # First 3 entries = 0, 3, 6 hours out
                                rain_chances_8hr.append(item.get("pop", 0) * 100)
                            if rain_chances_8hr:
                                result["current"]["avg_rain_chance_8hr"] = round(sum(rain_chances_8hr) / len(rain_chances_8hr))
                            else:
                                result["current"]["avg_rain_chance_8hr"] = 0
                            
                            # Get TODAY's date for extracting today's high/low
                            today_str = datetime.now().strftime("%Y-%m-%d")
                            today_temps = []
                            
                            # Group by day for weekly forecast
                            daily_forecasts = {}
                            for item in data["list"]:
                                # Parse the datetime
                                dt = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
                                day_key = dt.strftime("%A")
                                item_date = dt.strftime("%Y-%m-%d")
                                
                                # Collect today's temps
                                if item_date == today_str:
                                    today_temps.append(item["main"]["temp_max"])
                                    today_temps.append(item["main"]["temp_min"])
                                
                                if day_key not in daily_forecasts:
                                    daily_forecasts[day_key] = {
                                        "date": dt.strftime("%B %d"),
                                        "high": item["main"]["temp_max"],
                                        "low": item["main"]["temp_min"],
                                        "conditions": item["weather"][0]["description"].title(),
                                        "rain_chance": item.get("pop", 0) * 100
                                    }
                                else:
                                    # Update high/low
                                    daily_forecasts[day_key]["high"] = max(
                                        daily_forecasts[day_key]["high"], 
                                        item["main"]["temp_max"]
                                    )
                                    daily_forecasts[day_key]["low"] = min(
                                        daily_forecasts[day_key]["low"], 
                                        item["main"]["temp_min"]
                                    )
                                    # Take noon conditions if available
                                    if dt.hour == 12:
                                        daily_forecasts[day_key]["conditions"] = item["weather"][0]["description"].title()
                                        daily_forecasts[day_key]["rain_chance"] = item.get("pop", 0) * 100
                            
                            # ADD TODAY'S HIGH/LOW TO CURRENT (for easy LLM access)
                            # Get current day name for lookup
                            current_day = datetime.now().strftime("%A")
                            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A")
                            
                            # Use daily forecast aggregation for today's high
                            if current_day in daily_forecasts:
                                result["current"]["todays_high"] = round(daily_forecasts[current_day]["high"])
                                # For low, use tomorrow's low (overnight low) which is more accurate
                                # since we may have missed this morning's low
                                if tomorrow in daily_forecasts:
                                    result["current"]["todays_low"] = round(daily_forecasts[tomorrow]["low"])
                                else:
                                    result["current"]["todays_low"] = round(daily_forecasts[current_day]["low"])
                            elif today_temps:
                                result["current"]["todays_high"] = round(max(today_temps))
                                result["current"]["todays_low"] = round(min(today_temps))
                            else:
                                # Fallback to first forecast day if no today data
                                first_day = list(daily_forecasts.values())[0] if daily_forecasts else None
                                if first_day:
                                    result["current"]["todays_high"] = round(first_day["high"])
                                    result["current"]["todays_low"] = round(first_day["low"])
                            
                            _LOGGER.info("Today's forecast: high=%s, low=%s, rain_next_hour=%s%%, rain_avg_8hr=%s%%", 
                                        result["current"].get("todays_high"), 
                                        result["current"].get("todays_low"),
                                        result["current"].get("rain_chance_next_hour"),
                                        result["current"].get("avg_rain_chance_8hr"))
                            
                            # Format weekly forecast (only if requested)
                            if forecast_type in ["weekly", "both"]:
                                forecast_list = []
                                for day, forecast in list(daily_forecasts.items())[:5]:
                                    forecast_list.append({
                                        "day": day,
                                        "date": forecast["date"],
                                        "high": round(forecast["high"]),
                                        "low": round(forecast["low"]),
                                        "conditions": forecast["conditions"],
                                        "rain_chance": round(forecast["rain_chance"])
                                    })
                                
                                result["forecast"] = forecast_list
                                _LOGGER.info("Weather forecast: %d days", len(forecast_list))
                
                if not result:
                    return {"error": "No weather data retrieved"}
                    
                return result
                
            except Exception as err:
                _LOGGER.error("Error getting weather: %s", err, exc_info=True)
                return {"error": f"Failed to get weather: {str(err)}"}
        
        elif tool_name == "calculate_age":
            # Look up person's age via Wikipedia/Wikidata
            person_name = arguments.get("person_name", "")
            if not person_name:
                return {"error": "No person name provided"}
            
            try:
                headers = {
                    "User-Agent": "HomeAssistant-LMStudio/1.0 (https://github.com/home-assistant)"
                }
                
                # Search Wikipedia for the person
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={person_name}&srlimit=1"
                async with self._session.get(search_url, headers=headers) as resp:
                    if resp.status != 200:
                        return {"error": "Wikipedia search failed"}
                    search_data = await resp.json()
                
                search_results = search_data.get("query", {}).get("search", [])
                if not search_results:
                    return {"error": f"No Wikipedia article found for {person_name}"}
                
                title = search_results[0].get("title", "")
                
                # Get summary with Wikidata ID
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                async with self._session.get(summary_url, headers=headers) as resp:
                    if resp.status != 200:
                        return {"error": "Failed to get Wikipedia summary"}
                    summary_data = await resp.json()
                
                wikibase_item = summary_data.get("wikibase_item", "")
                
                if not wikibase_item:
                    return {"error": f"No Wikidata ID found for {person_name}"}
                
                # Get birthdate from Wikidata P569 property
                wikidata_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikibase_item}.json"
                async with self._session.get(wikidata_url, headers=headers) as wd_resp:
                    if wd_resp.status != 200:
                        return {"error": "Wikidata lookup failed"}
                    wd_data = await wd_resp.json()
                
                entity = wd_data.get("entities", {}).get(wikibase_item, {})
                claims = entity.get("claims", {})
                
                # P569 is birth date
                if "P569" not in claims:
                    return {"error": f"No birthdate found for {person_name}"}
                
                birth_claim = claims["P569"][0]
                birth_value = birth_claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                birth_time = birth_value.get("time", "")  # Format: +1984-12-30T00:00:00Z
                
                if not birth_time:
                    return {"error": f"Could not parse birthdate for {person_name}"}
                
                # Parse the Wikidata date format
                birth_time = birth_time.lstrip("+")
                birthdate = datetime.strptime(birth_time[:10], "%Y-%m-%d")
                
                # Calculate age
                today = datetime.now()
                age = today.year - birthdate.year
                if (today.month, today.day) < (birthdate.month, birthdate.day):
                    age -= 1
                
                _LOGGER.info("Wikidata birthdate found: %s, age: %d", birthdate.strftime("%B %d, %Y"), age)
                
                return {
                    "person": person_name,
                    "age": age,
                    "birthdate": birthdate.strftime("%B %d, %Y"),
                    "response_text": f"According to Wikipedia, {person_name} was born on {birthdate.strftime('%B %d, %Y')}, making them {age} years old today."
                }
                
            except Exception as err:
                _LOGGER.error("Age calculation error: %s", err)
                return {"error": f"Failed to look up age: {str(err)}"}
        
        elif tool_name == "get_wikipedia_summary":
            # Search Wikipedia for information
            query = arguments.get("topic", "") or arguments.get("query", "")
            if not query:
                return {"error": "No search query provided"}
            
            try:
                headers = {
                    "User-Agent": "HomeAssistant-LMStudio/1.0 (https://home-assistant.io)"
                }
                
                self._track_api_call("wikipedia")
                
                async with asyncio.timeout(API_TIMEOUT):
                    # First, search for pages
                    search_url = "https://en.wikipedia.org/w/api.php"
                    search_params = {
                        "action": "query",
                        "format": "json",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": 1,
                    }
                    
                    _LOGGER.info("Wikipedia search for: %s", query)
                    
                    async with self._session.get(search_url, params=search_params, headers=headers) as resp:
                        _LOGGER.info("Wikipedia search response status: %s", resp.status)
                        if resp.status != 200:
                            return {"error": f"Wikipedia search error: {resp.status}"}
                        
                        search_data = await resp.json()
                        search_results = search_data.get("query", {}).get("search", [])
                        
                        if not search_results:
                            return {"error": f"No Wikipedia articles found for '{query}'"}
                        
                        title = search_results[0].get("title", "")
                        _LOGGER.info("Wikipedia found title: %s", title)
                        
                        # Get the summary using the REST API
                        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                        
                        async with self._session.get(summary_url, headers=headers) as summary_resp:
                            _LOGGER.info("Wikipedia summary response status: %s", summary_resp.status)
                            if summary_resp.status == 200:
                                summary_data = await summary_resp.json()
                                extract = summary_data.get("extract", "")
                                page_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")
                                wikibase_item = summary_data.get("wikibase_item", "")
                                
                                result = {
                                    "title": title,
                                    "summary": extract,
                                    "url": page_url
                                }
                                
                                # Try to get birthdate from Wikidata if we have a wikibase_item
                                if wikibase_item:
                                    try:
                                        wikidata_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikibase_item}.json"
                                        async with self._session.get(wikidata_url, headers=headers) as wd_resp:
                                            if wd_resp.status == 200:
                                                wd_data = await wd_resp.json()
                                                entity = wd_data.get("entities", {}).get(wikibase_item, {})
                                                claims = entity.get("claims", {})
                                                
                                                # P569 is birth date
                                                if "P569" in claims:
                                                    birth_claim = claims["P569"][0]
                                                    time_value = birth_claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "")
                                                    
                                                    if time_value:
                                                        # Format: +1984-12-30T00:00:00Z
                                                        match = re.match(r'\+(\d{4})-(\d{2})-(\d{2})', time_value)
                                                        if match:
                                                            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                                                            birthdate = datetime(year, month, day)
                                                            
                                                            today = datetime.now()
                                                            age = today.year - birthdate.year
                                                            if (today.month, today.day) < (birthdate.month, birthdate.day):
                                                                age -= 1
                                                            
                                                            result["birthdate"] = birthdate.strftime("%B %d, %Y")
                                                            result["current_age"] = age
                                                            result["age_note"] = f"Born {birthdate.strftime('%B %d, %Y')}, currently {age} years old"
                                                            _LOGGER.info("Wikidata birthdate found: %s, age: %d", result["birthdate"], age)
                                    except Exception as wd_err:
                                        _LOGGER.warning("Could not fetch Wikidata: %s", wd_err)
                                
                                return result
                            else:
                                # Fallback to snippet from search
                                snippet = search_results[0].get("snippet", "")
                                snippet = re.sub(r"<[^>]+>", "", snippet)  # Remove HTML tags
                                return {
                                    "title": title,
                                    "summary": snippet
                                }
                                
            except Exception as err:
                _LOGGER.error("Wikipedia search error: %s", err)
                return {"error": f"Failed to search Wikipedia: {str(err)}"}
        
        elif tool_name == "get_calendar_events":
            # Get calendar events from Home Assistant
            query_type = arguments.get("query_type", "upcoming").lower()
            
            try:
                now = datetime.now(dt_util.get_time_zone(self.hass.config.time_zone))
                
                # Determine time range based on query type
                if query_type == "today":
                    # Today's events only
                    start_time = now.replace(hour=0, minute=0, second=0)
                    end_time = now.replace(hour=23, minute=59, second=59)
                    max_results = 100
                    period_desc = "today"
                elif query_type == "tomorrow":
                    # Tomorrow's events only
                    tomorrow = now + timedelta(days=1)
                    start_time = tomorrow.replace(hour=0, minute=0, second=0)
                    end_time = tomorrow.replace(hour=23, minute=59, second=59)
                    max_results = 100
                    period_desc = "tomorrow"
                elif query_type == "week":
                    # ALL events this week (next 7 days)
                    start_time = now
                    end_time = now + timedelta(days=7)
                    max_results = 100
                    period_desc = "this week"
                elif query_type == "month":
                    # ALL events this month (next 30 days)
                    start_time = now
                    end_time = now + timedelta(days=30)
                    max_results = 100
                    period_desc = "this month"
                elif query_type == "upcoming":
                    # Next 5 events
                    start_time = now
                    end_time = now + timedelta(days=365)
                    max_results = 5
                    period_desc = "upcoming"
                elif query_type == "birthday":
                    # Next birthday ONLY
                    start_time = now
                    end_time = now + timedelta(days=365)
                    max_results = 1
                    period_desc = "next birthday"
                else:
                    # Default to upcoming
                    start_time = now
                    end_time = now + timedelta(days=365)
                    max_results = 5
                    period_desc = "upcoming"
                
                _LOGGER.info("Calendar search (%s): %s to %s, max_results=%d",
                            query_type, start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"), max_results)

                # Use configured calendar entities
                all_calendar_entities = self.calendar_entities if self.calendar_entities else []

                if not all_calendar_entities:
                    # If no calendars configured, try to find any available
                    all_states = self.hass.states.async_all()
                    all_calendar_entities = [s.entity_id for s in all_states if s.entity_id.startswith("calendar.")]
                    _LOGGER.info("No calendars configured, auto-discovered: %s", all_calendar_entities)

                # Filter to birthday calendar only if query_type is "birthday"
                if query_type == "birthday":
                    calendar_entities = [c for c in all_calendar_entities if "birthday" in c.lower()]
                    if not calendar_entities:
                        calendar_entities = all_calendar_entities  # Fall back to all if no birthday calendar found
                    _LOGGER.info("Birthday query - using calendars: %s", calendar_entities)
                else:
                    calendar_entities = all_calendar_entities
                
                # Verify calendars exist
                existing_calendars = []
                for cal in calendar_entities:
                    cal_state = self.hass.states.get(cal)
                    if cal_state:
                        existing_calendars.append(cal)
                        _LOGGER.info("Found calendar: %s (state: %s)", cal, cal_state.state)
                    else:
                        _LOGGER.warning("Calendar not found: %s", cal)
                
                if not existing_calendars:
                    # Try to find any calendar entities
                    all_states = self.hass.states.async_all()
                    found_calendars = [s.entity_id for s in all_states if s.entity_id.startswith("calendar.")]
                    _LOGGER.info("Available calendars in HA: %s", found_calendars)
                    return {"error": f"No calendars found. Available: {found_calendars}"}
                
                all_events = []
                
                for cal_entity in existing_calendars:
                    try:
                        _LOGGER.info("Querying calendar: %s", cal_entity)
                        # Call calendar.get_events service
                        result = await self.hass.services.async_call(
                            "calendar",
                            "get_events",
                            {
                                "entity_id": cal_entity,
                                "start_date_time": start_time.isoformat(),
                                "end_date_time": end_time.isoformat(),
                            },
                            blocking=True,
                            return_response=True,
                        )
                        
                        _LOGGER.info("Calendar %s raw result: %s", cal_entity, result)
                        
                        if result and cal_entity in result:
                            events = result[cal_entity].get("events", [])
                            _LOGGER.info("Calendar %s returned %d events", cal_entity, len(events))
                            for event in events:
                                event_start = event.get("start")
                                event_summary = event.get("summary", "Untitled")
                                
                                # Parse start time for sorting
                                try:
                                    if "T" in str(event_start):
                                        event_dt = datetime.fromisoformat(str(event_start).replace("Z", "+00:00"))
                                        event_dt = event_dt.astimezone(dt_util.get_time_zone(self.hass.config.time_zone))
                                        time_str = event_dt.strftime("%B %d at %I:%M %p")
                                        sort_key = event_dt
                                    else:
                                        event_dt = datetime.strptime(str(event_start), "%Y-%m-%d")
                                        event_dt = event_dt.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                                        time_str = event_dt.strftime("%B %d") + " (all day)"
                                        sort_key = event_dt
                                except Exception as parse_err:
                                    _LOGGER.warning("Date parse error for %s: %s", event_start, parse_err)
                                    time_str = str(event_start)
                                    sort_key = now + timedelta(days=9999)  # Put parse errors at end
                                
                                is_birthday = "birthday" in cal_entity.lower()
                                
                                all_events.append({
                                    "title": event_summary,
                                    "time": time_str,
                                    "is_birthday": is_birthday,
                                    "calendar": "birthdays" if is_birthday else "main",
                                    "_sort_key": sort_key
                                })
                        else:
                            _LOGGER.warning("No events key in result for %s", cal_entity)
                    except Exception as cal_err:
                        _LOGGER.error("Error getting events from %s: %s", cal_entity, cal_err, exc_info=True)
                
                # Sort ALL events by actual date (combined)
                all_events.sort(key=lambda x: x["_sort_key"])
                
                # Remove sort key from output
                for e in all_events:
                    del e["_sort_key"]
                
                if not all_events:
                    return {
                        "query_type": query_type,
                        "period": period_desc,
                        "message": f"No events or birthdays found for {period_desc}",
                        "events": []
                    }
                
                # Return results based on query type
                result_events = all_events[:max_results]
                
                result = {
                    "query_type": query_type,
                    "period": period_desc,
                    "event_count": len(result_events),
                    "events": result_events
                }
                
                # Add helpful context based on query type
                if query_type == "next":
                    result["next_event"] = result_events[0] if result_events else None
                elif query_type == "week" or query_type == "month":
                    result["total_events"] = len(result_events)
                    if len(all_events) > max_results:
                        result["note"] = f"Showing all {len(result_events)} events"
                
                _LOGGER.info("Calendar events (combined): %s", result)
                return result
                
            except Exception as err:
                _LOGGER.error("Error getting calendar events: %s", err, exc_info=True)
                return {"error": f"Failed to get calendar events: {str(err)}"}
        
        elif tool_name == "get_sports_info":
            # Get sports info from ESPN API with dynamic team search
            team_name = arguments.get("team_name", "")
            query_type = arguments.get("query_type", "both")

            if not team_name:
                return {"error": "No team name provided"}

            try:
                headers = {"User-Agent": "HomeAssistant-PolyVoice/1.0"}
                team_key = team_name.lower().strip()

                # Check for league-specific keywords to prioritize search
                champions_league_keywords = ["champions league", "ucl", "champions"]
                prioritize_ucl = any(kw in team_key for kw in champions_league_keywords)
                # Clean team name if it contains league keywords
                for kw in champions_league_keywords:
                    team_key = team_key.replace(kw, "").strip()

                # Search for team in major leagues directly (search API is deprecated)
                # Order matters - first match wins (unless UCL is prioritized)
                if prioritize_ucl:
                    leagues_to_try = [
                        ("soccer", "uefa.champions"),  # Champions League FIRST
                        ("soccer", "eng.1"),  # Premier League
                        ("basketball", "nba"),
                        ("football", "nfl"),
                        ("baseball", "mlb"),
                        ("hockey", "nhl"),
                        ("football", "college-football"),  # NCAA Football
                        ("basketball", "mens-college-basketball"),  # NCAA Basketball
                    ]
                else:
                    leagues_to_try = [
                        ("basketball", "nba"),
                        ("football", "nfl"),
                        ("baseball", "mlb"),
                        ("hockey", "nhl"),
                        ("soccer", "eng.1"),  # Premier League
                        ("soccer", "uefa.champions"),  # Champions League
                        ("football", "college-football"),  # NCAA Football
                        ("basketball", "mens-college-basketball"),  # NCAA Basketball
                    ]

                team_found = False
                url = None
                full_name = team_name

                # Two-pass search: exact abbreviation match first, then substring match
                for match_type in ["abbrev", "name"]:
                    if team_found:
                        break
                    for sport, league in leagues_to_try:
                        teams_url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams?limit=500"
                        async with self._session.get(teams_url, headers=headers) as teams_resp:
                            if teams_resp.status == 200:
                                teams_data = await teams_resp.json()
                                for team in teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
                                    t = team.get("team", {})
                                    match = False
                                    if match_type == "abbrev":
                                        match = team_key == t.get("abbreviation", "").lower()
                                    else:
                                        match = (team_key in t.get("displayName", "").lower() or
                                                team_key in t.get("shortDisplayName", "").lower() or
                                                team_key in t.get("nickname", "").lower())
                                    if match:
                                        team_id = t.get("id", "")
                                        full_name = t.get("displayName", team_name)
                                        url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/schedule"
                                        team_found = True
                                        break
                        if team_found:
                            break

                if not team_found:
                    return {"error": f"Team '{team_name}' not found. Try the full team name (e.g., 'Miami Heat', 'New York Yankees')"}
                
                async with self._session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return {"error": f"ESPN API error: {resp.status}"}
                    data = await resp.json()
                
                events = data.get("events", [])
                
                if not events:
                    return {"error": f"No scheduled games found for {full_name}"}
                
                result = {"team": full_name}
                
                # Find last completed game, live game, and next upcoming game
                now = datetime.now()
                last_game = None
                next_game = None
                live_game = None

                for event in events:
                    status_info = event.get("competitions", [{}])[0].get("status", {}).get("type", {})
                    is_completed = status_info.get("completed", False)
                    status_name = status_info.get("name", "")
                    game_date_str = event.get("date", "")

                    if game_date_str:
                        try:
                            game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                            game_date_naive = game_date.replace(tzinfo=None)

                            if status_name == "STATUS_IN_PROGRESS":  # Live game
                                live_game = event
                            elif is_completed:  # Completed game
                                if last_game is None or game_date_naive > datetime.fromisoformat(last_game.get("date", "2000-01-01").replace("Z", "+00:00")).replace(tzinfo=None):
                                    last_game = event
                            else:  # Upcoming game
                                if next_game is None:
                                    next_game = event
                        except (ValueError, KeyError, TypeError):
                            pass
                
                # Format last game result
                if query_type in ["last_game", "both"] and last_game:
                    comp = last_game.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [])
                    home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                    away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})

                    home_name = home_team.get("team", {}).get("displayName", "Home")
                    away_name = away_team.get("team", {}).get("displayName", "Away")
                    # Score can be a dict {'value': 34.0, 'displayValue': '34'} or a string
                    home_score_raw = home_team.get("score", "0")
                    away_score_raw = away_team.get("score", "0")
                    home_score = home_score_raw.get("displayValue", home_score_raw) if isinstance(home_score_raw, dict) else home_score_raw
                    away_score = away_score_raw.get("displayValue", away_score_raw) if isinstance(away_score_raw, dict) else away_score_raw
                    
                    game_date = last_game.get("date", "")[:10]
                    
                    result["last_game"] = {
                        "date": game_date,
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_score": home_score,
                        "away_score": away_score,
                        "summary": f"{away_name} {away_score} @ {home_name} {home_score}"
                    }

                # Format live game (takes priority - always shown if there's a game in progress)
                if live_game:
                    comp = live_game.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [])
                    home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                    away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})

                    home_name = home_team.get("team", {}).get("displayName", "Home")
                    away_name = away_team.get("team", {}).get("displayName", "Away")
                    # Score can be a dict {'value': 34.0, 'displayValue': '34'} or a string
                    home_score_raw = home_team.get("score", "0")
                    away_score_raw = away_team.get("score", "0")
                    home_score = home_score_raw.get("displayValue", home_score_raw) if isinstance(home_score_raw, dict) else home_score_raw
                    away_score = away_score_raw.get("displayValue", away_score_raw) if isinstance(away_score_raw, dict) else away_score_raw

                    # Get game clock/period info
                    status_detail = comp.get("status", {}).get("type", {}).get("detail", "In Progress")

                    result["live_game"] = {
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_score": home_score,
                        "away_score": away_score,
                        "status": status_detail,
                        "summary": f"LIVE: {away_name} {away_score} @ {home_name} {home_score} ({status_detail})"
                    }

                # Format next game
                if query_type in ["next_game", "both"] and next_game:
                    comp = next_game.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [])
                    home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                    away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})
                    
                    home_name = home_team.get("team", {}).get("displayName", "Home")
                    away_name = away_team.get("team", {}).get("displayName", "Away")
                    
                    game_date_str = next_game.get("date", "")
                    if game_date_str:
                        try:
                            from zoneinfo import ZoneInfo
                            game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                            # Convert to HA configured timezone
                            game_dt_local = game_dt.astimezone(dt_util.get_time_zone(self.hass.config.time_zone))
                            formatted_date = game_dt_local.strftime("%A, %B %d at %I:%M %p")
                        except (ValueError, KeyError, TypeError, AttributeError):
                            formatted_date = game_date_str[:10]
                    else:
                        formatted_date = "TBD"
                    
                    venue = comp.get("venue", {}).get("fullName", "")
                    
                    result["next_game"] = {
                        "date": formatted_date,
                        "home_team": home_name,
                        "away_team": away_name,
                        "venue": venue,
                        "summary": f"{away_name} @ {home_name} - {formatted_date}"
                    }

                # Get standings if requested
                if query_type == "standings":
                    try:
                        # Extract sport and league from the schedule URL we built
                        # URL format: https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/schedule
                        url_parts = url.split("/")
                        sport_idx = url_parts.index("sports") + 1
                        standings_sport = url_parts[sport_idx]
                        standings_league = url_parts[sport_idx + 1]

                        standings_url = f"https://site.api.espn.com/apis/v2/sports/{standings_sport}/{standings_league}/standings"
                        async with self._session.get(standings_url, headers=headers) as standings_resp:
                            if standings_resp.status == 200:
                                standings_data = await standings_resp.json()

                                # Find team in standings
                                team_standing = None
                                conference_name = ""

                                # Handle different structures (conferences vs flat)
                                if "children" in standings_data:
                                    # Conference-based (NBA, NFL, etc.)
                                    for conf in standings_data.get("children", []):
                                        entries = conf.get("standings", {}).get("entries", [])
                                        # Sort by wins
                                        sorted_entries = sorted(entries, key=lambda x: int(next((s.get("value", 0) for s in x.get("stats", []) if s["name"] == "wins"), 0)), reverse=True)
                                        for rank, entry in enumerate(sorted_entries, 1):
                                            if entry.get("team", {}).get("displayName", "").lower() == full_name.lower():
                                                team_standing = {"rank": rank, "entry": entry}
                                                conference_name = conf.get("name", "")
                                                break
                                        if team_standing:
                                            break
                                else:
                                    # Flat standings (soccer leagues)
                                    entries = standings_data.get("standings", {}).get("entries", [])
                                    sorted_entries = sorted(entries, key=lambda x: int(next((s.get("value", 0) for s in x.get("stats", []) if s["name"] in ["wins", "points"]), 0)), reverse=True)
                                    for rank, entry in enumerate(sorted_entries, 1):
                                        if entry.get("team", {}).get("displayName", "").lower() == full_name.lower():
                                            team_standing = {"rank": rank, "entry": entry}
                                            break

                                if team_standing:
                                    entry = team_standing["entry"]
                                    stats = {s["name"]: s.get("displayValue", s.get("value")) for s in entry.get("stats", [])}
                                    wins = stats.get("wins", "?")
                                    losses = stats.get("losses", "?")
                                    rank = team_standing["rank"]

                                    result["standings"] = {
                                        "rank": rank,
                                        "wins": wins,
                                        "losses": losses,
                                        "conference": conference_name,
                                        "summary": f"{full_name} is #{rank} in the {conference_name or 'league'} with a {wins}-{losses} record"
                                    }
                    except Exception as standings_err:
                        _LOGGER.warning("Could not get standings: %s", standings_err)

                # Build response text
                response_parts = []
                if "live_game" in result:
                    lg = result["live_game"]
                    response_parts.append(lg['summary'])
                if "last_game" in result:
                    lg = result["last_game"]
                    response_parts.append(f"Last game: {lg['summary']} on {lg['date']}")
                if "next_game" in result:
                    ng = result["next_game"]
                    response_parts.append(f"Next game: {ng['summary']}")
                if "standings" in result:
                    st = result["standings"]
                    response_parts.append(st['summary'])

                result["response_text"] = ". ".join(response_parts) if response_parts else f"No game info found for {full_name}"
                
                _LOGGER.info("Sports info for %s: %s", full_name, result.get("response_text", ""))
                return result
                
            except Exception as err:
                _LOGGER.error("Sports API error: %s", err, exc_info=True)
                return {"error": f"Failed to get sports info: {str(err)}"}

        elif tool_name == "get_ufc_info":
            # Get UFC/MMA event information from ESPN API
            query_type = arguments.get("query_type", "next_event")

            try:
                self._track_api_call("sports")
                headers = {"User-Agent": "HomeAssistant-PolyVoice/1.0"}

                async with asyncio.timeout(API_TIMEOUT):
                    events_url = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
                    async with self._session.get(events_url, headers=headers) as resp:
                        if resp.status != 200:
                            return {"error": f"ESPN UFC API error: {resp.status}"}
                        data = await resp.json()

                # Get upcoming events from calendar
                leagues = data.get("leagues", [{}])
                calendar = leagues[0].get("calendar", []) if leagues else []

                if not calendar:
                    return {"error": "No upcoming UFC events found"}

                result = {"events": []}

                for event in calendar[:5]:  # Get up to 5 upcoming events
                    event_info = {
                        "name": event.get("label", "Unknown Event"),
                        "date": event.get("startDate", "")[:10] if event.get("startDate") else "TBD"
                    }
                    # Format date nicely
                    if event_info["date"] and event_info["date"] != "TBD":
                        try:
                            event_dt = datetime.fromisoformat(event_info["date"])
                            event_info["formatted_date"] = event_dt.strftime("%B %d, %Y")
                        except:
                            event_info["formatted_date"] = event_info["date"]
                    result["events"].append(event_info)

                if query_type == "next_event" and result["events"]:
                    next_evt = result["events"][0]
                    result["response_text"] = f"The next UFC event is {next_evt['name']} on {next_evt.get('formatted_date', next_evt['date'])}."
                elif query_type == "upcoming" and result["events"]:
                    event_list = [f"{e['name']} ({e.get('formatted_date', e['date'])})" for e in result["events"]]
                    result["response_text"] = "Upcoming UFC events: " + ", ".join(event_list)
                else:
                    result["response_text"] = "No upcoming UFC events found."

                _LOGGER.info("UFC info: %s", result.get("response_text", ""))
                return result

            except Exception as err:
                _LOGGER.error("UFC API error: %s", err, exc_info=True)
                return {"error": f"Failed to get UFC info: {str(err)}"}

        elif tool_name == "get_stock_price":
            # Get stock price from Yahoo Finance API (free, no key required)
            symbol = arguments.get("symbol", "").upper().strip()

            if not symbol:
                return {"error": "No stock symbol provided"}

            # Common company name to symbol mappings
            company_to_symbol = {
                "apple": "AAPL", "tesla": "TSLA", "google": "GOOGL", "alphabet": "GOOGL",
                "microsoft": "MSFT", "amazon": "AMZN", "meta": "META", "facebook": "META",
                "nvidia": "NVDA", "netflix": "NFLX", "disney": "DIS", "nike": "NKE",
                "coca-cola": "KO", "coke": "KO", "pepsi": "PEP", "walmart": "WMT",
                "costco": "COST", "starbucks": "SBUX", "mcdonalds": "MCD", "boeing": "BA",
                "intel": "INTC", "amd": "AMD", "paypal": "PYPL", "visa": "V",
                "mastercard": "MA", "jpmorgan": "JPM", "goldman": "GS", "berkshire": "BRK-B",
                "johnson": "JNJ", "pfizer": "PFE", "moderna": "MRNA", "uber": "UBER",
                "lyft": "LYFT", "airbnb": "ABNB", "spotify": "SPOT", "snap": "SNAP",
                "twitter": "X", "x": "X", "salesforce": "CRM", "oracle": "ORCL",
                "ibm": "IBM", "cisco": "CSCO", "adobe": "ADBE", "zoom": "ZM",
            }

            # Convert company name to symbol if needed
            symbol_lookup = symbol.lower()
            if symbol_lookup in company_to_symbol:
                symbol = company_to_symbol[symbol_lookup]

            try:
                self._track_api_call("stocks")
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

                async with asyncio.timeout(API_TIMEOUT):
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
                    async with self._session.get(url, headers=headers) as resp:
                        if resp.status != 200:
                            return {"error": f"Could not find stock symbol '{symbol}'"}
                        data = await resp.json()

                result_data = data.get("chart", {}).get("result", [{}])[0]
                meta = result_data.get("meta", {})

                if not meta or "regularMarketPrice" not in meta:
                    return {"error": f"Could not find stock data for '{symbol}'"}

                price = meta.get("regularMarketPrice", 0)
                prev_close = meta.get("previousClose", meta.get("chartPreviousClose", price))
                change = price - prev_close if prev_close else 0
                pct_change = (change / prev_close * 100) if prev_close else 0
                company_name = meta.get("shortName", meta.get("longName", symbol))

                # Format the response
                direction = "up" if change >= 0 else "down"
                result = {
                    "symbol": symbol,
                    "company": company_name,
                    "price": round(price, 2),
                    "change": round(change, 2),
                    "percent_change": round(pct_change, 2),
                    "previous_close": round(prev_close, 2) if prev_close else None,
                    "response_text": f"{company_name} ({symbol}) is at ${price:.2f}, {direction} ${abs(change):.2f} ({pct_change:+.2f}%) today."
                }

                _LOGGER.info("Stock price for %s: %s", symbol, result.get("response_text", ""))
                return result

            except Exception as err:
                _LOGGER.error("Stock API error: %s", err, exc_info=True)
                return {"error": f"Failed to get stock price: {str(err)}"}

        elif tool_name == "get_news":
            # Get news from TheNewsAPI.com (free tier - real-time!)
            
            # API key from config - required
            api_key = self.newsapi_key
            if not api_key:
                return {"error": "TheNewsAPI key not configured. Add it in Settings → PolyVoice → API Keys."}
            
            category = arguments.get("category", "")
            topic = arguments.get("topic", "")
            max_results = min(arguments.get("max_results", 5), 25)  # Basic plan: up to 25 articles
            
            # TheNewsAPI categories: general, science, sports, business, health, entertainment, tech, politics, food, travel
            valid_categories = ["general", "science", "sports", "business", "health", "entertainment", "tech", "politics", "food", "travel"]
            
            # Map common aliases
            category_map = {
                "technology": "tech",
                "world": "general",
                "nation": "politics",
            }
            
            # Categories that should be treated as topic searches (not native API categories)
            topic_categories = ["fashion", "gaming", "crypto", "cryptocurrency", "auto", "automotive", "cars", "real estate", "weather", "local"]
            
            try:
                base_url = "https://api.thenewsapi.com/v1/news"
                
                # Basic plan: up to 25 articles per request
                fetch_limit = 25
                
                if topic:
                    # Search for specific topic using /all endpoint
                    from urllib.parse import quote
                    encoded_topic = quote(topic)
                    url = f"{base_url}/all?api_token={api_key}&search={encoded_topic}&language=en&limit={fetch_limit}"
                    display_topic = topic
                elif category:
                    # Map category if needed
                    mapped_category = category_map.get(category.lower(), category.lower())
                    
                    if mapped_category in valid_categories:
                        # Use native category endpoint
                        url = f"{base_url}/top?api_token={api_key}&locale=us&language=en&categories={mapped_category}&limit={fetch_limit}"
                        display_topic = f"{category.title()} News"
                    elif mapped_category in topic_categories or mapped_category not in valid_categories:
                        # Treat as topic search instead of defaulting to general
                        from urllib.parse import quote
                        encoded_topic = quote(mapped_category)
                        url = f"{base_url}/all?api_token={api_key}&search={encoded_topic}&language=en&limit={fetch_limit}"
                        display_topic = f"{category.title()} News"
                else:
                    # Default top headlines
                    url = f"{base_url}/top?api_token={api_key}&locale=us&language=en&limit={fetch_limit}"
                    display_topic = "Top Headlines"
                
                _LOGGER.info("Fetching news from TheNewsAPI: %s", display_topic)
                
                # Add cache-busting headers
                headers = {
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
                
                self._track_api_call("news")
                
                async with asyncio.timeout(API_TIMEOUT):
                    async with self._session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            articles = data.get("data", [])
                            
                            if not articles:
                                return {"message": f"No news found for '{display_topic}'"}
                            
                            headlines = []
                            seen_titles = set()  # Dedupe by title instead (avoid exact duplicates)
                            blocked_sources = {"yahoo.com", "finance.yahoo.com"}  # Yahoo sucks - tags everything wrong
                            blocked_count = 0
                            
                            for article in articles:
                                # Stop once we have enough articles
                                if len(headlines) >= max_results:
                                    break
                                
                                source_name = article.get("source", "Unknown")
                                title = article.get("title", "No title")
                                
                                # Skip blocked sources (Yahoo tags everything as all categories)
                                if source_name in blocked_sources:
                                    blocked_count += 1
                                    continue
                                
                                # Skip exact duplicate titles
                                if title in seen_titles:
                                    continue
                                seen_titles.add(title)
                                
                                title = article.get("title", "No title")
                                description = article.get("description", "")
                                snippet = article.get("snippet", "")
                                published_at = article.get("published_at", "")
                                
                                # Use snippet if description is empty
                                summary = description if description else snippet
                                if summary and len(summary) > 200:
                                    summary = summary[:200] + "..."
                                
                                # Parse date to Eastern time
                                date_text = ""
                                if published_at:
                                    try:
                                        # TheNewsAPI format: 2025-12-18T15:32:20.000000Z
                                        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                                        dt_local = dt.astimezone(dt_util.get_time_zone(self.hass.config.time_zone))
                                        date_text = dt_local.strftime("%B %d at %I:%M %p")
                                    except (ValueError, KeyError, TypeError, AttributeError):
                                        date_text = published_at
                                
                                headlines.append({
                                    "headline": title,
                                    "summary": summary,
                                    "source": source_name,
                                    "published": date_text
                                })
                            
                            result = {
                                "topic": display_topic,
                                "article_count": len(headlines),
                                "articles": headlines
                            }
                            
                            _LOGGER.info("TheNewsAPI: %d returned, %d blocked (Yahoo), %d passed filters", len(articles), blocked_count, len(headlines))
                            return result
                        
                        elif response.status == 401:
                            return {"error": "Invalid TheNewsAPI key. Check your API token."}
                        elif response.status == 402:
                            return {"error": "TheNewsAPI usage limit reached."}
                        elif response.status == 429:
                            return {"error": "TheNewsAPI rate limit exceeded. Try again later."}
                        else:
                            error_data = await response.text()
                            _LOGGER.error("TheNewsAPI error: %s - %s", response.status, error_data)
                            return {"error": f"Failed to fetch news: HTTP {response.status}"}
                
            except Exception as err:
                _LOGGER.error("Error getting news: %s", err, exc_info=True)
                return {"error": f"Failed to get news: {str(err)}"}
        
        elif tool_name == "get_sports_scores":
            # Get sports scores from ESPN unofficial API
            
            sport = arguments.get("sport", "nba").lower()
            team = arguments.get("team", "")
            days_back = min(arguments.get("days_back", 3), 7)  # Can go back up to 7 days
            
            # ESPN API endpoints
            sport_endpoints = {
                "nba": "basketball/nba",
                "nfl": "football/nfl",
                "nhl": "hockey/nhl",
                "mlb": "baseball/mlb",
                "mls": "soccer/usa.1",
                "epl": "soccer/eng.1",
                "ncaaf": "football/college-football",
                "ncaab": "basketball/mens-college-basketball"
            }
            
            # Team name mappings for filtering
            team_aliases = {
                "heat": "miami heat",
                "panthers": "florida panthers",
                "dolphins": "miami dolphins",
                "man city": "manchester city",
                "city": "manchester city",
                "inter miami": "inter miami",
                "marlins": "miami marlins",
                "united": "manchester united",
                "liverpool": "liverpool",
                "arsenal": "arsenal",
                "chelsea": "chelsea",
                "spurs": "tottenham hotspur",
                "tottenham": "tottenham hotspur",
            }

            if sport not in sport_endpoints:
                return {"error": f"Unknown sport: {sport}. Supported: nba, nfl, nhl, mlb, mls, epl, ncaaf, ncaab"}
            
            endpoint = sport_endpoints[sport]
            base_url = f"http://site.api.espn.com/apis/site/v2/sports/{endpoint}/scoreboard"
            
            completed_games = []
            upcoming_games = []
            live_games = []
            
            try:
                self._track_api_call("sports")
                
                async with asyncio.timeout(API_TIMEOUT):
                    # Check past days for completed games
                    dates_to_check = []
                    for i in range(days_back + 1):
                        check_date = datetime.now() - timedelta(days=i)
                        dates_to_check.append(check_date.strftime("%Y%m%d"))
                    
                    # Also check next 7 days for upcoming games
                    for i in range(1, 8):
                        future_date = datetime.now() + timedelta(days=i)
                        dates_to_check.append(future_date.strftime("%Y%m%d"))
                    
                    for date_str in dates_to_check:
                        url = f"{base_url}?dates={date_str}"
                        _LOGGER.info("Fetching ESPN scores from: %s", url)
                        
                        async with self._session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                events = data.get("events", [])
                                
                                for event in events:
                                    competition = event.get("competitions", [{}])[0]
                                    competitors = competition.get("competitors", [])
                                    
                                    if len(competitors) < 2:
                                        continue
                                    
                                    # Get team info
                                    home_team = None
                                    away_team = None
                                    for comp in competitors:
                                        team_info = {
                                            "name": comp.get("team", {}).get("displayName", "Unknown"),
                                            "abbreviation": comp.get("team", {}).get("abbreviation", ""),
                                            "score": comp.get("score", "0"),
                                            "winner": comp.get("winner", False)
                                        }
                                        if comp.get("homeAway") == "home":
                                            home_team = team_info
                                        else:
                                            away_team = team_info
                                    
                                    if not home_team or not away_team:
                                        continue
                                    
                                    # Filter by team if specified
                                    if team:
                                        team_lower = team_aliases.get(team.lower(), team.lower())
                                        home_name = home_team["name"].lower()
                                        away_name = away_team["name"].lower()
                                        
                                        if team_lower not in home_name and team_lower not in away_name:
                                            # Also check abbreviation
                                            if team.upper() != home_team["abbreviation"] and team.upper() != away_team["abbreviation"]:
                                                continue
                                    
                                    # Get game status
                                    status = event.get("status", {})
                                    state = status.get("type", {}).get("state", "")  # pre, in, post
                                    status_detail = status.get("type", {}).get("shortDetail", "")
                                    
                                    # Get game time
                                    game_date = event.get("date", "")
                                    try:
                                        game_dt = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
                                        game_local = game_dt.astimezone(dt_util.get_time_zone(self.hass.config.time_zone))
                                        game_time = game_local.strftime("%B %d at %I:%M %p")
                                        game_date_short = game_local.strftime("%m/%d")
                                    except (ValueError, KeyError, TypeError, AttributeError):
                                        game_time = game_date
                                        game_date_short = ""
                                    
                                    home_score = int(home_team["score"]) if home_team["score"].isdigit() else 0
                                    away_score = int(away_team["score"]) if away_team["score"].isdigit() else 0
                                    
                                    if state == "post":
                                        # Completed game
                                        if home_score > away_score:
                                            result = f"{home_team['name']} beat {away_team['name']} {home_score}-{away_score}"
                                        elif away_score > home_score:
                                            result = f"{away_team['name']} beat {home_team['name']} {away_score}-{home_score}"
                                        else:
                                            result = f"{home_team['name']} and {away_team['name']} tied {home_score}-{away_score}"
                                        
                                        # Add date for context
                                        if game_date_short:
                                            result += f" ({game_date_short})"
                                        
                                        # Avoid duplicates
                                        if result not in completed_games:
                                            completed_games.append(result)
                                    
                                    elif state == "in":
                                        # Live game
                                        live_info = f"LIVE: {away_team['name']} {away_score} at {home_team['name']} {home_score} ({status_detail})"
                                        if live_info not in live_games:
                                            live_games.append(live_info)
                                    
                                    elif state == "pre":
                                        # Upcoming game
                                        upcoming_info = f"{away_team['name']} at {home_team['name']} on {game_time}"
                                        if upcoming_info not in upcoming_games:
                                            upcoming_games.append(upcoming_info)
                            else:
                                _LOGGER.warning("ESPN API returned status %s", response.status)
                
                # Build response with clear sections
                result = {}
                
                if completed_games:
                    result["last_game"] = completed_games[0]  # Most recent completed
                    if len(completed_games) > 1:
                        result["recent_results"] = completed_games[:5]
                
                if live_games:
                    result["live"] = live_games
                
                if upcoming_games:
                    result["next_game"] = upcoming_games[0]  # Next upcoming
                
                if not result:
                    return {"message": f"No games found for {team or sport} in the last {days_back} days."}
                
                _LOGGER.info("Sports scores result: %s", result)
                return result
                
            except Exception as err:
                _LOGGER.error("Error getting sports scores: %s", err, exc_info=True)
                return {"error": f"Failed to get sports scores: {str(err)}"}
        
        elif tool_name == "find_nearby_places":
            query = arguments.get("query", "")
            max_results = min(arguments.get("max_results", 5), 20)
            return await self._find_nearby_places(query, max_results)
        
        elif tool_name == "control_thermostat":
            action = arguments.get("action", "").lower()
            temp_arg = arguments.get("temperature")
            
            if action not in ["raise", "lower", "set", "check"]:
                return {"error": "Invalid action. Use 'raise', 'lower', 'set', or 'check'"}
            
            try:
                # Get current thermostat state
                thermostat = self.hass.states.get(self.thermostat_entity)
                if not thermostat:
                    return {"error": "Thermostat not found"}
                
                current_target = thermostat.attributes.get("temperature")
                current_temp = thermostat.attributes.get("current_temperature")
                hvac_mode = thermostat.attributes.get("hvac_mode", thermostat.state)
                
                if current_target is None:
                    current_target = 72  # Fallback default
                
                # Handle check action - just return status
                if action == "check":
                    response_text = f"The thermostat is set to {hvac_mode} with a target temperature of {self.format_temp(current_target)}. The current temperature in the home is {self.format_temp(current_temp)}."
                    return {"response_text": response_text}
                
                # Calculate new temperature
                if action == "set":
                    if temp_arg is None:
                        return {"error": "Please specify a temperature to set"}
                    new_temp = int(temp_arg)
                elif action == "raise":
                    new_temp = int(current_target + self.thermostat_temp_step)
                else:  # lower
                    new_temp = int(current_target - self.thermostat_temp_step)

                # Clamp to user-configurable range
                new_temp = max(self.thermostat_min_temp, min(self.thermostat_max_temp, new_temp))
                
                _LOGGER.info("Thermostat control: action=%s, current=%s, new=%s", action, current_target, new_temp)
                
                # Call climate.set_temperature directly
                await self.hass.services.async_call(
                    "climate",
                    "set_temperature",
                    {
                        "entity_id": self.thermostat_entity,
                        "temperature": new_temp
                    },
                    blocking=True
                )
                
                # Build deterministic response
                if action == "set":
                    response_text = f"I've set the thermostat to {self.format_temp(new_temp)}."
                elif action == "raise":
                    response_text = f"I've raised the thermostat to {self.format_temp(new_temp)}."
                else:
                    response_text = f"I've lowered the thermostat to {self.format_temp(new_temp)}."
                
                return {"response_text": response_text}
                
            except Exception as err:
                _LOGGER.error("Error controlling thermostat: %s", err, exc_info=True)
                return {"error": f"Failed to control thermostat: {str(err)}"}
        
        elif tool_name == "check_device_status":
            # Check CURRENT status of any device
            # IMPORTANT: Don't trust LLM's extracted device name - extract from original query ourselves
            device = arguments.get("device", "").strip()
            
            # Get the original user query for better device extraction
            original_query = getattr(self, '_current_user_query', '').lower()
            
            # Extract device name from original query using patterns
            extracted_device = None
            
            # Common patterns: "what's the X status", "is the X open/locked/on", "status of X", "check the X"
            # re already imported at top
            patterns = [
                r"(?:what(?:'s| is) the |status of (?:the )?|check (?:the )?|is (?:the )?)([a-z ]+?)(?:\s+(?:status|open|closed|locked|unlocked|on|off|state)|\?|$)",
                r"(?:is |are )(?:the )?([a-z ]+?)(?:\s+(?:open|closed|locked|unlocked|on|off)|\?)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, original_query)
                if match:
                    extracted_device = match.group(1).strip()
                    # Remove trailing words that aren't part of device name
                    for suffix in [' status', ' state', ' currently', ' right now']:
                        if extracted_device.endswith(suffix):
                            extracted_device = extracted_device[:-len(suffix)].strip()
                    break
            
            # Use extracted device if we got one, otherwise fall back to LLM's extraction
            if extracted_device and len(extracted_device) > len(device):
                _LOGGER.info("Device extraction: LLM said '%s', extracted '%s' from query '%s'", device, extracted_device, original_query)
                device = extracted_device
            
            if not device:
                return {"error": "No device specified. Please specify a device name like 'front door', 'garage', 'kitchen light', etc."}
            
            # Search for entity using device aliases + HA aliases + friendly names
            entity_id, friendly_name = find_entity_by_name(self.hass, device, self.device_aliases)
            
            if not entity_id:
                return {"error": f"Could not find a device matching '{device}'. Try using the exact name as shown in Home Assistant."}
            
            # Get current state
            state = self.hass.states.get(entity_id)
            if not state:
                return {"error": f"Entity '{entity_id}' not found"}
            
            domain = entity_id.split(".")[0]
            current_state = state.state
            
            # Handle climate domain specially (thermostat)
            if domain == "climate":
                target_temp = state.attributes.get("temperature")
                current_temp = state.attributes.get("current_temperature")
                hvac_mode = state.attributes.get("hvac_mode", current_state)

                status_parts = []
                if target_temp:
                    status_parts.append(f"set to {self.format_temp(target_temp)}")
                if current_temp:
                    status_parts.append(f"currently {self.format_temp(current_temp)}")
                if hvac_mode:
                    status_parts.append(f"mode: {hvac_mode}")
                
                status = ", ".join(status_parts) if status_parts else current_state.upper()
                
                _LOGGER.info("Device status check: %s -> %s (%s) status=%s", device, friendly_name, entity_id, status)
                
                return {
                    "device": friendly_name,
                    "status": status,
                    "target_temperature": target_temp,
                    "current_temperature": current_temp,
                    "hvac_mode": hvac_mode,
                    "entity_id": entity_id
                }
            
            # Handle sensors with units
            if domain == "sensor":
                unit = state.attributes.get("unit_of_measurement", "")
                if unit:
                    status = f"{current_state} {unit}"
                else:
                    status = current_state
                
                _LOGGER.info("Device status check: %s -> %s (%s) status=%s", device, friendly_name, entity_id, status)
                
                return {
                    "device": friendly_name,
                    "status": status,
                    "entity_id": entity_id
                }
            
            # Use helper for other domains
            status = format_human_readable_state(entity_id, current_state)
            
            _LOGGER.info("Device status check: %s -> %s (%s) domain=%s raw_state=%s status=%s", device, friendly_name, entity_id, domain, current_state, status)
            
            return {
                "device": friendly_name,
                "status": status,
                "entity_id": entity_id
            }
        
        elif tool_name == "get_device_history":
            # Get HISTORICAL state changes from HA Recorder
            device = arguments.get("device", "").strip()
            days_back = min(arguments.get("days_back", 1), 10)  # Default 1 day, max 10
            specific_date = arguments.get("date", "")  # Optional YYYY-MM-DD
            
            # Extract device name from original query (don't trust LLM's extraction)
            original_query = getattr(self, '_current_user_query', '').lower()
            # re already imported at top
            patterns = [
                r"(?:history (?:of |for )?(?:the )?|when (?:was |did )(?:the )?|how many times (?:was |did )(?:the )?)([a-z ]+?)(?:\s+(?:open|closed|locked|unlocked|today|yesterday|last|this)|\?|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, original_query)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) > len(device):
                        _LOGGER.info("History device extraction: LLM said '%s', extracted '%s'", device, extracted)
                        device = extracted
                    break
            
            if not device:
                return {"error": "No device specified. Please specify a device name like 'front door', 'garage', 'mailbox', etc."}
            
            # Search for entity using device aliases + HA aliases + friendly names
            entity_id, friendly_name = find_entity_by_name(self.hass, device, self.device_aliases)
            
            if not entity_id:
                return {"error": f"Could not find a device matching '{device}'. Try using the exact name as shown in Home Assistant."}
            
            try:
                # Get current state first
                current_state = self.hass.states.get(entity_id)
                if not current_state:
                    return {"error": f"Entity '{entity_id}' not found"}
                
                friendly_name = get_friendly_name(entity_id, current_state)
                
                # Determine time range using HA configured timezone
                now = datetime.now(dt_util.get_time_zone(self.hass.config.time_zone))

                if specific_date:
                    # Query specific date
                    try:
                        target_date = datetime.strptime(specific_date, "%Y-%m-%d")
                        tz = dt_util.get_time_zone(self.hass.config.time_zone)
                        start_time = target_date.replace(hour=0, minute=0, second=0, tzinfo=tz)
                        end_time = target_date.replace(hour=23, minute=59, second=59, tzinfo=tz)
                        period_desc = target_date.strftime("%B %d, %Y")
                    except ValueError:
                        return {"error": f"Invalid date format: {specific_date}. Use YYYY-MM-DD"}
                else:
                    # Query last N days
                    end_time = now
                    start_time = now - timedelta(days=days_back)
                    if days_back == 1:
                        period_desc = "today"
                    elif days_back <= 7:
                        period_desc = f"last {days_back} days"
                    else:
                        period_desc = f"last {days_back} days"
                
                # Get history from recorder
                from homeassistant.components.recorder import get_instance
                from homeassistant.components.recorder.history import get_significant_states
                
                _LOGGER.info("Fetching history for %s from %s to %s", entity_id, start_time, end_time)
                
                history_data = await get_instance(self.hass).async_add_executor_job(
                    get_significant_states,
                    self.hass,
                    start_time.astimezone(),
                    end_time.astimezone(),
                    [entity_id],
                )
                
                # Process history
                state_changes = []
                last_on = None
                last_off = None
                on_count = 0
                off_count = 0
                
                domain = entity_id.split(".")[0]
                
                # Determine "on" and "off" states based on domain
                if domain == "lock":
                    on_state = "unlocked"
                    off_state = "locked"
                    on_label = "unlocked"
                    off_label = "locked"
                elif domain == "binary_sensor":
                    on_state = "on"
                    off_state = "off"
                    if "door" in entity_id or "gate" in entity_id or "mailbox" in entity_id:
                        on_label = "opened"
                        off_label = "closed"
                    else:
                        on_label = "detected"
                        off_label = "clear"
                elif domain in ("light", "switch", "fan"):
                    on_state = "on"
                    off_state = "off"
                    on_label = "turned on"
                    off_label = "turned off"
                else:
                    on_state = "on"
                    off_state = "off"
                    on_label = "on"
                    off_label = "off"
                
                if entity_id in history_data:
                    for state in history_data[entity_id]:
                        if state.state in ("unavailable", "unknown"):
                            continue
                        
                        try:
                            state_time = state.last_changed.astimezone(dt_util.get_time_zone(self.hass.config.time_zone))
                            time_str = state_time.strftime("%B %d at %I:%M %p")
                            
                            if state.state == on_state:
                                on_count += 1
                                last_on = time_str
                                state_changes.append({
                                    "action": on_label,
                                    "time": time_str
                                })
                            elif state.state == off_state:
                                off_count += 1
                                last_off = time_str
                                state_changes.append({
                                    "action": off_label,
                                    "time": time_str
                                })
                        except Exception as parse_err:
                            _LOGGER.warning("Error parsing state time: %s", parse_err)
                
                # Build result
                result = {
                    "device": friendly_name,
                    "entity_id": entity_id,
                    "period": period_desc,
                    "total_changes": len(state_changes),
                }
                
                # Add last on/off times
                if last_on:
                    result[f"last_{on_label.replace(' ', '_')}"] = last_on
                if last_off:
                    result[f"last_{off_label.replace(' ', '_')}"] = last_off
                
                # Add counts
                result[f"{on_label.replace(' ', '_')}_count"] = on_count
                result[f"{off_label.replace(' ', '_')}_count"] = off_count
                
                # Add recent activity (last 10 changes)
                if state_changes:
                    result["recent_activity"] = state_changes[-10:][::-1]  # Most recent first
                else:
                    result["message"] = f"No state changes found for {period_desc}"
                
                _LOGGER.info("Device history result: %s", result)
                return result
                
            except ImportError as ie:
                _LOGGER.error("Recorder import error: %s", ie)
                return {"error": "History component not available"}
            except Exception as err:
                _LOGGER.error("Error getting device history: %s", err, exc_info=True)
                return {"error": f"Failed to get device history: {str(err)}"}
        
        # =========================================================================
        # CAMERA CHECK HANDLER (via ha_video_vision integration)
        # =========================================================================
        elif tool_name == "check_camera":
            # Unified camera check - supports both quick and detailed modes
            location = arguments.get("location", "").lower().strip()
            query = arguments.get("query", "")
            detail_level = arguments.get("detail_level", "")

            # If query provided, default to detailed mode
            if query and not detail_level:
                detail_level = "detailed"
            elif not detail_level:
                detail_level = "quick"

            if not location:
                return {"error": "No camera location specified"}

            # Get friendly name for display
            friendly_name = CAMERA_FRIENDLY_NAMES.get(location, location.replace("_", " ").title())

            try:
                # Build service call data with duration based on detail level
                service_data = {
                    "camera": location,
                    "duration": 3 if detail_level == "detailed" else 2,
                }
                if query:
                    service_data["user_query"] = query

                # Call ha_video_vision integration service
                result = await self.hass.services.async_call(
                    "ha_video_vision",
                    "analyze_camera",
                    service_data,
                    blocking=True,
                    return_response=True,
                )

                if not result or not result.get("success"):
                    error_msg = result.get('error', 'Unknown error') if result else 'Service unavailable'
                    return {
                        "location": friendly_name,
                        "status": "unavailable",
                        "error": f"Could not access {friendly_name} camera: {error_msg}"
                    }

                identified = result.get("identified_people", [])
                analysis = result.get("description", "Unable to analyze camera feed")
                person_detected = result.get("person_detected", False) or bool(identified)

                if detail_level == "detailed":
                    # Full detailed response
                    response = {
                        "location": friendly_name,
                        "status": "checked",
                        "person_detected": person_detected,
                        "description": analysis
                    }
                    if identified:
                        people_str = ", ".join([f"{p['name']} ({p['confidence']}%)" for p in identified])
                        response["identified"] = people_str
                    return response
                else:
                    # Quick presence check response
                    brief = analysis.split('.')[0] + '.' if analysis else "No activity."

                    if identified:
                        names = ", ".join([p['name'] for p in identified])
                        return {
                            "location": friendly_name,
                            "anyone_there": True,
                            "who": names,
                            "brief": brief
                        }
                    elif person_detected:
                        return {
                            "location": friendly_name,
                            "anyone_there": True,
                            "who": "Unknown person",
                            "brief": brief
                        }
                    else:
                        return {
                            "location": friendly_name,
                            "anyone_there": False,
                            "brief": brief
                        }

            except Exception as err:
                _LOGGER.error("Error checking camera %s: %s", location, err, exc_info=True)
                return {
                    "location": friendly_name,
                    "status": "error",
                    "error": f"Failed to check {friendly_name} camera: {str(err)}"
                }

        elif tool_name == "get_restaurant_recommendations":
            query = arguments.get("query", "")
            max_results = min(arguments.get("max_results", 5), 10)
            
            if not query:
                return {"error": "No restaurant/food type specified"}
            
            # API key from config - required
            api_key = self.yelp_api_key
            if not api_key:
                return {"error": "Yelp API key not configured. Add it in Settings → PolyVoice → API Keys."}
            
            # Use custom location if set, otherwise fall back to defaults
            latitude = self.custom_latitude or self.hass.config.latitude
            longitude = self.custom_longitude or self.hass.config.longitude
            
            try:
                from urllib.parse import quote
                
                encoded_query = quote(query)
                url = f"https://api.yelp.com/v3/businesses/search?term={encoded_query}&latitude={latitude}&longitude={longitude}&limit={max_results}&sort_by=rating"
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json"
                }
                
                _LOGGER.info("Searching Yelp for: %s", query)
                
                self._track_api_call("restaurants")
                
                async with asyncio.timeout(API_TIMEOUT):
                    async with self._session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            businesses = data.get("businesses", [])
                            
                            if not businesses:
                                return {"message": f"No restaurants found for '{query}'"}
                            
                            results = []
                            for biz in businesses:
                                result = {
                                    "name": biz.get("name"),
                                    "rating": biz.get("rating"),
                                    "review_count": biz.get("review_count"),
                                    "price": biz.get("price", "N/A"),
                                    "address": ", ".join(biz.get("location", {}).get("display_address", [])),
                                }
                                
                                # Add categories
                                categories = [cat.get("title") for cat in biz.get("categories", [])]
                                if categories:
                                    result["cuisine"] = ", ".join(categories[:2])
                                
                                # Add distance
                                distance_meters = biz.get("distance", 0)
                                distance_miles = distance_meters / 1609.34
                                result["distance"] = f"{distance_miles:.1f} miles"
                                
                                # Add if open now
                                if not biz.get("is_closed", True):
                                    result["status"] = "Open now"
                                
                                results.append(result)
                            
                            _LOGGER.info("Yelp found %d restaurants", len(results))
                            return {
                                "query": query,
                                "count": len(results),
                                "restaurants": results
                            }
                        else:
                            _LOGGER.error("Yelp API error: %s", response.status)
                            return {"error": f"Yelp API returned status {response.status}"}
                            
            except Exception as err:
                _LOGGER.error("Error searching Yelp: %s", err, exc_info=True)
                return {"error": f"Failed to search restaurants: {str(err)}"}

        elif tool_name == "control_music":
            action = arguments.get("action", "").lower()
            query = arguments.get("query", "")
            media_type = arguments.get("media_type", "artist")
            room = arguments.get("room", "").lower() if arguments.get("room") else ""
            shuffle = arguments.get("shuffle", False)

            _LOGGER.info("=== MUSIC CONTROL START ===")
            _LOGGER.info("room_player_mapping config: %s", self.room_player_mapping)

            players = self.room_player_mapping  # {room: entity_id}
            all_players = list(players.values())

            _LOGGER.info("all_players list: %s", all_players)

            if not all_players:
                _LOGGER.error("No players configured! room_player_mapping is empty")
                return {"error": "No music players configured. Go to PolyVoice → Entity Configuration → Room to Player Mapping and add entries like: living room: media_player.your_player"}

            # Helper: find player in a specific state
            def find_player_by_state(target_state):
                """Scan all players and return the one in the target state."""
                for pid in all_players:
                    state = self.hass.states.get(pid)
                    if state:
                        _LOGGER.info("  %s → %s", pid, state.state)
                        if state.state == target_state:
                            return pid
                return None

            # Helper: get room name from entity_id
            def get_room_name(entity_id):
                for rname, pid in players.items():
                    if pid == entity_id:
                        return rname
                return "unknown"

            try:
                _LOGGER.info("=== MUSIC: %s ===", action.upper())

                # Determine target player(s) for play action
                if room in players:
                    target_players = [players[room]]
                elif room:
                    # Fuzzy match room name
                    for rname, pid in players.items():
                        if room in rname or rname in room:
                            target_players = [pid]
                            break
                    else:
                        target_players = []
                else:
                    target_players = []

                if action == "play":
                    if not query:
                        return {"error": "No music query specified"}
                    if not target_players:
                        return {"error": f"Unknown room: {room}. Available: {', '.join(players.keys())}"}

                    for player in target_players:
                        await self.hass.services.async_call(
                            "music_assistant", "play_media",
                            {"media_id": query, "media_type": media_type, "enqueue": "replace"},
                            target={"entity_id": player},
                            blocking=True
                        )
                        if shuffle or media_type == "genre":
                            await self.hass.services.async_call(
                                "media_player", "shuffle_set",
                                {"entity_id": player, "shuffle": True},
                                blocking=True
                            )

                    # Track last active
                    if self.last_active_speaker and target_players:
                        await self.hass.services.async_call(
                            "input_text", "set_value",
                            {"entity_id": self.last_active_speaker, "value": target_players[0]}
                        )

                    return {"status": "playing", "message": f"Playing {query} in the {room}"}

                elif action == "pause":
                    # Find the player that's currently PLAYING and pause it
                    _LOGGER.info("Looking for player in 'playing' state...")
                    playing = find_player_by_state("playing")
                    if playing:
                        await self.hass.services.async_call("media_player", "media_pause", {"entity_id": playing})
                        return {"status": "paused", "message": f"Paused in {get_room_name(playing)}"}
                    return {"error": "No music is currently playing"}

                elif action == "resume":
                    # Find the player that's currently PAUSED and resume it
                    _LOGGER.info("Looking for player in 'paused' state...")
                    paused = find_player_by_state("paused")
                    if paused:
                        await self.hass.services.async_call("media_player", "media_play", {"entity_id": paused})
                        return {"status": "resumed", "message": f"Resumed in {get_room_name(paused)}"}
                    return {"error": "No paused music to resume"}

                elif action == "stop":
                    # Find the player that's PLAYING or PAUSED and stop it
                    _LOGGER.info("Looking for player in 'playing' or 'paused' state...")
                    playing = find_player_by_state("playing")
                    if playing:
                        await self.hass.services.async_call("media_player", "media_stop", {"entity_id": playing})
                        return {"status": "stopped", "message": f"Stopped in {get_room_name(playing)}"}
                    paused = find_player_by_state("paused")
                    if paused:
                        await self.hass.services.async_call("media_player", "media_stop", {"entity_id": paused})
                        return {"status": "stopped", "message": f"Stopped in {get_room_name(paused)}"}
                    return {"message": "No music is playing"}

                elif action == "skip_next":
                    # Find the player that's PLAYING and skip
                    _LOGGER.info("Looking for player in 'playing' state...")
                    playing = find_player_by_state("playing")
                    if playing:
                        await self.hass.services.async_call("media_player", "media_next_track", {"entity_id": playing})
                        return {"status": "skipped", "message": "Skipped to next track"}
                    return {"error": "No music is playing to skip"}

                elif action == "skip_previous":
                    # Find the player that's PLAYING and go back
                    _LOGGER.info("Looking for player in 'playing' state...")
                    playing = find_player_by_state("playing")
                    if playing:
                        await self.hass.services.async_call("media_player", "media_previous_track", {"entity_id": playing})
                        return {"status": "skipped", "message": "Previous track"}
                    return {"error": "No music is playing"}

                elif action == "what_playing":
                    # Find player that's playing
                    _LOGGER.info("Looking for player in 'playing' state...")
                    playing = find_player_by_state("playing")
                    if playing:
                        state = self.hass.states.get(playing)
                        attrs = state.attributes
                        return {
                            "title": attrs.get("media_title", "Unknown"),
                            "artist": attrs.get("media_artist", "Unknown"),
                            "album": attrs.get("media_album_name", ""),
                            "room": get_room_name(playing)
                        }
                    return {"message": "No music currently playing"}

                elif action == "transfer":
                    # Find player that's playing and transfer to target room
                    _LOGGER.info("Looking for player in 'playing' state...")
                    playing = find_player_by_state("playing")
                    if not playing:
                        return {"error": "No music playing to transfer"}
                    if not target_players:
                        return {"error": f"No target room specified. Available: {', '.join(players.keys())}"}

                    target = target_players[0]
                    _LOGGER.info("Transferring from %s to %s", playing, target)

                    # Transfer queue from source to target
                    await self.hass.services.async_call(
                        "music_assistant", "transfer_queue",
                        {"source_player": playing, "auto_play": True},
                        target={"entity_id": target},
                        blocking=True
                    )
                    # Update last active
                    if self.last_active_speaker:
                        await self.hass.services.async_call(
                            "input_text", "set_value",
                            {"entity_id": self.last_active_speaker, "value": target}
                        )
                    return {"status": "transferred", "message": f"Music transferred to {get_room_name(target)}"}

                else:
                    return {"error": f"Unknown action: {action}"}

            except Exception as err:
                _LOGGER.error("Music control error: %s", err, exc_info=True)
                return {"error": f"Music control failed: {str(err)}"}

        # Default: try to call as a script
        if self.hass.services.has_service("script", tool_name):
            _LOGGER.info("Calling script: %s with args: %s", tool_name, arguments)
            
            response = await self.hass.services.async_call(
                "script", tool_name, arguments, blocking=True, return_response=True
            )
            
            _LOGGER.info("Script %s response: %s", tool_name, response)
            
            if response is not None:
                script_entity = f"script.{tool_name}"
                if isinstance(response, dict):
                    if script_entity in response:
                        return response[script_entity]
                    if response:
                        return response
            
            return {"status": "success", "script": tool_name}
        
        _LOGGER.warning("Custom function '%s' called but no handler implemented. Arguments: %s", tool_name, arguments)
        return {"success": True, "message": f"Custom function {tool_name} called", "arguments": arguments}

    async def _find_nearby_places(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Find nearby places using Google Places API."""
        # API key from config - required
        api_key = self.google_places_api_key
        if not api_key:
            return {"error": "Google Places API key not configured. Add it in Settings → PolyVoice → API Keys."}
        
        # Use custom location if set, otherwise fall back to HA location
        latitude = self.custom_latitude or self.hass.config.latitude
        longitude = self.custom_longitude or self.hass.config.longitude
        
        try:
            url = "https://places.googleapis.com/v1/places:searchText"
            
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating,places.currentOpeningHours"
            }
            
            body = {
                "textQuery": query,
                "locationBias": {
                    "circle": {
                        "center": {
                            "latitude": latitude,
                            "longitude": longitude
                        },
                        "radius": 10000.0
                    }
                },
                "maxResultCount": max_results,
                "rankPreference": "DISTANCE"
            }
            
            self._track_api_call("places")
            
            async with asyncio.timeout(API_TIMEOUT):
                async with self._session.post(url, json=body, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error("Google Places HTTP error: %s - %s", response.status, error_text)
                        return {"error": f"Google Places API error: {response.status} - {error_text}"}
                    
                    data = await response.json()
                    _LOGGER.info("Google Places API response: %s", data)
                
                places = data.get("places", [])
                
                if not places:
                    return {"results": f"No results found for '{query}' near you."}
                
                results = []
                
                for idx, place in enumerate(places[:max_results], 1):
                    place_name = place.get("displayName", {}).get("text", "Unknown")
                    address = place.get("formattedAddress", "Address not available")
                    rating = place.get("rating", "No rating")
                    
                    is_open = "Unknown hours"
                    if "currentOpeningHours" in place:
                        opening_hours = place.get("currentOpeningHours", {})
                        is_open = "Open now" if opening_hours.get("openNow") else "Closed"
                    
                    place_lat = place.get("location", {}).get("latitude")
                    place_lng = place.get("location", {}).get("longitude")
                    
                    distance_str = ""
                    if place_lat and place_lng:
                        miles = calculate_distance_miles(latitude, longitude, place_lat, place_lng)
                        distance_str = f" ({miles:.1f} miles away)"
                    
                    result_text = f"{idx}. {place_name} - {address}{distance_str}. Rating: {rating}/5. {is_open}."
                    results.append(result_text)
                
                response_text = f"Found {len(results)} places for '{query}' near you (sorted by distance):\n\n" + "\n".join(results)
                
                return {"results": response_text}
                
        except Exception as err:
            _LOGGER.error("Error calling Google Places API: %s", err, exc_info=True)
            return {"error": f"Failed to search for places: {str(err)}"}

    async def _check_single_camera(self, camera_key: str, user_query: str = "") -> dict[str, Any]:
        """Check a single camera and return the analysis result.

        This method is used by the facial_recognition service for automation triggers.

        Args:
            camera_key: The camera key (e.g., "porch", "driveway")
            user_query: Optional query about what to look for

        Returns:
            Dict with analysis results including identified_people and analysis text
        """
        friendly_name = self.camera_friendly_names.get(
            camera_key,
            camera_key.replace("_", " ").title()
        )

        try:
            service_data = {"camera": camera_key, "duration": 3}
            if user_query:
                service_data["user_query"] = user_query

            result = await self.hass.services.async_call(
                "ha_video_vision",
                "analyze_camera",
                service_data,
                blocking=True,
                return_response=True,
            )

            if not result or not result.get("success"):
                error_msg = result.get('error', 'Unknown error') if result else 'Service unavailable'
                return {
                    "location": friendly_name,
                    "status": "unavailable",
                    "error": f"Could not access {friendly_name} camera: {error_msg}",
                    "identified_people": [],
                    "analysis": ""
                }

            identified = result.get("identified_people", [])
            analysis = result.get("description", "Unable to analyze camera feed")

            return {
                "location": friendly_name,
                "status": "checked",
                "person_detected": len(identified) > 0,
                "identified_people": identified,
                "analysis": analysis
            }

        except Exception as err:
            _LOGGER.error("Error checking camera %s: %s", camera_key, err, exc_info=True)
            return {
                "location": friendly_name,
                "status": "error",
                "error": f"Failed to check {friendly_name} camera: {str(err)}",
                "identified_people": [],
                "analysis": ""
            }
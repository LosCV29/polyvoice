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
    CONF_ENABLE_MUSIC,
    CONF_ENABLE_CAMERAS,
    CONF_ENABLE_SPORTS,
    CONF_ENABLE_NEWS,
    CONF_ENABLE_PLACES,
    CONF_ENABLE_RESTAURANTS,
    CONF_ENABLE_THERMOSTAT,
    CONF_ENABLE_DEVICE_STATUS,
    CONF_ENABLE_WIKIPEDIA,
    # Entity config
    CONF_THERMOSTAT_ENTITY,
    CONF_CALENDAR_ENTITIES,
    CONF_MUSIC_PLAYERS,
    CONF_DEFAULT_MUSIC_PLAYER,
    CONF_ROOM_PLAYER_MAPPING,
    CONF_LAST_ACTIVE_SPEAKER,
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
    DEFAULT_ENABLE_MUSIC,
    DEFAULT_ENABLE_CAMERAS,
    DEFAULT_ENABLE_SPORTS,
    DEFAULT_ENABLE_NEWS,
    DEFAULT_ENABLE_PLACES,
    DEFAULT_ENABLE_RESTAURANTS,
    DEFAULT_ENABLE_THERMOSTAT,
    DEFAULT_ENABLE_DEVICE_STATUS,
    DEFAULT_ENABLE_WIKIPEDIA,
    CAMERA_FRIENDLY_NAMES,
    ALL_NATIVE_INTENTS,
    # Thermostat settings
    CONF_THERMOSTAT_MIN_TEMP,
    CONF_THERMOSTAT_MAX_TEMP,
    CONF_THERMOSTAT_TEMP_STEP,
    DEFAULT_THERMOSTAT_MIN_TEMP,
    DEFAULT_THERMOSTAT_MAX_TEMP,
    DEFAULT_THERMOSTAT_TEMP_STEP,
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
        
        self.excluded_intents = set(config.get(CONF_EXCLUDED_INTENTS, DEFAULT_EXCLUDED_INTENTS))

        custom_excluded = config.get(CONF_CUSTOM_EXCLUDED_INTENTS, DEFAULT_CUSTOM_EXCLUDED_INTENTS)
        if custom_excluded:
            custom_list = [i.strip() for i in custom_excluded.split(",") if i.strip()]
            self.excluded_intents.update(custom_list)

        _LOGGER.warning("=== EXCLUDED INTENTS === %s", self.excluded_intents)

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
        self.enable_music = config.get(CONF_ENABLE_MUSIC, DEFAULT_ENABLE_MUSIC)
        self.enable_cameras = config.get(CONF_ENABLE_CAMERAS, DEFAULT_ENABLE_CAMERAS)
        self.enable_sports = config.get(CONF_ENABLE_SPORTS, DEFAULT_ENABLE_SPORTS)
        self.enable_news = config.get(CONF_ENABLE_NEWS, DEFAULT_ENABLE_NEWS)
        self.enable_places = config.get(CONF_ENABLE_PLACES, DEFAULT_ENABLE_PLACES)
        self.enable_restaurants = config.get(CONF_ENABLE_RESTAURANTS, DEFAULT_ENABLE_RESTAURANTS)
        self.enable_thermostat = config.get(CONF_ENABLE_THERMOSTAT, DEFAULT_ENABLE_THERMOSTAT)
        self.enable_device_status = config.get(CONF_ENABLE_DEVICE_STATUS, DEFAULT_ENABLE_DEVICE_STATUS)
        self.enable_wikipedia = config.get(CONF_ENABLE_WIKIPEDIA, DEFAULT_ENABLE_WIKIPEDIA)
        
        # Entity configuration from UI
        self.thermostat_entity = config.get(CONF_THERMOSTAT_ENTITY, "")
        self.calendar_entities = parse_list_config(config.get(CONF_CALENDAR_ENTITIES, ""))
        self.camera_entities = parse_list_config(config.get(CONF_CAMERA_ENTITIES, ""))
        self.default_music_player = config.get(CONF_DEFAULT_MUSIC_PLAYER, "")

        # Music players - now a simple list of entity_ids
        music_players_config = config.get(CONF_MUSIC_PLAYERS, [])
        if isinstance(music_players_config, list):
            self.music_players = music_players_config
        elif isinstance(music_players_config, str) and music_players_config:
            # Migrate old format: parse "room:entity_id" lines to just entity_ids
            self.music_players = []
            for line in music_players_config.split("\n"):
                if ":" in line:
                    self.music_players.append(line.split(":", 1)[1].strip())
                elif line.strip().startswith("media_player."):
                    self.music_players.append(line.strip())
        else:
            self.music_players = []

        # Room to player explicit mapping (e.g., "living room: media_player.chromecast")
        self.room_player_mapping = parse_entity_config(config.get(CONF_ROOM_PLAYER_MAPPING, ""))
        _LOGGER.debug("Room player mapping: %s", self.room_player_mapping)

        self.last_active_speaker = config.get(CONF_LAST_ACTIVE_SPEAKER, "")
        _LOGGER.warning("=== CONFIG LOADED === last_active_speaker='%s' (key exists: %s)",
                        self.last_active_speaker, CONF_LAST_ACTIVE_SPEAKER in config)
        self.device_aliases = parse_entity_config(config.get(CONF_DEVICE_ALIASES, ""))
        self.notification_service = config.get(CONF_NOTIFICATION_SERVICE, "")

        # Thermostat settings (user-configurable limits and step)
        self.thermostat_min_temp = int(config.get(CONF_THERMOSTAT_MIN_TEMP, DEFAULT_THERMOSTAT_MIN_TEMP))
        self.thermostat_max_temp = int(config.get(CONF_THERMOSTAT_MAX_TEMP, DEFAULT_THERMOSTAT_MAX_TEMP))
        self.thermostat_temp_step = int(config.get(CONF_THERMOSTAT_TEMP_STEP, DEFAULT_THERMOSTAT_TEMP_STEP))

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
            tools.append({
                "type": "function",
                "function": {
                    "name": "control_thermostat",
                    "description": "Control or check the thermostat/AC/air. Use for: 'raise/lower the AC' (±2°F), 'set AC to 72', 'what is the AC set to', 'what's the temp inside'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["raise", "lower", "set", "check"], "description": "'raise' = +2°F, 'lower' = -2°F, 'set' = specific temp, 'check' = get current status"},
                            "temperature": {"type": "number", "description": "Target temperature (only for 'set' action)"}
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
                            "team_name": {"type": "string", "description": "Team name (e.g., 'Florida Panthers', 'Miami Heat', 'Manchester City', 'Inter Miami', 'Miami Dolphins')"},
                            "query_type": {"type": "string", "enum": ["last_game", "next_game", "standings", "both"], "description": "What info to get: 'last_game' for recent result, 'next_game' for upcoming, 'standings' for league position, 'both' for last and next games (default)"}
                        },
                        "required": ["team_name"]
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

        # ===== MUSIC CONTROL (if enabled and players configured) =====
        if self.enable_music and self.music_players:
            tools.append({
                "type": "function",
                "function": {
                    "name": "control_music",
                    "description": "REQUIRED for ALL music commands - you MUST call this tool whenever user mentions music. Trigger words: 'play', 'put on', 'I want to hear', 'listen to', 'shuffle', 'skip', 'next', 'previous', 'pause', 'resume', 'stop'. Example: 'play jazz' -> call control_music(action='play', query='jazz'). NEVER respond to music requests without calling this tool first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["play", "transfer", "skip", "previous", "pause", "resume", "stop"], "description": "The action to perform"},
                            "query": {"type": "string", "description": "What to play: artist, song, album, playlist, or genre (e.g., 'jazz', 'Beatles', 'workout playlist')"},
                            "room": {"type": "string", "description": "Target room for playback or transfer (e.g., 'kitchen', 'bedroom', 'office')"},
                            "shuffle": {"type": "boolean", "description": "Shuffle the playlist (default: false)"}
                        },
                        "required": ["action"]
                    }
                }
            })

        # ===== CAMERA CHECKS (if enabled) =====
        # Uses ha_video_vision integration for AI analysis
        if self.enable_cameras:
            # Detailed camera check - full description + person identification
            tools.append({
                "type": "function",
                "function": {
                    "name": "check_camera",
                    "description": "Check a camera with detailed AI vision analysis + facial recognition. Returns full scene description and identifies people. Use for: 'check the [location] camera', 'what's happening in [location]', 'show me the [location]', 'who's at the [location]'. Works with any camera location: garage, kitchen, nursery, driveway, porch, backyard, living room, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The camera location to check. Examples: 'garage', 'kitchen', 'nursery', 'living room', 'driveway', 'porch', 'backyard', 'front door', 'doorbell'"
                            },
                            "query": {
                                "type": "string",
                                "description": "Optional specific question about what to look for (e.g., 'is the baby sleeping', 'is there a package', 'is the car there')"
                            }
                        },
                        "required": ["location"]
                    }
                }
            })

            # Quick presence check - fast response, just person detection
            tools.append({
                "type": "function",
                "function": {
                    "name": "quick_camera_check",
                    "description": "FAST camera check - quickly confirms if anyone is present + one sentence description. Use for: 'is there anyone in [location]', 'is someone at the [location]', 'anyone in the [location]?'. Returns quick yes/no with brief description.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The camera location to check. Examples: 'garage', 'kitchen', 'driveway', 'porch', 'backyard', 'front door'"
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
        
        return tools

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        conversation_id = user_input.conversation_id or ulid.ulid_now()
        
        # Store original query for tools to access (for reliable device name extraction)
        self._current_user_query = user_input.text
        
        _LOGGER.info("=== Incoming request: '%s' (conv_id: %s) ===", user_input.text, conversation_id[:8])
        
        if self.use_native_intents:
            native_result = await self._try_native_intent(user_input, conversation_id)
            if native_result is not None:
                _LOGGER.info("Handled by native intent (not LLM): %s", user_input.text)
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

            # FALLBACK: Parse tool calls from text (for models like Qwen that output as text)
            # Skip if we already called control_music this session (any variant)
            already_called_music = any(k.startswith("control_music:") for k in called_tools)
            if not valid_tool_calls and accumulated_content and not already_called_music:
                import re
                content_lower = accumulated_content.lower()
                parsed_args = None

                # Pattern 1: control_music(...) function call in text
                tool_match = re.search(r'control_music\s*\(\s*([^)]+)\s*\)', accumulated_content)
                if tool_match:
                    _LOGGER.warning("Detected text-based tool call, parsing: %s", tool_match.group(0))
                    args_str = tool_match.group(1)
                    parsed_args = {}
                    for match in re.finditer(r'(\w+)\s*=\s*["\']?([^"\'",\)]+)["\']?', args_str):
                        key, val = match.group(1), match.group(2).strip()
                        if val.lower() == 'true':
                            parsed_args[key] = True
                        elif val.lower() == 'false':
                            parsed_args[key] = False
                        else:
                            parsed_args[key] = val

                # Pattern 2: "I'll play X" or "Playing X" or "Now playing X" (LLM responded conversationally)
                if not parsed_args:
                    play_patterns = [
                        r"(?:i'll|i will|let me|going to|now)\s+play(?:ing)?\s+(.+?)(?:\s+(?:for you|in the|on the|now)|\.|$)",
                        r"playing\s+(.+?)(?:\s+(?:for you|in the|on the|now)|\.|$)",
                        r"(?:putting on|starting)\s+(.+?)(?:\s+(?:for you|in the|on the|now)|\.|$)",
                    ]
                    for pattern in play_patterns:
                        match = re.search(pattern, content_lower)
                        if match:
                            query = match.group(1).strip()
                            if query and len(query) > 1:
                                _LOGGER.warning("=== FALLBACK DETECTED === LLM said '%s', extracting music request: '%s'", accumulated_content[:100], query)
                                parsed_args = {"action": "play", "query": query}
                                break

                if parsed_args:
                    # Check if we already called control_music (prevent duplicates)
                    tool_key = f"control_music:{json.dumps(parsed_args, sort_keys=True)}"
                    if tool_key not in called_tools:
                        called_tools.add(tool_key)
                        _LOGGER.warning("Parsed text tool call: control_music(%s)", parsed_args)
                        valid_tool_calls = [{
                            "id": "text_parsed_1",
                            "type": "function",
                            "function": {
                                "name": "control_music",
                                "arguments": json.dumps(parsed_args)
                            }
                        }]
                        accumulated_content = ""  # Clear the text since we're executing the tool
                    else:
                        _LOGGER.debug("Skipping duplicate fallback tool call: control_music")

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

                    # For control_music, return immediately with the message (don't loop back to LLM)
                    if tool_call["function"]["name"] == "control_music" and isinstance(result, dict):
                        _LOGGER.warning("=== CONTROL_MUSIC EARLY RETURN === result=%s", result)
                        if "message" in result:
                            return result["message"]
                        elif "error" in result:
                            return result["error"]
                        # Fallback: return status if no message/error
                        return f"Music control: {result.get('status', 'done')}"

                continue
            
            if accumulated_content:
                return full_response
            
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

                # Search for team in major US leagues directly (search API is deprecated)
                leagues_to_try = [
                    ("basketball", "nba"),
                    ("football", "nfl"),
                    ("baseball", "mlb"),
                    ("hockey", "nhl"),
                ]

                team_found = False
                url = None
                full_name = team_name
                for sport, league in leagues_to_try:
                    teams_url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams"
                    async with self._session.get(teams_url, headers=headers) as teams_resp:
                        if teams_resp.status == 200:
                            teams_data = await teams_resp.json()
                            for team in teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
                                t = team.get("team", {})
                                if (team_key in t.get("displayName", "").lower() or
                                    team_key in t.get("shortDisplayName", "").lower() or
                                    team_key in t.get("nickname", "").lower() or
                                    team_key == t.get("abbreviation", "").lower()):
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
                
                # Find last completed game and next upcoming game
                now = datetime.now()
                last_game = None
                next_game = None
                
                for event in events:
                    status = event.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed", False)
                    game_date_str = event.get("date", "")
                    
                    if game_date_str:
                        try:
                            game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                            game_date_naive = game_date.replace(tzinfo=None)
                            
                            if status:  # Completed game
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
                    home_score = home_team.get("score", "0")
                    away_score = away_team.get("score", "0")
                    
                    game_date = last_game.get("date", "")[:10]
                    
                    result["last_game"] = {
                        "date": game_date,
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_score": home_score,
                        "away_score": away_score,
                        "summary": f"{away_name} {away_score} @ {home_name} {home_score}"
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
                
                # Build response text
                response_parts = []
                if "last_game" in result:
                    lg = result["last_game"]
                    response_parts.append(f"Last game: {lg['summary']} on {lg['date']}")
                if "next_game" in result:
                    ng = result["next_game"]
                    response_parts.append(f"Next game: {ng['summary']}")
                
                result["response_text"] = ". ".join(response_parts) if response_parts else f"No game info found for {full_name}"
                
                _LOGGER.info("Sports info for %s: %s", full_name, result.get("response_text", ""))
                return result
                
            except Exception as err:
                _LOGGER.error("Sports API error: %s", err, exc_info=True)
                return {"error": f"Failed to get sports info: {str(err)}"}
        
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
            }
            
            if sport not in sport_endpoints:
                return {"error": f"Unknown sport: {sport}. Supported: nba, nfl, nhl, mlb, mls, epl"}
            
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
                    response_text = f"The thermostat is set to {hvac_mode} with a target temperature of {int(current_target)}°F. The current temperature in the home is {int(current_temp)}°F."
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
                    response_text = f"I've set the thermostat to {new_temp}°F."
                elif action == "raise":
                    response_text = f"I've raised the thermostat to {new_temp}°F."
                else:
                    response_text = f"I've lowered the thermostat to {new_temp}°F."
                
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
                    status_parts.append(f"set to {target_temp}°F")
                if current_temp:
                    status_parts.append(f"currently {current_temp}°F")
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
        # CAMERA CHECK HANDLERS (via ha_video_vision integration)
        # =========================================================================
        elif tool_name == "check_camera":
            # Detailed camera check - full description + person identification
            location = arguments.get("location", "").lower().strip()
            query = arguments.get("query", "")

            if not location:
                return {"error": "No camera location specified"}

            # Get friendly name for display
            friendly_name = CAMERA_FRIENDLY_NAMES.get(location, location.replace("_", " ").title())

            try:
                # Build service call data
                service_data = {
                    "camera": location,
                    "duration": 3,
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

                # Build response with full details
                identified = result.get("identified_people", [])
                analysis = result.get("description", "Unable to analyze camera feed")
                person_detected = result.get("person_detected", False)

                response = {
                    "location": friendly_name,
                    "status": "checked",
                    "person_detected": person_detected or bool(identified),
                    "description": analysis
                }

                if identified:
                    people_str = ", ".join([f"{p['name']} ({p['confidence']}%)" for p in identified])
                    response["identified"] = people_str

                return response

            except Exception as err:
                _LOGGER.error("Error checking camera %s: %s", location, err, exc_info=True)
                return {
                    "location": friendly_name,
                    "status": "error",
                    "error": f"Failed to check {friendly_name} camera: {str(err)}"
                }

        elif tool_name == "quick_camera_check":
            # Fast presence check - just person detection + one sentence
            location = arguments.get("location", "").lower().strip()

            if not location:
                return {"error": "No camera location specified"}

            # Get friendly name for display
            friendly_name = CAMERA_FRIENDLY_NAMES.get(location, location.replace("_", " ").title())

            try:
                # Call ha_video_vision with quick mode (shorter duration)
                result = await self.hass.services.async_call(
                    "ha_video_vision",
                    "analyze_camera",
                    {"camera": location, "duration": 2},
                    blocking=True,
                    return_response=True,
                )

                if not result or not result.get("success"):
                    return {"location": friendly_name, "error": "Camera unavailable"}

                identified = result.get("identified_people", [])
                person_detected = result.get("person_detected", False) or bool(identified)
                analysis = result.get("description", "")

                # Extract just first sentence for quick response
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
                _LOGGER.error("Error quick-checking camera %s: %s", location, err)
                return {"location": friendly_name, "error": "Check failed"}

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
            # =================================================================
            # MUSIC CONTROL - Play, shuffle, and transfer via Music Assistant
            # Pause/resume/skip handled by native HA intents
            # =================================================================
            action = arguments.get("action", "").lower()
            query = arguments.get("query", "")
            room = arguments.get("room", "").lower().strip()
            shuffle = arguments.get("shuffle", False)

            _LOGGER.warning("=== CONTROL_MUSIC CALLED === action=%s, query='%s', room='%s', shuffle=%s, last_active_speaker=%s",
                           action, query, room, shuffle, self.last_active_speaker)

            # Get configured players
            all_players = self.music_players if self.music_players else []
            default_player = self.default_music_player or (all_players[0] if all_players else None)

            if not all_players:
                return {"error": "No music players configured in PolyVoice settings."}

            # Build room -> player mapping: EXPLICIT mapping takes priority
            def get_room_name(entity_id: str) -> str:
                """Get room name from entity's friendly_name."""
                state = self.hass.states.get(entity_id)
                if state:
                    fname = state.attributes.get("friendly_name", "").lower()
                    fname = fname.replace("home assistant voice", "").strip()
                    fname = fname.replace("media player", "").strip()
                    fname = fname.replace("speaker", "").strip()
                    if fname:
                        return fname.strip()
                return entity_id.replace("media_player.", "").replace("_", " ")

            # Start with explicit room mapping (user-configured, highest priority)
            room_to_player = dict(self.room_player_mapping)
            _LOGGER.info("Explicit room mapping: %s", room_to_player)

            # Add auto-generated mappings for players NOT already mapped
            mapped_players = set(room_to_player.values())
            for p in all_players:
                if p not in mapped_players:
                    auto_room = get_room_name(p)
                    if auto_room not in room_to_player:
                        room_to_player[auto_room] = p

            _LOGGER.info("control_music: action=%s, query='%s', room='%s', shuffle=%s",
                        action, query, room, shuffle)
            _LOGGER.info("Final room mapping: %s", room_to_player)

            # Find target player for the room
            def find_player_for_room(target_room: str) -> str:
                if not target_room:
                    return default_player
                target_room_lower = target_room.lower()
                # Exact match (case-insensitive)
                for rname, player in room_to_player.items():
                    if rname.lower() == target_room_lower:
                        return player
                # Partial match
                for rname, player in room_to_player.items():
                    if target_room_lower in rname.lower() or rname.lower() in target_room_lower:
                        return player
                return default_player

            # Helper to get the last active speaker from the input_text helper
            def get_last_active_player() -> str | None:
                _LOGGER.warning("get_last_active_player: checking helper=%s", self.last_active_speaker)
                if self.last_active_speaker:
                    state = self.hass.states.get(self.last_active_speaker)
                    _LOGGER.warning("get_last_active_player: state=%s", state.state if state else "None")
                    if state and state.state and state.state.startswith("media_player."):
                        return state.state
                return None

            # Helper to update the last active speaker
            async def set_last_active_player(player: str):
                _LOGGER.warning("=== SET_LAST_ACTIVE_PLAYER === player=%s, helper=%s", player, self.last_active_speaker)
                if self.last_active_speaker and player:
                    try:
                        await self.hass.services.async_call(
                            "input_text", "set_value",
                            {"entity_id": self.last_active_speaker, "value": player},
                            blocking=True
                        )
                        _LOGGER.warning("Successfully updated last active speaker to: %s", player)
                    except Exception as e:
                        _LOGGER.error("Failed to update last active speaker: %s", e)
                else:
                    _LOGGER.warning("Cannot update last active speaker: helper=%s, player=%s", self.last_active_speaker, player)

            # Helper to find currently playing player (only from configured music_players)
            def find_playing_player() -> str | None:
                for player in all_players:
                    state = self.hass.states.get(player)
                    if state and state.state == "playing":
                        return player
                return None

            try:
                # ===== PLAY ACTION =====
                if action == "play":
                    _LOGGER.warning("=== PLAY ACTION START === query='%s', room='%s', shuffle=%s", query, room, shuffle)

                    if not query:
                        return {"error": "What would you like to play?"}

                    # Priority: room mapping → last active player → default player
                    if room:
                        target_player = find_player_for_room(room)
                        _LOGGER.warning("=== PLAY TARGET (from room) === room='%s' -> player='%s'", room, target_player)
                    else:
                        target_player = get_last_active_player()
                        _LOGGER.warning("=== PLAY TARGET (from helper) === helper value='%s'", target_player)

                    if not target_player:
                        target_player = default_player
                        _LOGGER.warning("=== PLAY TARGET (fallback to default) === player='%s'", target_player)

                    if not target_player:
                        return {"error": "No music player found. Please configure music players in PolyVoice settings."}

                    _LOGGER.warning("=== PLAY EXECUTING === Playing '%s' on %s (shuffle=%s)", query, target_player, shuffle)

                    # If shuffle requested, enable shuffle FIRST
                    if shuffle:
                        await self.hass.services.async_call(
                            "media_player", "shuffle_set",
                            {"entity_id": target_player, "shuffle": True},
                            blocking=True
                        )
                        _LOGGER.info("Shuffle enabled for %s", target_player)

                    # Play via Music Assistant
                    await self.hass.services.async_call(
                        "music_assistant", "play_media",
                        {"media_id": query, "enqueue": "replace"},
                        target={"entity_id": target_player},
                        blocking=True
                    )

                    # If shuffle, skip to get fresh shuffle order
                    if shuffle:
                        await asyncio.sleep(1.5)  # Let queue build
                        await self.hass.services.async_call(
                            "media_player", "media_next_track",
                            {"entity_id": target_player},
                            blocking=True
                        )
                        _LOGGER.info("Skipped to fresh shuffle position")

                    # Update last active speaker helper
                    _LOGGER.warning("=== PLAY UPDATING HELPER === About to set helper to: %s", target_player)
                    await set_last_active_player(target_player)
                    _LOGGER.warning("=== PLAY COMPLETE === Helper updated, returning success")

                    room_name = room if room else get_room_name(target_player)
                    msg = f"Shuffling {query}" if shuffle else f"Playing {query}"
                    return {"status": "playing", "message": f"{msg} in the {room_name}"}

                # ===== TRANSFER ACTION =====
                elif action == "transfer":
                    if not room:
                        return {"error": "Where would you like to transfer the music?"}

                    target_player = find_player_for_room(room)
                    if not target_player:
                        return {"error": f"No player found for room '{room}'"}

                    _LOGGER.info("Transferring queue to %s", target_player)

                    # Music Assistant transfer_queue
                    # source_player omitted = "first playing player" (per HA docs)
                    await self.hass.services.async_call(
                        "music_assistant", "transfer_queue",
                        {"auto_play": True},
                        target={"entity_id": target_player},
                        blocking=True
                    )

                    # Update last active speaker helper
                    await set_last_active_player(target_player)

                    room_name = get_room_name(target_player)
                    return {"status": "transferred", "message": f"Music transferred to the {room_name}"}

                # ===== SKIP ACTION =====
                elif action == "skip":
                    # Priority: 1. last_active_speaker helper, 2. currently playing, 3. default
                    target_player = get_last_active_player()
                    if not target_player:
                        target_player = find_playing_player()
                    if not target_player:
                        target_player = default_player

                    if not target_player:
                        return {"error": "No music player found. Please specify a room or configure a default player."}

                    _LOGGER.info("Skipping to next track on %s", target_player)
                    await self.hass.services.async_call(
                        "media_player", "media_next_track",
                        {"entity_id": target_player},
                        blocking=True
                    )
                    return {"status": "skipped", "message": "Skipped to next track"}

                # ===== PREVIOUS ACTION =====
                elif action == "previous":
                    # Priority: 1. last_active_speaker helper, 2. currently playing, 3. default
                    target_player = get_last_active_player()
                    if not target_player:
                        target_player = find_playing_player()
                    if not target_player:
                        target_player = default_player

                    if not target_player:
                        return {"error": "No music player found. Please specify a room or configure a default player."}

                    _LOGGER.info("Going to previous track on %s", target_player)
                    await self.hass.services.async_call(
                        "media_player", "media_previous_track",
                        {"entity_id": target_player},
                        blocking=True
                    )
                    return {"status": "previous", "message": "Playing previous track"}

                # ===== PAUSE ACTION =====
                elif action == "pause":
                    # Priority: 1. last_active_speaker helper, 2. currently playing, 3. default
                    target_player = get_last_active_player()
                    if not target_player:
                        target_player = find_playing_player()
                    if not target_player:
                        target_player = default_player

                    if not target_player:
                        return {"error": "No music player found. Please specify a room or configure a default player."}

                    _LOGGER.info("Pausing music on %s", target_player)
                    await self.hass.services.async_call(
                        "media_player", "media_pause",
                        {"entity_id": target_player},
                        blocking=True
                    )
                    return {"status": "paused", "message": "Music paused"}

                # ===== RESUME ACTION =====
                elif action == "resume":
                    # Priority: 1. last_active_speaker helper, 2. paused player, 3. default
                    target_player = get_last_active_player()
                    if not target_player:
                        # Find paused player
                        for player in all_players:
                            state = self.hass.states.get(player)
                            if state and state.state == "paused":
                                target_player = player
                                break
                    if not target_player:
                        target_player = default_player

                    if not target_player:
                        return {"error": "No music player found. Please specify a room or configure a default player."}

                    _LOGGER.info("Resuming music on %s", target_player)
                    await self.hass.services.async_call(
                        "media_player", "media_play",
                        {"entity_id": target_player},
                        blocking=True
                    )
                    return {"status": "resumed", "message": "Music resumed"}

                # ===== STOP ACTION =====
                elif action == "stop":
                    # Priority: 1. last_active_speaker helper, 2. currently playing, 3. default
                    target_player = get_last_active_player()
                    if not target_player:
                        target_player = find_playing_player()
                    if not target_player:
                        target_player = default_player

                    if not target_player:
                        return {"error": "No music player found. Please specify a room or configure a default player."}

                    _LOGGER.info("Stopping music on %s", target_player)
                    await self.hass.services.async_call(
                        "media_player", "media_stop",
                        {"entity_id": target_player},
                        blocking=True
                    )
                    return {"status": "stopped", "message": "Music stopped"}

                else:
                    return {"error": f"Unknown action: {action}"}

            except Exception as err:
                _LOGGER.error("Error in control_music: %s", err, exc_info=True)
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
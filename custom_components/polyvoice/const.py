"""Constants for PolyVoice."""
from typing import Final

DOMAIN: Final = "polyvoice"

# =============================================================================
# LLM PROVIDER SETTINGS
# =============================================================================
CONF_PROVIDER: Final = "provider"
CONF_BASE_URL: Final = "base_url"
CONF_API_KEY: Final = "api_key"
CONF_MODEL: Final = "model"
CONF_TEMPERATURE: Final = "temperature"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_TOP_P: Final = "top_p"

# Provider choices
PROVIDER_LM_STUDIO: Final = "lm_studio"
PROVIDER_OPENAI: Final = "openai"
PROVIDER_ANTHROPIC: Final = "anthropic"
PROVIDER_GOOGLE: Final = "google"
PROVIDER_GROQ: Final = "groq"
PROVIDER_OPENROUTER: Final = "openrouter"

ALL_PROVIDERS: Final = [
    PROVIDER_LM_STUDIO,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
]

PROVIDER_NAMES: Final = {
    PROVIDER_LM_STUDIO: "LM Studio (Local)",
    PROVIDER_OPENAI: "OpenAI",
    PROVIDER_ANTHROPIC: "Anthropic (Claude)",
    PROVIDER_GOOGLE: "Google Gemini",
    PROVIDER_GROQ: "Groq",
    PROVIDER_OPENROUTER: "OpenRouter",
}

# Default base URLs per provider
PROVIDER_BASE_URLS: Final = {
    PROVIDER_LM_STUDIO: "http://localhost:1234/v1",
    PROVIDER_OPENAI: "https://api.openai.com/v1",
    PROVIDER_ANTHROPIC: "https://api.anthropic.com",
    PROVIDER_GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
    PROVIDER_GROQ: "https://api.groq.com/openai/v1",
    PROVIDER_OPENROUTER: "https://openrouter.ai/api/v1",
}

# Default models per provider
PROVIDER_DEFAULT_MODELS: Final = {
    PROVIDER_LM_STUDIO: "local-model",
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_ANTHROPIC: "claude-sonnet-4-20250514",
    PROVIDER_GOOGLE: "gemini-1.5-flash",
    PROVIDER_GROQ: "llama-3.3-70b-versatile",
    PROVIDER_OPENROUTER: "openai/gpt-4o-mini",
}

# Suggested models per provider (for UI hints)
PROVIDER_MODELS: Final = {
    PROVIDER_LM_STUDIO: ["local-model", "qwen2.5-7b-instruct", "llama-3.2-3b"],
    PROVIDER_OPENAI: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    PROVIDER_ANTHROPIC: ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
    PROVIDER_GOOGLE: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
    PROVIDER_GROQ: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    PROVIDER_OPENROUTER: ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "google/gemini-flash-1.5"],
}

# Providers that use OpenAI-compatible API
OPENAI_COMPATIBLE_PROVIDERS: Final = [
    PROVIDER_LM_STUDIO,
    PROVIDER_OPENAI,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
]

DEFAULT_PROVIDER: Final = PROVIDER_LM_STUDIO
DEFAULT_BASE_URL: Final = "http://localhost:1234/v1"
DEFAULT_API_KEY: Final = "lm-studio"
DEFAULT_MODEL: Final = "local-model"
DEFAULT_TEMPERATURE: Final = 0.7
DEFAULT_MAX_TOKENS: Final = 2000
DEFAULT_TOP_P: Final = 0.95

# =============================================================================
# FEATURE TOGGLES - Enable/disable function categories
# =============================================================================
CONF_ENABLE_WEATHER: Final = "enable_weather"
CONF_ENABLE_CALENDAR: Final = "enable_calendar"
CONF_ENABLE_MUSIC: Final = "enable_music"
CONF_ENABLE_CAMERAS: Final = "enable_cameras"
CONF_ENABLE_SPORTS: Final = "enable_sports"
CONF_ENABLE_NEWS: Final = "enable_news"
CONF_ENABLE_PLACES: Final = "enable_places"
CONF_ENABLE_RESTAURANTS: Final = "enable_restaurants"
CONF_ENABLE_THERMOSTAT: Final = "enable_thermostat"
CONF_ENABLE_DEVICE_STATUS: Final = "enable_device_status"
CONF_ENABLE_WIKIPEDIA: Final = "enable_wikipedia"

DEFAULT_ENABLE_WEATHER: Final = True
DEFAULT_ENABLE_CALENDAR: Final = True
DEFAULT_ENABLE_MUSIC: Final = True
DEFAULT_ENABLE_CAMERAS: Final = False  # Requires vllm_video integration
DEFAULT_ENABLE_SPORTS: Final = True
DEFAULT_ENABLE_NEWS: Final = True
DEFAULT_ENABLE_PLACES: Final = True
DEFAULT_ENABLE_RESTAURANTS: Final = True
DEFAULT_ENABLE_THERMOSTAT: Final = True
DEFAULT_ENABLE_DEVICE_STATUS: Final = True
DEFAULT_ENABLE_WIKIPEDIA: Final = True

# =============================================================================
# ENTITY CONFIGURATION - User-defined entities
# =============================================================================
CONF_THERMOSTAT_ENTITY: Final = "thermostat_entity"
CONF_CALENDAR_ENTITIES: Final = "calendar_entities"
CONF_MUSIC_PLAYERS: Final = "music_players"
CONF_DEFAULT_MUSIC_PLAYER: Final = "default_music_player"
CONF_DEVICE_ALIASES: Final = "device_aliases"

DEFAULT_THERMOSTAT_ENTITY: Final = ""
DEFAULT_CALENDAR_ENTITIES: Final = ""
DEFAULT_MUSIC_PLAYERS: Final = ""
DEFAULT_DEFAULT_MUSIC_PLAYER: Final = ""
DEFAULT_DEVICE_ALIASES: Final = ""

# =============================================================================
# NATIVE INTENTS
# =============================================================================
CONF_USE_NATIVE_INTENTS: Final = "use_native_intents"
CONF_EXCLUDED_INTENTS: Final = "excluded_intents"
CONF_CUSTOM_EXCLUDED_INTENTS: Final = "custom_excluded_intents"
CONF_ENABLE_ASSIST: Final = "enable_assist"
CONF_LLM_HASS_API: Final = "llm_hass_api"

DEFAULT_USE_NATIVE_INTENTS: Final = True
DEFAULT_EXCLUDED_INTENTS: Final = [
    "HassGetState",
    "HassNevermind",
    "HassClimateGetTemperature",
    "HassClimateSetTemperature",
    "HassTimerStatus",
]
DEFAULT_CUSTOM_EXCLUDED_INTENTS: Final = ""
DEFAULT_ENABLE_ASSIST: Final = True
DEFAULT_LLM_HASS_API: Final = "assist"

ALL_NATIVE_INTENTS: Final = [
    "HassClimateGetTemperature",
    "HassClimateSetTemperature",
    "HassGetState",
    "HassLightSet",
    "HassMediaNext",
    "HassMediaPause",
    "HassMediaPrevious",
    "HassMediaUnpause",
    "HassNevermind",
    "HassSetPosition",
    "HassSetVolume",
    "HassTimerCancel",
    "HassTimerStart",
    "HassTimerStatus",
    "HassToggle",
    "HassTurnOff",
    "HassTurnOn",
    "HassVacuumReturnToBase",
    "HassVacuumStart",
]

# =============================================================================
# SYSTEM PROMPT
# =============================================================================
CONF_SYSTEM_PROMPT: Final = "system_prompt"

DEFAULT_SYSTEM_PROMPT: Final = """You are a smart home assistant. Be concise (1-2 sentences for voice responses).
When using tools, respond DIRECTLY with the result. Do NOT say "I'll look this up" or "Let me check" - just give the answer.

[CURRENT_DATE_WILL_BE_INJECTED_HERE]

GENERAL GUIDELINES:
- For weather questions, call get_weather_forecast
- For music control, use control_music with appropriate action
- For camera checks, use the camera check tools
- For thermostat control, use control_thermostat
- For device status, use check_device_status
- For sports questions, ALWAYS call get_sports_info (never answer from memory)
- For Wikipedia/knowledge questions, use get_wikipedia_summary
- For age questions, use calculate_age (never guess ages)
- For places/directions, use find_nearby_places
- For restaurant recommendations, use get_restaurant_recommendations
- For news, use get_news
- For calendar events, use get_calendar_events
- For native HA control (lights, locks), let native HA Assist handle those
"""

# =============================================================================
# LOCATION
# =============================================================================
CONF_CUSTOM_LATITUDE: Final = "custom_latitude"
CONF_CUSTOM_LONGITUDE: Final = "custom_longitude"

# Use 0.0 as default to indicate "use Home Assistant's configured location"
DEFAULT_CUSTOM_LATITUDE: Final = 0.0
DEFAULT_CUSTOM_LONGITUDE: Final = 0.0

# =============================================================================
# API KEYS
# =============================================================================
CONF_OPENWEATHERMAP_API_KEY: Final = "openweathermap_api_key"
CONF_GOOGLE_PLACES_API_KEY: Final = "google_places_api_key"
CONF_YELP_API_KEY: Final = "yelp_api_key"
CONF_NEWSAPI_KEY: Final = "newsapi_key"

DEFAULT_OPENWEATHERMAP_API_KEY: Final = ""
DEFAULT_GOOGLE_PLACES_API_KEY: Final = ""
DEFAULT_YELP_API_KEY: Final = ""
DEFAULT_NEWSAPI_KEY: Final = ""

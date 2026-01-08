"""Config flow for PureLLM integration."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
import homeassistant.helpers.config_validation as cv

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
    ALL_PROVIDERS,
    PROVIDER_NAMES,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_MODELS,
    PROVIDER_LM_STUDIO,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
    PROVIDER_AZURE,
    PROVIDER_OLLAMA,
    DEFAULT_PROVIDER,
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    # System settings
    CONF_SYSTEM_PROMPT,
    CONF_CUSTOM_LATITUDE,
    CONF_CUSTOM_LONGITUDE,
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
    CONF_CALENDAR_ENTITIES,
    CONF_ROOM_PLAYER_MAPPING,
    CONF_DEVICE_ALIASES,
    CONF_CAMERA_ENTITIES,
    # Thermostat settings
    CONF_THERMOSTAT_MIN_TEMP,
    CONF_THERMOSTAT_MAX_TEMP,
    CONF_THERMOSTAT_TEMP_STEP,
    CONF_THERMOSTAT_USE_CELSIUS,
    # Defaults
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_CUSTOM_LATITUDE,
    DEFAULT_CUSTOM_LONGITUDE,
    DEFAULT_OPENWEATHERMAP_API_KEY,
    DEFAULT_GOOGLE_PLACES_API_KEY,
    DEFAULT_YELP_API_KEY,
    DEFAULT_NEWSAPI_KEY,
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
    DEFAULT_THERMOSTAT_ENTITY,
    DEFAULT_CALENDAR_ENTITIES,
    DEFAULT_ROOM_PLAYER_MAPPING,
    DEFAULT_DEVICE_ALIASES,
    DEFAULT_CAMERA_ENTITIES,
    # Thermostat defaults
    DEFAULT_THERMOSTAT_MIN_TEMP,
    DEFAULT_THERMOSTAT_MAX_TEMP,
    DEFAULT_THERMOSTAT_TEMP_STEP,
    DEFAULT_THERMOSTAT_USE_CELSIUS,
    DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS,
)

_LOGGER = logging.getLogger(__name__)

# Providers that support dynamic model fetching via OpenAI-compatible /models endpoint
DYNAMIC_MODEL_PROVIDERS = [
    PROVIDER_OPENAI,
    PROVIDER_GROQ,
    PROVIDER_OPENROUTER,
    PROVIDER_LM_STUDIO,
    PROVIDER_OLLAMA,
    PROVIDER_GOOGLE,
]


async def fetch_provider_models(
    provider: str, base_url: str, api_key: str
) -> list[str]:
    """Fetch available models from provider API.

    Returns a list of model IDs, or empty list if fetch fails.
    """
    if provider not in DYNAMIC_MODEL_PROVIDERS:
        return []

    try:
        async with aiohttp.ClientSession() as session:
            # Google Gemini uses different API format
            if provider == PROVIDER_GOOGLE:
                # Request max models per page - use header auth instead of URL param
                url = f"{base_url}/models?pageSize=1000"
                headers = {"x-goog-api-key": api_key}
            else:
                # OpenAI-compatible providers
                url = f"{base_url}/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    _LOGGER.warning(
                        "Failed to fetch models from %s: status %s",
                        provider, response.status
                    )
                    return []

                data = await response.json()

                # Google returns {"models": [...]} with "name" field
                # OpenAI-compatible returns {"data": [...]} with "id" field
                if provider == PROVIDER_GOOGLE:
                    models = data.get("models", [])
                    _LOGGER.warning("Google API returned %d total models", len(models))
                    model_ids = []
                    for model in models:
                        # Google format: "models/gemini-1.5-pro" -> "gemini-1.5-pro"
                        name = model.get("name", "")
                        if not name:
                            continue

                        # Extract model ID from full name
                        model_id = name.replace("models/", "")

                        # Include all gemini models (filter out embedding/aqa/imagen)
                        lower_id = model_id.lower()
                        if any(skip in lower_id for skip in ["embedding", "aqa", "imagen"]):
                            continue

                        model_ids.append(model_id)

                    _LOGGER.warning("Google filtered to %d gemini models: %s", len(model_ids), model_ids[:10])
                else:
                    # OpenAI-compatible providers
                    models = data.get("data", [])
                    model_ids = []
                    for model in models:
                        model_id = model.get("id", "")
                        if not model_id:
                            continue

                        # Skip non-chat models
                        lower_id = model_id.lower()
                        if any(skip in lower_id for skip in [
                            "whisper", "tts", "embedding", "embed",
                            "guard", "safeguard", "moderation",
                            "audio", "speech", "vision-preview"
                        ]):
                            continue

                        model_ids.append(model_id)

                # Sort alphabetically for better UX
                model_ids.sort()
                _LOGGER.info("Fetched %d models from %s", len(model_ids), provider)
                return model_ids

    except Exception as e:
        _LOGGER.warning("Error fetching models from %s: %s", provider, e)
        return []


class PureLLMConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PureLLM."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize config flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - provider selection."""
        if user_input is not None:
            self._data[CONF_PROVIDER] = user_input[CONF_PROVIDER]
            return await self.async_step_credentials()

        # Build provider options for selector
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_PROVIDER, default=DEFAULT_PROVIDER): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=provider_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle credentials step."""
        errors = {}
        provider = self._data.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        
        if user_input is not None:
            self._data.update(user_input)
            
            # Set base URL based on provider if not custom
            if not user_input.get(CONF_BASE_URL):
                self._data[CONF_BASE_URL] = PROVIDER_BASE_URLS.get(provider, DEFAULT_BASE_URL)
            
            # Validate connection
            try:
                valid = await self._test_connection(
                    provider,
                    self._data.get(CONF_BASE_URL, PROVIDER_BASE_URLS.get(provider)),
                    self._data.get(CONF_API_KEY, ""),
                )
                if not valid:
                    errors["base"] = "cannot_connect"
            except Exception as e:
                _LOGGER.error("Connection test failed: %s", e)
                errors["base"] = "cannot_connect"
            
            if not errors:
                return self.async_create_entry(
                    title=f"PureLLM ({PROVIDER_NAMES[provider]})",
                    data=self._data,
                )

        # Get defaults for this provider
        default_url = PROVIDER_BASE_URLS.get(provider, DEFAULT_BASE_URL)
        default_model = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)

        # Show different fields based on provider
        # Show URL for local providers (LM Studio, Ollama) and Azure (requires custom endpoint)
        show_base_url = provider in [PROVIDER_LM_STUDIO, PROVIDER_OLLAMA, PROVIDER_AZURE]

        schema_dict = {}

        if show_base_url:
            # Azure needs a placeholder hint since URL is required
            if provider == PROVIDER_AZURE:
                schema_dict[vol.Required(CONF_BASE_URL, default="")] = str
            else:
                schema_dict[vol.Required(CONF_BASE_URL, default=default_url)] = str

        # API key: optional for Ollama, required for others
        if provider == PROVIDER_OLLAMA:
            schema_dict[vol.Optional(CONF_API_KEY, default="")] = str
        elif provider == PROVIDER_LM_STUDIO:
            schema_dict[vol.Required(CONF_API_KEY, default="lm-studio")] = str
        else:
            schema_dict[vol.Required(CONF_API_KEY, default="")] = str

        # Build model options for the current provider
        provider_models = PROVIDER_MODELS.get(provider, [default_model])
        model_options = [
            selector.SelectOptionDict(value=m, label=m)
            for m in provider_models
        ]

        schema_dict[vol.Required(CONF_MODEL, default=default_model)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=model_options,
                mode=selector.SelectSelectorMode.DROPDOWN,
                custom_value=True,
            )
        )

        return self.async_show_form(
            step_id="credentials",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
            description_placeholders={
                "provider": PROVIDER_NAMES[provider],
            },
        )

    async def _test_connection(self, provider: str, base_url: str, api_key: str) -> bool:
        """Test connection to the LLM provider."""
        try:
            async with aiohttp.ClientSession() as session:
                if provider == PROVIDER_ANTHROPIC:
                    # Anthropic uses different endpoint
                    headers = {
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    }
                    async with session.get(
                        f"{base_url}/v1/models",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        return response.status in (200, 401)  # 401 means API is reachable but key invalid
                elif provider == PROVIDER_GOOGLE:
                    # Google Gemini
                    async with session.get(
                        f"{base_url}/models?key={api_key}",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        return response.status in (200, 400, 403)
                elif provider == PROVIDER_AZURE:
                    # Azure OpenAI uses api-key header
                    if not base_url:
                        return False  # Azure requires a base URL
                    headers = {"api-key": api_key} if api_key else {}
                    # Azure endpoint format: https://{resource}.openai.azure.com/openai/deployments/{deployment}
                    # Test with models endpoint at the resource level
                    # Extract resource URL from deployment URL
                    try:
                        # Try to extract base resource URL for testing
                        if "/openai/deployments/" in base_url:
                            resource_url = base_url.split("/openai/deployments/")[0]
                            test_url = f"{resource_url}/openai/models?api-version=2024-02-01"
                        else:
                            test_url = f"{base_url}/openai/models?api-version=2024-02-01"
                        async with session.get(
                            test_url,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            return response.status in (200, 401, 403)
                    except Exception:
                        return True  # Allow setup if URL parsing fails
                elif provider == PROVIDER_OLLAMA:
                    # Ollama - no auth required, just check if server is responding
                    async with session.get(
                        f"{base_url}/models",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        return response.status in (200, 401)
                else:
                    # OpenAI-compatible (LM Studio, OpenAI, Groq, OpenRouter)
                    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    async with session.get(
                        f"{base_url}/models",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        return response.status in (200, 401)
        except Exception as e:
            _LOGGER.warning("Connection test exception: %s", e)
            # For local providers, connection refused is expected if server isn't running
            if provider in [PROVIDER_LM_STUDIO, PROVIDER_OLLAMA]:
                return True  # Allow setup even if server isn't running
            return False

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return PureLLMOptionsFlowHandler(config_entry)


class PureLLMOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for PureLLM."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "connection": "Connection Settings",
                "model": "Model Settings",
                "features": "Enable/Disable Features",
                "entities": "PureLLM Default Entities",
                "device_aliases": "Device Aliases",
                "music_rooms": "Music Room Mapping",
                "api_keys": "API Keys",
                "location": "Location Settings",
                "advanced": "System Prompt",
            },
        )

    async def async_step_connection(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle connection settings including provider change."""
        if user_input is not None:
            # If provider changed, update base_url to default for new provider
            new_provider = user_input.get(CONF_PROVIDER)
            old_provider = self._entry.data.get(CONF_PROVIDER) or self._entry.options.get(CONF_PROVIDER)
            
            if new_provider and new_provider != old_provider:
                # Set default URL for new provider unless custom URL provided
                if not user_input.get(CONF_BASE_URL) or user_input.get(CONF_BASE_URL) == PROVIDER_BASE_URLS.get(old_provider):
                    user_input[CONF_BASE_URL] = PROVIDER_BASE_URLS.get(new_provider, DEFAULT_BASE_URL)
            
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        current_provider = current.get(CONF_PROVIDER, DEFAULT_PROVIDER)

        # Build provider options for selector
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        return self.async_show_form(
            step_id="connection",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_PROVIDER,
                        default=current_provider,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=provider_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_BASE_URL,
                        default=current.get(CONF_BASE_URL, PROVIDER_BASE_URLS.get(current_provider, DEFAULT_BASE_URL)),
                    ): str,
                    vol.Required(
                        CONF_API_KEY,
                        default=current.get(CONF_API_KEY, DEFAULT_API_KEY),
                    ): str,
                }
            ),
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle model settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        current_provider = current.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        current_model = current.get(CONF_MODEL, DEFAULT_MODEL)
        current_base_url = current.get(CONF_BASE_URL, PROVIDER_BASE_URLS.get(current_provider, DEFAULT_BASE_URL))
        current_api_key = current.get(CONF_API_KEY, "")

        # Try to fetch models dynamically from the provider API
        provider_models = await fetch_provider_models(
            current_provider, current_base_url, current_api_key
        )

        # Fall back to static list if dynamic fetch fails or returns empty
        if not provider_models:
            provider_models = list(PROVIDER_MODELS.get(current_provider, [DEFAULT_MODEL]))
            _LOGGER.debug("Using static model list for %s", current_provider)

        # Ensure current model is in the list (for custom models)
        if current_model and current_model not in provider_models:
            provider_models = [current_model] + provider_models

        model_options = [
            selector.SelectOptionDict(value=m, label=m)
            for m in provider_models
        ]

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=current_model,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                            custom_value=True,
                        )
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=current.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=current.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                    ): cv.positive_int,
                    vol.Optional(
                        CONF_TOP_P,
                        default=current.get(CONF_TOP_P, DEFAULT_TOP_P),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                }
            ),
        )

    async def async_step_features(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle feature toggles."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="features",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_ENABLE_WEATHER,
                        default=current.get(CONF_ENABLE_WEATHER, DEFAULT_ENABLE_WEATHER),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_CALENDAR,
                        default=current.get(CONF_ENABLE_CALENDAR, DEFAULT_ENABLE_CALENDAR),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_CAMERAS,
                        default=current.get(CONF_ENABLE_CAMERAS, DEFAULT_ENABLE_CAMERAS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_SPORTS,
                        default=current.get(CONF_ENABLE_SPORTS, DEFAULT_ENABLE_SPORTS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_STOCKS,
                        default=current.get(CONF_ENABLE_STOCKS, DEFAULT_ENABLE_STOCKS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_NEWS,
                        default=current.get(CONF_ENABLE_NEWS, DEFAULT_ENABLE_NEWS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_PLACES,
                        default=current.get(CONF_ENABLE_PLACES, DEFAULT_ENABLE_PLACES),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_RESTAURANTS,
                        default=current.get(CONF_ENABLE_RESTAURANTS, DEFAULT_ENABLE_RESTAURANTS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_THERMOSTAT,
                        default=current.get(CONF_ENABLE_THERMOSTAT, DEFAULT_ENABLE_THERMOSTAT),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_DEVICE_STATUS,
                        default=current.get(CONF_ENABLE_DEVICE_STATUS, DEFAULT_ENABLE_DEVICE_STATUS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_WIKIPEDIA,
                        default=current.get(CONF_ENABLE_WIKIPEDIA, DEFAULT_ENABLE_WIKIPEDIA),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_MUSIC,
                        default=current.get(CONF_ENABLE_MUSIC, DEFAULT_ENABLE_MUSIC),
                    ): cv.boolean,
                }
            ),
        )

    async def async_step_entities(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle entity configuration."""
        if user_input is not None:
            # Convert entity lists to the format we need
            processed_input = {}

            # Handle thermostat - single entity
            if CONF_THERMOSTAT_ENTITY in user_input:
                processed_input[CONF_THERMOSTAT_ENTITY] = user_input[CONF_THERMOSTAT_ENTITY]

            # Handle calendars - convert list to newline-separated string
            if CONF_CALENDAR_ENTITIES in user_input:
                cal_list = user_input[CONF_CALENDAR_ENTITIES]
                if isinstance(cal_list, list):
                    processed_input[CONF_CALENDAR_ENTITIES] = "\n".join(cal_list)
                else:
                    processed_input[CONF_CALENDAR_ENTITIES] = cal_list

            # Handle cameras - convert list to newline-separated string
            if CONF_CAMERA_ENTITIES in user_input:
                cam_list = user_input[CONF_CAMERA_ENTITIES]
                if isinstance(cam_list, list):
                    processed_input[CONF_CAMERA_ENTITIES] = "\n".join(cam_list)
                else:
                    processed_input[CONF_CAMERA_ENTITIES] = cam_list

            # Handle thermostat settings
            if CONF_THERMOSTAT_MIN_TEMP in user_input:
                processed_input[CONF_THERMOSTAT_MIN_TEMP] = user_input[CONF_THERMOSTAT_MIN_TEMP]
            if CONF_THERMOSTAT_MAX_TEMP in user_input:
                processed_input[CONF_THERMOSTAT_MAX_TEMP] = user_input[CONF_THERMOSTAT_MAX_TEMP]
            if CONF_THERMOSTAT_TEMP_STEP in user_input:
                processed_input[CONF_THERMOSTAT_TEMP_STEP] = user_input[CONF_THERMOSTAT_TEMP_STEP]
            if CONF_THERMOSTAT_USE_CELSIUS in user_input:
                processed_input[CONF_THERMOSTAT_USE_CELSIUS] = user_input[CONF_THERMOSTAT_USE_CELSIUS]

            new_options = {**self._entry.options, **processed_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        # Parse current calendar entities back to list
        current_calendars = current.get(CONF_CALENDAR_ENTITIES, DEFAULT_CALENDAR_ENTITIES)
        if isinstance(current_calendars, str) and current_calendars:
            current_calendars = [c.strip() for c in current_calendars.split("\n") if c.strip()]
        elif not current_calendars:
            current_calendars = []

        # Parse current camera entities back to list
        current_cameras = current.get(CONF_CAMERA_ENTITIES, DEFAULT_CAMERA_ENTITIES)
        if isinstance(current_cameras, str) and current_cameras:
            current_cameras = [c.strip() for c in current_cameras.split("\n") if c.strip()]
        elif not current_cameras:
            current_cameras = []

        # Determine if using Celsius and set appropriate defaults/ranges
        use_celsius = current.get(CONF_THERMOSTAT_USE_CELSIUS, DEFAULT_THERMOSTAT_USE_CELSIUS)
        if use_celsius:
            temp_unit = "°C"
            temp_min_range = 0
            temp_max_range = 50
            default_min = DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS
            default_max = DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS
        else:
            temp_unit = "°F"
            temp_min_range = 32
            temp_max_range = 100
            default_min = DEFAULT_THERMOSTAT_MIN_TEMP
            default_max = DEFAULT_THERMOSTAT_MAX_TEMP
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP

        # Get current values, using appropriate defaults if not set or if switching units
        current_min = current.get(CONF_THERMOSTAT_MIN_TEMP)
        current_max = current.get(CONF_THERMOSTAT_MAX_TEMP)
        current_step = current.get(CONF_THERMOSTAT_TEMP_STEP)

        # If values look like wrong unit (e.g., 60-85 with Celsius enabled), use unit defaults
        if use_celsius:
            if current_min is not None and current_min > 40:  # Likely Fahrenheit value
                current_min = default_min
            if current_max is not None and current_max > 50:  # Likely Fahrenheit value
                current_max = default_max
        else:
            if current_min is not None and current_min < 32:  # Likely Celsius value
                current_min = default_min
            if current_max is not None and current_max < 50:  # Likely Celsius value
                current_max = default_max

        # Use defaults if not set
        if current_min is None:
            current_min = default_min
        if current_max is None:
            current_max = default_max
        if current_step is None:
            current_step = default_step

        return self.async_show_form(
            step_id="entities",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_THERMOSTAT_ENTITY,
                        default=current.get(CONF_THERMOSTAT_ENTITY, DEFAULT_THERMOSTAT_ENTITY),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="climate",
                            multiple=False,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMOSTAT_USE_CELSIUS,
                        default=use_celsius,
                    ): cv.boolean,
                    vol.Optional(
                        CONF_CALENDAR_ENTITIES,
                        default=current_calendars,
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="calendar",
                            multiple=True,
                        )
                    ),
                    vol.Optional(
                        CONF_CAMERA_ENTITIES,
                        default=current_cameras,
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="camera",
                            multiple=True,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMOSTAT_MIN_TEMP,
                        default=current_min,
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=temp_min_range,
                            max=temp_max_range,
                            step=1,
                            unit_of_measurement=temp_unit,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMOSTAT_MAX_TEMP,
                        default=current_max,
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=temp_min_range,
                            max=temp_max_range,
                            step=1,
                            unit_of_measurement=temp_unit,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMOSTAT_TEMP_STEP,
                        default=current_step,
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            max=10,
                            step=1,
                            unit_of_measurement=temp_unit,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                }
            ),
        )

    async def async_step_device_aliases(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle device aliases configuration.

        Device aliases allow users to define custom voice names for entities.
        Format: alias_name: entity_id (one per line)
        """
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="device_aliases",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_DEVICE_ALIASES,
                        default=current.get(CONF_DEVICE_ALIASES, DEFAULT_DEVICE_ALIASES),
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.TEXT,
                            multiline=True,
                        )
                    ),
                }
            ),
            description_placeholders={
                "format_hint": "Format: alias_name: entity_id (one per line). Example:\nkitchen light: light.kitchen_ceiling\nfront door: lock.front_door",
            },
        )

    async def async_step_music_rooms(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle music room to player mapping configuration with edit/delete support."""
        current = {**self._entry.data, **self._entry.options}
        current_mapping = current.get(CONF_ROOM_PLAYER_MAPPING, "")

        # Parse current mappings into dict {room_name: entity_id}
        rooms_dict = {}
        if current_mapping:
            for line in current_mapping.split("\n"):
                line = line.strip()
                if ": " in line:
                    room_name, entity_id = line.split(": ", 1)
                    rooms_dict[room_name.strip().lower()] = entity_id.strip()

        if user_input is not None:
            selected = user_input.get("select_room", "")
            new_player = user_input.get("room_player", "")
            new_room = user_input.get("room_name", "").strip().lower()
            action = user_input.get("action", "add")

            if action == "delete" and selected:
                # Delete selected room
                if selected in rooms_dict:
                    del rooms_dict[selected]
            elif action == "update" and selected and new_player:
                # Update selected room (delete old, add new)
                if selected in rooms_dict:
                    del rooms_dict[selected]
                room_key = new_room if new_room else selected
                rooms_dict[room_key] = new_player
            elif action == "add" and new_player and new_room:
                # Add new room mapping
                rooms_dict[new_room] = new_player
            elif not selected and not new_player and not new_room:
                # Empty submit - return to menu
                return self.async_create_entry(title="", data=self._entry.options)

            # Save updated mappings
            if rooms_dict:
                updated_mapping = "\n".join([f"{k}: {v}" for k, v in rooms_dict.items()])
            else:
                updated_mapping = ""

            new_options = {**self._entry.options, CONF_ROOM_PLAYER_MAPPING: updated_mapping}
            self.hass.config_entries.async_update_entry(self._entry, options=new_options)
            current_mapping = updated_mapping
            # Rebuild dict for display
            rooms_dict = {}
            if current_mapping:
                for line in current_mapping.split("\n"):
                    line = line.strip()
                    if ": " in line:
                        room_name, entity_id = line.split(": ", 1)
                        rooms_dict[room_name.strip().lower()] = entity_id.strip()

        # Build description showing current mappings
        if rooms_dict:
            mapping_display = "\n".join([f"• {k} → {v}" for k, v in rooms_dict.items()])
            description = f"**Current room mappings:**\n{mapping_display}\n\nSelect one to edit/delete, or add a new one below."
        else:
            description = "No room mappings configured. Add your first room below."

        # Build select options for existing rooms (no "none" option - just leave empty to add new)
        select_options = []
        for room_name in rooms_dict.keys():
            select_options.append(selector.SelectOptionDict(value=room_name, label=f"{room_name} → {rooms_dict[room_name]}"))

        return self.async_show_form(
            step_id="music_rooms",
            description_placeholders={"mappings": description},
            data_schema=vol.Schema(
                {
                    vol.Optional("select_room"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=select_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("room_player"): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="media_player",
                            multiple=False,
                        )
                    ),
                    vol.Optional("room_name"): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.TEXT,
                        )
                    ),
                    vol.Optional("action", default="add"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value="add", label="Add New"),
                                selector.SelectOptionDict(value="update", label="Update Selected"),
                                selector.SelectOptionDict(value="delete", label="Delete Selected"),
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_api_keys(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle API key configuration."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="api_keys",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_OPENWEATHERMAP_API_KEY,
                        default=current.get(CONF_OPENWEATHERMAP_API_KEY, DEFAULT_OPENWEATHERMAP_API_KEY),
                    ): str,
                    vol.Optional(
                        CONF_GOOGLE_PLACES_API_KEY,
                        default=current.get(CONF_GOOGLE_PLACES_API_KEY, DEFAULT_GOOGLE_PLACES_API_KEY),
                    ): str,
                    vol.Optional(
                        CONF_YELP_API_KEY,
                        default=current.get(CONF_YELP_API_KEY, DEFAULT_YELP_API_KEY),
                    ): str,
                    vol.Optional(
                        CONF_NEWSAPI_KEY,
                        default=current.get(CONF_NEWSAPI_KEY, DEFAULT_NEWSAPI_KEY),
                    ): str,
                }
            ),
        )

    async def async_step_location(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle location configuration."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="location",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_CUSTOM_LATITUDE,
                        default=current.get(CONF_CUSTOM_LATITUDE, DEFAULT_CUSTOM_LATITUDE),
                    ): vol.Coerce(float),
                    vol.Optional(
                        CONF_CUSTOM_LONGITUDE,
                        default=current.get(CONF_CUSTOM_LONGITUDE, DEFAULT_CUSTOM_LONGITUDE),
                    ): vol.Coerce(float),
                }
            ),
            description_placeholders={
                "location_note": "Leave as 0 to use Home Assistant's configured location",
            },
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle advanced settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="advanced",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_SYSTEM_PROMPT,
                        description={"suggested_value": current.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)},
                    ): selector.TemplateSelector(),
                }
            ),
        )
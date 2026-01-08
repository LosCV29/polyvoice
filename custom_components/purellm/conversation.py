"""PureLLM Conversation Entity - Pure LLM Voice Assistant v4.0.

This is the main conversation entity that handles ALL voice commands
through the LLM pipeline with tool calling. No native HA intent interception.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, TYPE_CHECKING

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from openai import AsyncOpenAI, AsyncAzureOpenAI

from .const import (
    DOMAIN,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CALENDAR_ENTITIES,
    CONF_CAMERA_ENTITIES,
    CONF_CUSTOM_LATITUDE,
    CONF_CUSTOM_LONGITUDE,
    CONF_DEVICE_ALIASES,
    CONF_ENABLE_CALENDAR,
    CONF_ENABLE_CAMERAS,
    CONF_ENABLE_DEVICE_STATUS,
    CONF_ENABLE_MUSIC,
    CONF_ENABLE_NEWS,
    CONF_ENABLE_PLACES,
    CONF_ENABLE_RESTAURANTS,
    CONF_ENABLE_SPORTS,
    CONF_ENABLE_STOCKS,
    CONF_ENABLE_THERMOSTAT,
    CONF_ENABLE_WEATHER,
    CONF_ENABLE_WIKIPEDIA,
    CONF_GOOGLE_PLACES_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_NEWSAPI_KEY,
    CONF_OPENWEATHERMAP_API_KEY,
    CONF_PROVIDER,
    CONF_ROOM_PLAYER_MAPPING,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_THERMOSTAT_ENTITY,
    CONF_THERMOSTAT_MAX_TEMP,
    CONF_THERMOSTAT_MIN_TEMP,
    CONF_THERMOSTAT_TEMP_STEP,
    CONF_THERMOSTAT_USE_CELSIUS,
    CONF_TOP_P,
    CONF_YELP_API_KEY,
    DEFAULT_API_KEY,
    DEFAULT_ENABLE_CALENDAR,
    DEFAULT_ENABLE_CAMERAS,
    DEFAULT_ENABLE_DEVICE_STATUS,
    DEFAULT_ENABLE_MUSIC,
    DEFAULT_ENABLE_NEWS,
    DEFAULT_ENABLE_PLACES,
    DEFAULT_ENABLE_RESTAURANTS,
    DEFAULT_ENABLE_SPORTS,
    DEFAULT_ENABLE_STOCKS,
    DEFAULT_ENABLE_THERMOSTAT,
    DEFAULT_ENABLE_WEATHER,
    DEFAULT_ENABLE_WIKIPEDIA,
    DEFAULT_PROVIDER,
    DEFAULT_ROOM_PLAYER_MAPPING,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_THERMOSTAT_MAX_TEMP,
    DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_MIN_TEMP,
    DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_TEMP_STEP,
    DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS,
    DEFAULT_THERMOSTAT_USE_CELSIUS,
    OPENAI_COMPATIBLE_PROVIDERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE,
    PROVIDER_BASE_URLS,
    PROVIDER_GOOGLE,
    get_version,
)

# Import from new modules
from .utils.parsing import parse_entity_config, parse_list_config
from .utils.fuzzy_matching import find_entity_by_name
from .utils.helpers import format_human_readable_state, get_friendly_name

from .tools.definitions import build_tools, ToolConfig
from .tools.registry import ToolRegistry

# Tool handlers
from .tools import weather as weather_tool
from .tools import sports as sports_tool
from .tools import stocks as stocks_tool
from .tools import news as news_tool
from .tools import places as places_tool
from .tools import wikipedia as wikipedia_tool
from .tools import calendar as calendar_tool
from .tools import camera as camera_tool
from .tools import thermostat as thermostat_tool
from .tools import device as device_tool
from .tools.music import MusicController
from .tools import timer as timer_tool
from .tools import lists as lists_tool
from .tools import reminders as reminders_tool

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Query patterns for simple responses (no LLM needed)
SIMPLE_QUERY_PATTERNS = [
    (r"\b(what('?s| is) the (current )?(time|date)|what time is it|what day is it)\b", "datetime"),
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entity."""
    agent = PureLLMConversationEntity(config_entry)
    async_add_entities([agent])

    # Store agent reference for service calls
    hass.data.setdefault("purellm", {})
    hass.data["purellm"][config_entry.entry_id] = agent


class PureLLMConversationEntity(ConversationEntity):
    """PureLLM conversation agent entity - Pure LLM pipeline."""

    _attr_has_entity_name = True
    _attr_name = None

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages - use MATCH_ALL for all languages."""
        return conversation.MATCH_ALL

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = config_entry
        self._attr_unique_id = config_entry.entry_id
        self._session: aiohttp.ClientSession | None = None

        # Usage tracking
        self._api_calls = {
            "weather": 0, "places": 0, "restaurants": 0, "news": 0,
            "sports": 0, "wikipedia": 0, "llm": 0, "stocks": 0,
        }
        self._tokens_used = {"input": 0, "output": 0}

        # Caches
        self._tools: list[dict] | None = None
        self._cached_system_prompt: str | None = None
        self._cached_system_prompt_date: str | None = None

        # Music controller (initialized after config)
        self._music_controller: MusicController | None = None

        # Current query for tool context
        self._current_user_query: str = ""

        # Initialize config
        self._update_from_config({**config_entry.data, **config_entry.options})

    @property
    def device_info(self):
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self.entry.entry_id)},
            "name": self.entry.title,
            "manufacturer": "LosCV29",
            "model": "Voice Assistant",
            "entry_type": "service",
            "sw_version": get_version(),
        }

    def _update_from_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        # Provider settings
        self.provider = config.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        self.api_key = config.get(CONF_API_KEY, DEFAULT_API_KEY)
        self.model = config.get(CONF_MODEL, "")
        self.temperature = config.get(CONF_TEMPERATURE, 0.7)
        self.max_tokens = config.get(CONF_MAX_TOKENS, 2000)
        self.top_p = config.get(CONF_TOP_P, 0.95)

        # Base URL
        base_url = config.get(CONF_BASE_URL)
        if not base_url:
            base_url = PROVIDER_BASE_URLS.get(self.provider, "http://localhost:1234/v1")
        self.base_url = base_url

        # Mark client for deferred initialization (avoid blocking SSL on event loop)
        self.client = None
        self._client_init_needed = True

        # Conversation features
        self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL
        self.system_prompt = config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)

        # Custom location
        try:
            lat = float(config.get(CONF_CUSTOM_LATITUDE) or 0)
            self.custom_latitude = lat if lat != 0 else None
        except (ValueError, TypeError):
            self.custom_latitude = None

        try:
            lon = float(config.get(CONF_CUSTOM_LONGITUDE) or 0)
            self.custom_longitude = lon if lon != 0 else None
        except (ValueError, TypeError):
            self.custom_longitude = None

        # API keys
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

        # Entity configuration
        self.room_player_mapping = parse_entity_config(config.get(CONF_ROOM_PLAYER_MAPPING, DEFAULT_ROOM_PLAYER_MAPPING))
        self.thermostat_entity = config.get(CONF_THERMOSTAT_ENTITY, "")
        self.calendar_entities = parse_list_config(config.get(CONF_CALENDAR_ENTITIES, ""))
        self.camera_entities = parse_list_config(config.get(CONF_CAMERA_ENTITIES, ""))
        self.device_aliases = parse_entity_config(config.get(CONF_DEVICE_ALIASES, ""))

        # Thermostat settings
        self.thermostat_use_celsius = config.get(CONF_THERMOSTAT_USE_CELSIUS, DEFAULT_THERMOSTAT_USE_CELSIUS)
        if self.thermostat_use_celsius:
            default_min, default_max = DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS, DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS
        else:
            default_min, default_max = DEFAULT_THERMOSTAT_MIN_TEMP, DEFAULT_THERMOSTAT_MAX_TEMP
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP

        self.thermostat_min_temp = config.get(CONF_THERMOSTAT_MIN_TEMP) or default_min
        self.thermostat_max_temp = config.get(CONF_THERMOSTAT_MAX_TEMP) or default_max
        self.thermostat_temp_step = config.get(CONF_THERMOSTAT_TEMP_STEP) or default_step

        # Clear caches on config update
        self._tools = None
        self._cached_system_prompt = None

    @property
    def temp_unit(self) -> str:
        """Get temperature unit string."""
        return "°C" if self.thermostat_use_celsius else "°F"

    def format_temp(self, temp: float | int | None) -> str:
        """Format temperature with unit."""
        if temp is None:
            return "unknown"
        return f"{int(temp)}{self.temp_unit}"

    def _create_openai_client(self):
        """Create OpenAI client (runs in executor to avoid blocking SSL)."""
        if self.provider == PROVIDER_AZURE:
            azure_endpoint = self.base_url
            if "/openai/deployments/" in azure_endpoint:
                azure_endpoint = azure_endpoint.split("/openai/deployments/")[0]
            return AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=self.api_key,
                api_version="2024-02-01",
            )
        elif self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            return AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key if self.api_key else "ollama",
                timeout=60.0,
                max_retries=2,
            )
        return None

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()

        # Initialize OpenAI client in executor (SSL cert loading is blocking)
        if self._client_init_needed:
            self.client = await self.hass.async_add_executor_job(self._create_openai_client)
            self._client_init_needed = False

        # Initialize shared session
        self._session = async_get_clientsession(self.hass)

        # Initialize music controller
        if self.enable_music and self.room_player_mapping:
            self._music_controller = MusicController(self.hass, self.room_player_mapping)

        # Listen for config updates
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_config_updated)
        )

    @staticmethod
    async def _async_config_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle config entry update."""
        await hass.config_entries.async_reload(entry.entry_id)

    def _track_api_call(self, api_name: str) -> None:
        """Track API usage."""
        if api_name in self._api_calls:
            self._api_calls[api_name] += 1

    def _get_effective_system_prompt(self) -> str:
        """Get system prompt with current date."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cached_system_prompt and self._cached_system_prompt_date == today:
            return self._cached_system_prompt

        prompt = self.system_prompt.replace("{current_date}", today)
        self._cached_system_prompt = prompt
        self._cached_system_prompt_date = today
        return prompt

    def _build_tools(self) -> list[dict]:
        """Build tools list based on enabled features."""
        if self._tools is not None:
            return self._tools

        self._tools = build_tools(ToolConfig(self))
        return self._tools

    async def async_process(
        self,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Process user input."""
        user_text = user_input.text.strip()
        self._current_user_query = user_text

        _LOGGER.debug("Processing query: '%s'", user_text)

        # Check for simple queries that don't need LLM
        for pattern, query_type in SIMPLE_QUERY_PATTERNS:
            if re.search(pattern, user_text, re.IGNORECASE):
                if query_type == "datetime":
                    now = datetime.now(dt_util.get_time_zone(self.hass.config.time_zone))
                    response = now.strftime("It's %I:%M %p on %A, %B %d, %Y")
                    return self._create_response(response, user_input)

        # Build tools and system prompt
        tools = self._build_tools()
        system_prompt = self._get_effective_system_prompt()
        max_tokens = self._calculate_max_tokens(user_text)

        # Route to appropriate provider
        try:
            if self.provider in OPENAI_COMPATIBLE_PROVIDERS or self.provider == PROVIDER_AZURE:
                response = await self._call_openai_compatible(user_input, tools, system_prompt, max_tokens)
            elif self.provider == PROVIDER_ANTHROPIC:
                response = await self._call_anthropic(user_input, tools, system_prompt, max_tokens)
            elif self.provider == PROVIDER_GOOGLE:
                response = await self._call_google(user_input, tools, system_prompt, max_tokens)
            else:
                response = "Unknown provider configured."
        except Exception as err:
            _LOGGER.error("Error processing request: %s", err, exc_info=True)
            response = "Sorry, there was an error processing your request."

        return self._create_response(response, user_input)

    async def async_process_intercepted(self, command: str) -> str:
        """Process an intercepted intent command."""
        _LOGGER.info("Processing intercepted command: '%s'", command)
        self._current_user_query = command

        # Create a minimal conversation input
        user_input = conversation.ConversationInput(
            text=command,
            context=None,
            conversation_id=None,
            language=self.hass.config.language,
            device_id=None,
            satellite_id=None,
            agent_id=self.entity_id,
        )

        result = await self.async_process(user_input)
        return result.response.speech.get("plain", {}).get("speech", "Command executed.")

    def _create_response(
        self,
        text: str,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Create a conversation result."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)
        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )

    def _calculate_max_tokens(self, user_text: str) -> int:
        """Calculate max tokens based on query complexity."""
        base = self.max_tokens
        # Short queries need shorter responses
        if len(user_text) < 30:
            return min(base, 500)
        elif len(user_text) < 100:
            return min(base, 1000)
        return base

    # =========================================================================
    # LLM Provider Methods
    # =========================================================================

    async def _call_openai_compatible(
        self,
        user_input: conversation.ConversationInput,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
    ) -> str:
        """Call OpenAI-compatible API with streaming and tool support."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input.text},
        ]

        full_response = ""
        called_tools: set[str] = set()

        for iteration in range(5):  # Max 5 tool iterations
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
            tool_calls_buffer: list[dict] = []

            self._track_api_call("llm")

            try:
                stream = await self.client.chat.completions.create(**kwargs)

                try:
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
                finally:
                    await stream.close()

                # Process tool calls
                valid_tool_calls = [
                    tc for tc in tool_calls_buffer
                    if tc.get("id") and tc.get("function", {}).get("name")
                ]

                unique_tool_calls = []
                for tc in valid_tool_calls:
                    tool_key = f"{tc['function']['name']}:{tc['function']['arguments']}"
                    if tool_key not in called_tools:
                        called_tools.add(tool_key)
                        unique_tool_calls.append(tc)

                if unique_tool_calls:
                    _LOGGER.info("Processing %d tool call(s)", len(unique_tool_calls))

                    messages.append({
                        "role": "assistant",
                        "content": accumulated_content if accumulated_content else None,
                        "tool_calls": unique_tool_calls
                    })

                    # Execute tools in parallel
                    tool_tasks = []
                    for tool_call in unique_tool_calls:
                        tool_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}

                        _LOGGER.info("Tool call: %s(%s)", tool_name, arguments)
                        tool_tasks.append(self._execute_tool(tool_name, arguments, user_input))

                    tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                    for tool_call, result in zip(unique_tool_calls, tool_results):
                        if isinstance(result, Exception):
                            _LOGGER.error("Tool %s failed: %s", tool_call["function"]["name"], result)
                            result = {"error": str(result)}

                        # If tool returned response_text, use it directly to prevent LLM reformatting
                        if isinstance(result, dict) and "response_text" in result:
                            content = result["response_text"]
                        else:
                            content = json.dumps(result)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": content,
                        })

                    continue

                if accumulated_content:
                    return full_response

                break

            except Exception as e:
                _LOGGER.error("OpenAI API error: %s", e)
                return "Sorry, there was an error processing your request."

        return full_response if full_response else "I apologize, but I couldn't complete that request."

    async def _call_anthropic(
        self,
        user_input: conversation.ConversationInput,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
    ) -> str:
        """Call Anthropic Claude API."""
        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })

        messages = [{"role": "user", "content": user_input.text}]
        full_response = ""
        called_tools: set[str] = set()

        for iteration in range(5):
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

            self._track_api_call("llm")

            try:
                async with self._session.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=60,
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        _LOGGER.error("Anthropic API error: %s", error)
                        return "Sorry, there was an error with the AI service."

                    data = await response.json()

                text_content = ""
                tool_calls = []

                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "arguments": block.get("input", {}),
                        })

                if text_content:
                    full_response += text_content

                if tool_calls:
                    messages.append({"role": "assistant", "content": data.get("content", [])})

                    # Execute tools in parallel
                    unique_tool_calls = []
                    for tc in tool_calls:
                        tool_key = f"{tc['name']}:{tc.get('arguments', '')}"
                        if tool_key not in called_tools:
                            called_tools.add(tool_key)
                            unique_tool_calls.append(tc)
                            _LOGGER.info("Tool call: %s(%s)", tc["name"], tc["arguments"])

                    # Run all tool calls concurrently
                    tool_tasks = [
                        self._execute_tool(tc["name"], tc["arguments"], user_input)
                        for tc in unique_tool_calls
                    ]
                    results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                    tool_results = []
                    for tc, result in zip(unique_tool_calls, results):
                        if isinstance(result, Exception):
                            content = json.dumps({"error": str(result)})
                        elif isinstance(result, dict) and "response_text" in result:
                            content = result["response_text"]
                        else:
                            content = json.dumps(result)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": content
                        })

                    messages.append({"role": "user", "content": tool_results})
                    continue

                if full_response:
                    return full_response

                break

            except Exception as e:
                _LOGGER.error("Anthropic API error: %s", e)
                return "Sorry, there was an error processing your request."

        return full_response if full_response else "I apologize, but I couldn't complete that request."

    async def _call_google(
        self,
        user_input: conversation.ConversationInput,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
    ) -> str:
        """Call Google Gemini API."""
        # Convert tools to Gemini format
        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            function_declarations.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters", {"type": "object", "properties": {}})
            })

        contents = [
            {"role": "user", "parts": [{"text": f"System: {system_prompt}"}]},
            {"role": "model", "parts": [{"text": "Understood."}]},
            {"role": "user", "parts": [{"text": user_input.text}]},
        ]

        full_response = ""
        called_tools: set[str] = set()

        for iteration in range(5):
            payload = {
                "contents": contents,
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": self.temperature},
            }
            if function_declarations:
                payload["tools"] = [{"functionDeclarations": function_declarations}]

            url = f"{self.base_url}/models/{self.model}:generateContent"
            headers = {"x-goog-api-key": self.api_key}

            self._track_api_call("llm")

            try:
                async with self._session.post(url, json=payload, headers=headers, timeout=60) as response:
                    if response.status != 200:
                        error = await response.text()
                        _LOGGER.error("Google API error: %s", error)
                        return "Sorry, there was an error with the AI service."

                    data = await response.json()

                candidates = data.get("candidates", [])
                if not candidates:
                    break

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])

                text_content = ""
                tool_calls = []

                for part in parts:
                    if "text" in part:
                        text_content += part["text"]
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append({
                            "name": fc.get("name"),
                            "arguments": fc.get("args", {}),
                        })

                if text_content:
                    full_response += text_content

                if tool_calls:
                    contents.append({"role": "model", "parts": parts})

                    # Execute tools in parallel
                    unique_tool_calls = []
                    for tc in tool_calls:
                        tool_key = f"{tc['name']}:{tc.get('arguments', '')}"
                        if tool_key not in called_tools:
                            called_tools.add(tool_key)
                            unique_tool_calls.append(tc)
                            _LOGGER.info("Tool call: %s(%s)", tc["name"], tc["arguments"])

                    # Run all tool calls concurrently
                    tool_tasks = [
                        self._execute_tool(tc["name"], tc["arguments"], user_input)
                        for tc in unique_tool_calls
                    ]
                    results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                    function_responses = []
                    for tc, result in zip(unique_tool_calls, results):
                        if isinstance(result, Exception):
                            response_content = {"error": str(result)}
                        elif isinstance(result, dict) and "response_text" in result:
                            response_content = {"text": result["response_text"]}
                        else:
                            response_content = result
                        function_responses.append({
                            "functionResponse": {"name": tc["name"], "response": response_content}
                        })

                    contents.append({"role": "user", "parts": function_responses})
                    continue

                if full_response:
                    return full_response

                break

            except Exception as e:
                _LOGGER.error("Google API error: %s", e)
                return "Sorry, there was an error processing your request."

        return full_response if full_response else "I apologize, but I couldn't complete that request."

    # =========================================================================
    # Tool Execution
    # =========================================================================

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_input: conversation.ConversationInput,
    ) -> dict[str, Any]:
        """Execute a tool call."""
        try:
            # Get location defaults
            latitude = self.custom_latitude or self.hass.config.latitude
            longitude = self.custom_longitude or self.hass.config.longitude
            hass_tz = dt_util.get_time_zone(self.hass.config.time_zone)

            # Route to appropriate handler
            if tool_name == "get_current_datetime":
                now = datetime.now(hass_tz)
                return {
                    "date": now.strftime("%A, %B %d, %Y"),
                    "time": now.strftime("%I:%M %p"),
                    "timezone": self.hass.config.time_zone,
                }

            elif tool_name == "get_weather_forecast":
                return await weather_tool.get_weather_forecast(
                    arguments, self._session, self.openweathermap_api_key,
                    latitude, longitude, self._track_api_call
                )

            elif tool_name == "get_sports_info":
                return await sports_tool.get_sports_info(
                    arguments, self._session, hass_tz, self._track_api_call
                )

            elif tool_name == "get_ufc_info":
                return await sports_tool.get_ufc_info(
                    arguments, self._session, hass_tz, self._track_api_call
                )

            elif tool_name == "get_stock_price":
                return await stocks_tool.get_stock_price(
                    arguments, self._session, self._track_api_call
                )

            elif tool_name == "get_news":
                return await news_tool.get_news(
                    arguments, self._session, self.newsapi_key, hass_tz, self._track_api_call
                )

            elif tool_name == "find_nearby_places":
                return await places_tool.find_nearby_places(
                    arguments, self._session, self.google_places_api_key,
                    latitude, longitude, self._track_api_call
                )

            elif tool_name == "get_restaurant_recommendations":
                return await places_tool.get_restaurant_recommendations(
                    arguments, self._session, self.yelp_api_key,
                    latitude, longitude, self._track_api_call
                )

            elif tool_name == "calculate_age":
                return await wikipedia_tool.calculate_age(
                    arguments, self._session, self._track_api_call
                )

            elif tool_name == "get_wikipedia_summary":
                return await wikipedia_tool.get_wikipedia_summary(
                    arguments, self._session, self._track_api_call
                )

            elif tool_name == "get_calendar_events":
                return await calendar_tool.get_calendar_events(
                    arguments, self.hass, self.calendar_entities, hass_tz
                )

            elif tool_name == "check_camera":
                return await camera_tool.check_camera(
                    arguments, self.hass, None
                )

            elif tool_name == "quick_camera_check":
                return await camera_tool.quick_camera_check(
                    arguments, self.hass, None
                )

            elif tool_name == "control_thermostat":
                return await thermostat_tool.control_thermostat(
                    arguments, self.hass, self.thermostat_entity,
                    self.thermostat_temp_step, self.thermostat_min_temp,
                    self.thermostat_max_temp, self.format_temp
                )

            elif tool_name == "check_device_status":
                return await device_tool.check_device_status(
                    arguments, self.hass, self.device_aliases,
                    self._current_user_query, self.format_temp
                )

            elif tool_name == "get_device_history":
                return await device_tool.get_device_history(
                    arguments, self.hass, self.device_aliases,
                    hass_tz, self._current_user_query
                )

            elif tool_name == "control_device":
                return await device_tool.control_device(
                    arguments, self.hass, self.device_aliases
                )

            elif tool_name == "control_music":
                if self._music_controller:
                    return await self._music_controller.control_music(arguments)
                return {"error": "Music control not configured"}

            elif tool_name == "control_timer":
                return await timer_tool.control_timer(
                    arguments, self.hass,
                    device_id=user_input.device_id,
                    room_player_mapping=self.room_player_mapping
                )

            elif tool_name == "manage_list":
                return await lists_tool.manage_list(arguments, self.hass)

            elif tool_name == "create_reminder":
                return await reminders_tool.create_reminder(arguments, self.hass, hass_tz)

            elif tool_name == "get_reminders":
                return await reminders_tool.get_reminders(arguments, self.hass, hass_tz)

            # Fall back to script execution
            elif self.hass.services.has_service("script", tool_name):
                response = await self.hass.services.async_call(
                    "script", tool_name, arguments, blocking=True, return_response=True
                )
                if response:
                    script_entity = f"script.{tool_name}"
                    if isinstance(response, dict) and script_entity in response:
                        return response[script_entity]
                    return response
                return {"status": "success", "script": tool_name}

            else:
                _LOGGER.warning("Unknown tool '%s' called", tool_name)
                return {"success": True, "message": f"Custom function {tool_name} called"}

        except Exception as err:
            _LOGGER.error("Error executing tool %s: %s", tool_name, err, exc_info=True)
            return {"error": str(err)}

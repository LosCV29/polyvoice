"""Tool registry for PolyVoice.

Provides a centralized registry for tool handlers with automatic dispatch.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Coroutine, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.components import conversation

_LOGGER = logging.getLogger(__name__)

# Type for tool handlers
ToolHandler = Callable[..., Coroutine[Any, Any, dict[str, Any]]]

# SAFETY: Allowlisted domains for HA service calls (prevents dangerous calls)
ALLOWED_SERVICE_DOMAINS = frozenset({
    "light", "switch", "cover", "lock", "climate", "media_player",
    "fan", "vacuum", "scene", "script", "input_boolean", "input_number",
    "input_select", "input_text", "automation", "timer", "counter",
    "number", "select", "button", "siren", "humidifier", "notify",
})


class ToolRegistry:
    """Registry for tool handlers.

    This class manages tool handlers and provides a unified interface
    for executing tools by name.

    Example:
        registry = ToolRegistry(hass)
        registry.register("get_weather", weather_handler)
        result = await registry.execute("get_weather", {"location": "Miami"}, user_input)
    """

    def __init__(self, hass: HomeAssistant):
        """Initialize the registry."""
        self._hass = hass
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        """Register a tool handler.

        Args:
            name: Tool name (must match the tool definition name)
            handler: Async function that handles the tool call
        """
        self._handlers[name] = handler

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_input: conversation.ConversationInput,
        user_query: str = "",
    ) -> dict[str, Any]:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            user_input: The user's conversation input
            user_query: Original user query text (for context)

        Returns:
            Tool result dictionary
        """
        try:
            # Try registered handlers first
            if tool_name in self._handlers:
                handler = self._handlers[tool_name]
                return await handler(arguments, user_input, user_query)

            # Try HA services directly (domain.service format)
            if "." in tool_name:
                return await self._execute_service(tool_name, arguments)

            # Try as script
            if self._hass.services.has_service("script", tool_name):
                return await self._execute_script(tool_name, arguments)

            _LOGGER.warning("Unknown tool '%s' called with arguments: %s", tool_name, arguments)
            return {"success": True, "message": f"Custom function {tool_name} called", "arguments": arguments}

        except Exception as err:
            _LOGGER.error("Error executing tool %s: %s", tool_name, err, exc_info=True)
            return {"success": False, "error": str(err)}

    async def _execute_service(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a Home Assistant service call.

        Args:
            tool_name: Service in domain.service format
            arguments: Service data

        Returns:
            Result dictionary
        """
        # SECURITY: Validate tool_name format before splitting
        if not re.match(r'^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$', tool_name):
            _LOGGER.error("SECURITY: Blocked invalid service format: %s", tool_name[:100])
            return {"error": "Invalid service format"}

        domain, service = tool_name.split(".", 1)

        # SECURITY: Check for path traversal or injection
        if ".." in tool_name or "/" in tool_name or "\\" in tool_name:
            _LOGGER.error("SECURITY: Blocked suspicious service name: %s", tool_name[:100])
            return {"error": "Invalid service name"}

        # SAFETY: Check domain allowlist
        if domain not in ALLOWED_SERVICE_DOMAINS:
            _LOGGER.warning("Blocked service call to non-allowlisted domain: %s.%s", domain, service)
            return {"error": f"Service domain '{domain}' is not allowed for safety reasons."}

        service_response = await self._hass.services.async_call(
            domain, service, arguments, blocking=True, return_response=True
        )

        if service_response:
            return {"success": True, "result": service_response}

        return {"success": True, "message": f"Successfully executed {tool_name}"}

    async def _execute_script(self, script_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a Home Assistant script.

        Args:
            script_name: Script name (without script. prefix)
            arguments: Script variables

        Returns:
            Result dictionary
        """
        _LOGGER.info("Calling script: %s with args: %s", script_name, arguments)

        response = await self._hass.services.async_call(
            "script", script_name, arguments, blocking=True, return_response=True
        )

        _LOGGER.info("Script %s response: %s", script_name, response)

        if response is not None:
            script_entity = f"script.{script_name}"
            if isinstance(response, dict):
                if script_entity in response:
                    return response[script_entity]
                if response:
                    return response

        return {"status": "success", "script": script_name}

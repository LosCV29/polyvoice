"""Google Gemini provider implementation."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp

from .base import BaseLLMProvider

_LOGGER = logging.getLogger(__name__)


class GoogleProvider(BaseLLMProvider):
    """Provider for Google Gemini API."""

    def format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool format to Gemini format."""
        if not tools:
            return []

        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            function_declarations.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters", {"type": "object", "properties": {}})
            })
        return [{"functionDeclarations": function_declarations}]

    async def make_request(
        self,
        conversation_state: dict,
        tools: list[dict],
        max_tokens: int,
    ) -> dict:
        """Make request to Google Gemini API."""
        payload = {
            "contents": conversation_state["contents"],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": self._temperature,
            }
        }
        if tools:
            payload["tools"] = tools

        url = f"{self._base_url}/models/{self._model}:generateContent"
        headers = {"x-goog-api-key": self._api_key}

        async with self._session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                error = await response.text()
                _LOGGER.error("Google API error: %s", error)
                raise Exception(f"API error: {response.status}")

            return await response.json()

    def parse_response(self, response: dict) -> tuple[str, list[dict] | None]:
        """Parse Gemini response."""
        candidates = response.get("candidates", [])
        if not candidates:
            return "", None

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
                    "id": fc.get("name"),  # Gemini doesn't have IDs, use name
                    "name": fc.get("name"),
                    "arguments": fc.get("args", {}),
                })

        return text_content, tool_calls if tool_calls else None

    def _init_conversation(self, user_text: str, system_prompt: str) -> dict:
        """Initialize Gemini conversation state."""
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        return {"contents": contents}

    def _add_tool_results(
        self,
        conversation_state: dict,
        response: dict,
        tool_calls: list[dict],
        tool_results: list[dict],
    ) -> dict:
        """Add tool results to Gemini conversation."""
        # Add model's response with function calls
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            conversation_state["contents"].append({"role": "model", "parts": parts})

        # Add function responses
        function_responses = []
        for result in tool_results:
            function_responses.append({
                "functionResponse": {
                    "name": result["name"],
                    "response": result["result"]
                }
            })

        conversation_state["contents"].append({"role": "user", "parts": function_responses})

        return conversation_state

"""Anthropic Claude provider implementation."""
from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from .base import BaseLLMProvider

_LOGGER = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude API."""

    def format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
        return anthropic_tools

    async def make_request(
        self,
        conversation_state: dict,
        tools: list[dict],
        max_tokens: int,
    ) -> dict:
        """Make request to Anthropic API."""
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": conversation_state["system"],
            "messages": conversation_state["messages"],
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with self._session.post(
            f"{self._base_url}/v1/messages",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                error = await response.text()
                _LOGGER.error("Anthropic API error: %s", error)
                raise Exception(f"API error: {response.status}")

            return await response.json()

    def parse_response(self, response: dict) -> tuple[str, list[dict] | None]:
        """Parse Anthropic response."""
        text_content = ""
        tool_calls = []

        for block in response.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": block.get("input", {}),
                })

        return text_content, tool_calls if tool_calls else None

    def _init_conversation(self, user_text: str, system_prompt: str) -> dict:
        """Initialize Anthropic conversation state."""
        return {
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_text}],
        }

    def _add_tool_results(
        self,
        conversation_state: dict,
        response: dict,
        tool_calls: list[dict],
        tool_results: list[dict],
    ) -> dict:
        """Add tool results to Anthropic conversation."""
        # Add assistant's response with tool calls
        conversation_state["messages"].append({
            "role": "assistant",
            "content": response.get("content", [])
        })

        # Add tool results
        tool_result_content = []
        for result in tool_results:
            tool_result_content.append({
                "type": "tool_result",
                "tool_use_id": result["id"],
                "content": json.dumps(result["result"])
            })

        conversation_state["messages"].append({
            "role": "user",
            "content": tool_result_content
        })

        return conversation_state

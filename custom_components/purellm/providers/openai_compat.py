"""OpenAI-compatible provider implementation.

Supports: OpenAI, LM Studio, Groq, OpenRouter, Azure OpenAI, Ollama
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.components import conversation
    from openai import AsyncOpenAI

from .base import BaseLLMProvider, ToolExecutor

_LOGGER = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI-compatible APIs.

    This provider uses the openai Python library for streaming support
    and handles LM Studio, OpenAI, Groq, OpenRouter, Azure, and Ollama.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float,
        top_p: float,
        track_api_call: Callable[[str], None],
    ):
        """Initialize the OpenAI-compatible provider.

        Note: This provider uses the openai library client instead of aiohttp.
        """
        self._client = client
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._track_api_call = track_api_call

    async def call_with_tools(
        self,
        user_input: conversation.ConversationInput,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
        tool_executor: ToolExecutor,
    ) -> str:
        """Call the LLM with tool support using streaming."""
        import asyncio

        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user message
        messages.append({"role": "user", "content": user_input.text})

        full_response = ""
        called_tools: set[str] = set()

        for iteration in range(self.MAX_ITERATIONS):
            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": max_tokens,
                "top_p": self._top_p,
                "stream": True,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            accumulated_content = ""
            tool_calls_buffer: list[dict] = []

            self._track_api_call("llm")

            try:
                stream = await self._client.chat.completions.create(**kwargs)

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
                    # Ensure stream is closed to release connection back to pool
                    await stream.close()

                # Process valid tool calls
                valid_tool_calls = [
                    tc for tc in tool_calls_buffer
                    if tc.get("id") and tc.get("function", {}).get("name")
                ]

                # Filter duplicates
                unique_tool_calls = []
                for tc in valid_tool_calls:
                    tool_key = f"{tc['function']['name']}:{tc['function']['arguments']}"
                    if tool_key not in called_tools:
                        called_tools.add(tool_key)
                        unique_tool_calls.append(tc)
                    else:
                        _LOGGER.debug("Skipping duplicate tool call: %s", tc['function']['name'])

                if unique_tool_calls:
                    _LOGGER.info("Processing %d tool call(s)", len(unique_tool_calls))

                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": accumulated_content if accumulated_content else None,
                        "tool_calls": unique_tool_calls
                    })

                    # Execute all tools in parallel
                    tool_tasks = []
                    for tool_call in unique_tool_calls:
                        tool_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}

                        _LOGGER.info("Tool call: %s(%s)", tool_name, arguments)
                        tool_tasks.append(tool_executor(tool_name, arguments, user_input))

                    # Execute all tools simultaneously
                    tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                    # Add results to messages
                    for tool_call, result in zip(unique_tool_calls, tool_results):
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

                # No tool calls - return response
                if accumulated_content:
                    return full_response

                break

            except Exception as e:
                _LOGGER.error("OpenAI-compatible API exception: %s", e)
                return "Sorry, there was an error processing your request."

        return full_response if full_response else "I apologize, but I couldn't complete that request."

    # These methods are not used for OpenAI-compatible provider
    # since we override call_with_tools completely for streaming support

    def format_tools(self, tools: list[dict]) -> list[dict]:
        """OpenAI format is the native format."""
        return tools

    async def make_request(self, conversation_state: Any, tools: Any, max_tokens: int) -> Any:
        """Not used - see call_with_tools."""
        raise NotImplementedError("Use call_with_tools for streaming")

    def parse_response(self, response: Any) -> tuple[str, list[dict] | None]:
        """Not used - see call_with_tools."""
        raise NotImplementedError("Use call_with_tools for streaming")

    def _init_conversation(self, user_text: str, system_prompt: str) -> Any:
        """Not used - see call_with_tools."""
        raise NotImplementedError("Use call_with_tools for streaming")

    def _add_tool_results(
        self,
        conversation_state: Any,
        response: Any,
        tool_calls: list[dict],
        tool_results: list[dict],
    ) -> Any:
        """Not used - see call_with_tools."""
        raise NotImplementedError("Use call_with_tools for streaming")

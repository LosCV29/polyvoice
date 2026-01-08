"""Base class for LLM providers."""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.components import conversation
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Type alias for tool executor function
ToolExecutor = Callable[[str, dict[str, Any], "conversation.ConversationInput"], Coroutine[Any, Any, dict[str, Any]]]


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    This class handles the common logic for all providers:
    - Tool calling loop (max 5 iterations)
    - Parallel tool execution
    - Error handling

    Subclasses implement:
    - format_tools(): Convert OpenAI tool format to provider-specific format
    - make_request(): Make the actual API call
    - parse_response(): Extract text content and tool calls from response
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        top_p: float,
        track_api_call: Callable[[str], None],
    ):
        """Initialize the provider.

        Args:
            session: aiohttp session for making requests
            api_key: API key for authentication
            base_url: Base URL for the API
            model: Model name/ID to use
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            track_api_call: Callback to track API usage
        """
        self._session = session
        self._api_key = api_key
        self._base_url = base_url
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
        """Call the LLM with tool support.

        This method handles the complete tool calling loop:
        1. Send user input to LLM
        2. If LLM returns tool calls, execute them in parallel
        3. Send tool results back to LLM
        4. Repeat until LLM returns text response or max iterations reached

        Args:
            user_input: The user's conversation input
            tools: List of available tools in OpenAI format
            system_prompt: System prompt to use
            max_tokens: Maximum tokens for response
            tool_executor: Function to execute tool calls

        Returns:
            The final text response from the LLM
        """
        # Format tools for this provider
        provider_tools = self.format_tools(tools)

        # Initialize conversation state
        conversation_state = self._init_conversation(user_input.text, system_prompt)

        full_response = ""
        called_tools: set[str] = set()  # Track to prevent duplicate calls

        for iteration in range(self.MAX_ITERATIONS):
            self._track_api_call("llm")

            try:
                # Make the API request
                response = await self.make_request(
                    conversation_state,
                    provider_tools,
                    max_tokens,
                )

                # Parse the response
                text_content, tool_calls = self.parse_response(response)

                if text_content:
                    full_response += text_content

                if tool_calls:
                    # Filter duplicate tool calls
                    unique_calls = []
                    for tc in tool_calls:
                        tool_key = f"{tc['name']}:{tc.get('arguments', '')}"
                        if tool_key not in called_tools:
                            called_tools.add(tool_key)
                            unique_calls.append(tc)
                        else:
                            _LOGGER.debug("Skipping duplicate tool call: %s", tc['name'])

                    if unique_calls:
                        _LOGGER.info("Processing %d tool call(s)", len(unique_calls))

                        # Execute all tools in parallel
                        tool_tasks = [
                            tool_executor(tc["name"], tc.get("arguments", {}), user_input)
                            for tc in unique_calls
                        ]
                        results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                        # Process results
                        tool_results = []
                        for tc, result in zip(unique_calls, results):
                            if isinstance(result, Exception):
                                _LOGGER.error("Tool %s failed: %s", tc["name"], result)
                                result = {"error": str(result)}
                            tool_results.append({
                                "id": tc.get("id"),
                                "name": tc["name"],
                                "result": result,
                            })

                        # Update conversation state with tool calls and results
                        conversation_state = self._add_tool_results(
                            conversation_state, response, unique_calls, tool_results
                        )
                        continue

                # No tool calls - return the response
                if full_response:
                    return full_response

            except Exception as e:
                _LOGGER.error("%s API exception: %s", self.__class__.__name__, e)
                return "Sorry, there was an error processing your request."

        return full_response if full_response else "I couldn't complete that request."

    @abstractmethod
    def format_tools(self, tools: list[dict]) -> Any:
        """Convert OpenAI tool format to provider-specific format.

        Args:
            tools: Tools in OpenAI format

        Returns:
            Tools in provider-specific format
        """
        pass

    @abstractmethod
    async def make_request(
        self,
        conversation_state: Any,
        tools: Any,
        max_tokens: int,
    ) -> Any:
        """Make the API request.

        Args:
            conversation_state: Provider-specific conversation state
            tools: Tools in provider-specific format
            max_tokens: Maximum tokens for response

        Returns:
            Raw API response
        """
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> tuple[str, list[dict] | None]:
        """Parse the API response.

        Args:
            response: Raw API response

        Returns:
            Tuple of (text_content, tool_calls)
            tool_calls is a list of {"id": str, "name": str, "arguments": dict}
        """
        pass

    @abstractmethod
    def _init_conversation(self, user_text: str, system_prompt: str) -> Any:
        """Initialize conversation state.

        Args:
            user_text: The user's message
            system_prompt: System prompt

        Returns:
            Provider-specific conversation state
        """
        pass

    @abstractmethod
    def _add_tool_results(
        self,
        conversation_state: Any,
        response: Any,
        tool_calls: list[dict],
        tool_results: list[dict],
    ) -> Any:
        """Add tool results to conversation state.

        Args:
            conversation_state: Current conversation state
            response: The response that contained tool calls
            tool_calls: The tool calls that were made
            tool_results: Results from executing the tools

        Returns:
            Updated conversation state
        """
        pass

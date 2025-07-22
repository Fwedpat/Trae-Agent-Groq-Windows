# Fwedpat Groq Client addition.

"""Groq API client wrapper with tool integration."""

import os
import json
import random
import time
import openai
from openai.types.chat import (
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition
from typing import override

from ..tools.base import Tool, ToolCall
from ..utils.config import ModelParameters
from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage


class GroqClient(BaseLLMClient):
    """Groq API client wrapper with tool integration."""

    def __init__(self, model_parameters: ModelParameters):
        """Initialize the GroqClient."""
        super().__init__(model_parameters)
        
        if self.api_key == "":
            self.api_key: str = os.getenv("GROQ_API_KEY", "")

        if self.api_key == "":
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY in environment variables or config file."
            )
        
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key, base_url="https://api.groq.com/openai/v1",
        )

        self.message_history: list[ChatCompletionMessageParam] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        model_parameters: ModelParameters,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to Groq with optional tool support."""
        groq_messages = self.parse_messages(messages)
        if reuse_history:
            self.message_history = self.message_history + groq_messages
        else:
            self.message_history = groq_messages

        tool_schemas = None
        if tools:
            tool_schemas = [
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name=tool.get_name(),
                        description=tool.get_description(),
                        parameters=tool.get_input_schema(),
                    ),
                    type="function",
                )
                for tool in tools
            ]

        
        response = None
        error_message = ""
        for i in range(model_parameters.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_parameters.model,
                    messages=self.message_history,
                    tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
                    temperature=model_parameters.temperature,
                    top_p=model_parameters.top_p,
                    max_tokens=model_parameters.max_tokens,
                    n=1,
                )
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {str(e)}\n"
                # Randomly sleep for 3-30 seconds
                time.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(
                f"Failed to get response from Groq after max retries: {error_message}"
            )

        choice = response.choices[0]

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls: list[ToolCall] | None = []
            for tool_call in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,
                        call_id=tool_call.id,
                        arguments=(
                            json.loads(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else {}
                        ),
                    )
                )

        llm_response = LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage=(
                LLMUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )
                if response.usage
                else None
            ),
        )

        # update message history
        if llm_response.tool_calls:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=llm_response.content,
                    tool_calls=[
                        ChatCompletionMessageToolCallParam(
                            id=tool_call.call_id,
                            function=Function(
                                name=tool_call.name,
                                arguments=json.dumps(tool_call.arguments),
                            ),
                            type="function",
                        )
                        for tool_call in llm_response.tool_calls
                    ],
                )
            )
        elif llm_response.content:
            self.message_history.append(
                ChatCompletionAssistantMessageParam(
                    content=llm_response.content, role="assistant"
                )
            )

        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="Groq",
                model=model_parameters.model,
                tools=tools,
            )

        return llm_response


    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""
        # Most modern models on Groq support tool calling
        # We'll be conservative and check for known capable models
        tool_capable_patterns = [
            "kimi-k2-instruct",
            "llama-4-maverick",
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-versatile",
            "gemma2",
        ]
        return any(
            pattern in model_parameters.model.lower()
            for pattern in tool_capable_patterns
        )

    
    def parse_messages(
        self, messages: list[LLMMessage]
    ) -> list[ChatCompletionMessageParam]:
        groq_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if msg.tool_call:
                groq_messages.append(
                    ChatCompletionFunctionMessageParam(
                        content=json.dumps(
                            {
                                "name": msg.tool_call.name,
                                "arguments": msg.tool_call.arguments,
                            }
                        ),
                        role="function",
                        name=msg.tool_call.name,
                    )
                )
            elif msg.tool_result:
                result: str = ""
                if msg.tool_result.result:
                    result = result + msg.tool_result.result + "\n"
                if msg.tool_result.error:
                    result += "Tool call failed with error:\n"
                    result += msg.tool_result.error
                result = result.strip()

                groq_messages.append(
                    ChatCompletionToolMessageParam(
                        content=result,
                        role="tool",
                        tool_call_id=msg.tool_result.call_id,
                    )
                )
            elif msg.role == "system":
                if not msg.content:
                    raise ValueError("System message content is required")
                groq_messages.append(
                    ChatCompletionSystemMessageParam(content=msg.content, role="system")
                )
            elif msg.role == "user":
                if not msg.content:
                    raise ValueError("User message content is required")
                groq_messages.append(
                    ChatCompletionUserMessageParam(content=msg.content, role="user")
                )
            elif msg.role == "assistant":
                if not msg.content:
                    raise ValueError("Assistant message content is required")
                groq_messages.append(
                    ChatCompletionAssistantMessageParam(
                        content=msg.content, role="assistant"
                    )
                )
            else:
                raise ValueError(f"Invalid message role: {msg.role}")
        return groq_messages
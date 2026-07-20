"""LLM-powered chat endpoint for natural-language survival predictions.

# LESSON POINT: LLMOps Interaction Layer
# This module adds a language model layer on top of the existing prediction API.
# The LLM does NOT make predictions itself. Instead, it extracts structured fields
# from natural language and calls the same predict_survival tool that the REST API
# uses. This separation keeps the deterministic model as the source of truth while
# the LLM handles the conversational interface.
#
# The system prompt injects a model card so the LLM can communicate limitations
# without fabricating them. Tool calls, latency, and token usage are traced for
# LLMOps observability (separate from the sklearn metrics tracked in MLflow).
"""

import json
import logging
import os
import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .llm_traces import LLMTrace, trace_store
from .model_card import MODEL_CARD
from .schemas import PassengerInput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
# LESSON POINT: Provider-Agnostic LLM Configuration
# By reading LLM_MODEL, LLM_API_KEY, and LLM_BASE_URL from environment variables
# and using the OpenAI-compatible API format, this code works with OpenAI, Azure,
# Anthropic (via proxy), Ollama, and any provider that speaks the same protocol.
# When LLM_API_KEY is not set, the chat endpoint falls back to a mock mode that
# simulates tool calls for development and CI without incurring API costs.

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")  # None = use default OpenAI base


# ---------------------------------------------------------------------------
#  Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")


class ToolCallInfo(BaseModel):
    tool_name: str
    arguments: dict
    result: dict | None = None


class ChatResponse(BaseModel):
    reply: str
    prediction: dict | None = None
    model_version: str | None = None
    tool_calls: list[ToolCallInfo] = []
    trace_id: str | None = None


# ---------------------------------------------------------------------------
#  Tool definition (OpenAI function-calling format)
# ---------------------------------------------------------------------------

PREDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_survival",
        "description": (
            "Predict whether a Titanic passenger would survive based on their attributes. "
            "Always use this tool for numerical predictions. Never guess or invent a prediction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "Pclass": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "Ticket class: 1=1st, 2=2nd, 3=3rd",
                },
                "Sex": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "Passenger sex",
                },
                "Age": {
                    "type": "number",
                    "description": "Age in years (0-120). Omit if unknown.",
                },
                "SibSp": {
                    "type": "integer",
                    "description": "Number of siblings or spouses aboard. Default 0.",
                },
                "Parch": {
                    "type": "integer",
                    "description": "Number of parents or children aboard. Default 0.",
                },
                "Fare": {
                    "type": "number",
                    "description": "Ticket fare in pounds.",
                },
                "Embarked": {
                    "type": "string",
                    "enum": ["S", "C", "Q"],
                    "description": "Port of embarkation: S=Southampton, C=Cherbourg, Q=Queenstown. Default S.",
                },
            },
            "required": ["Pclass", "Sex", "Fare"],
        },
    },
}

SYSTEM_PROMPT = f"""You are the Titanic Decision Copilot, a concise assistant for the Titanic survival prediction demo.

RULES:
- For any numerical prediction, you MUST call the predict_survival tool. Never invent or guess a survival probability.
- Never make causal claims. Do not say a feature "caused" survival. Say "the model associates..." or "historically, passengers with X had higher survival rates."
- If the user does not provide enough information to fill the required tool fields (Pclass, Sex, Fare), ask a short follow-up question. Do not guess missing values.
- Always mention that this is a historical-dataset demonstration, not a real-world decision system, at least once per conversation.
- When discussing results, mention relevant limitations from the model card below.
- Keep responses concise and factual. Do not use emojis or em dashes.

MODEL CARD:
{MODEL_CARD}
"""


# ---------------------------------------------------------------------------
#  Core logic
# ---------------------------------------------------------------------------


def _execute_tool_call(tool_name: str, arguments: dict, model_manager) -> dict:
    """Execute a tool call and return the result dict."""
    if tool_name != "predict_survival":
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        passenger = PassengerInput(
            Pclass=arguments["Pclass"],
            Sex=arguments["Sex"],
            Age=arguments.get("Age"),
            SibSp=arguments.get("SibSp", 0),
            Parch=arguments.get("Parch", 0),
            Fare=arguments["Fare"],
            Embarked=arguments.get("Embarked", "S"),
        )
        survived, probability = model_manager.predict(passenger)
        return {
            "survived": survived,
            "probability": round(probability, 4),
            "model_version": model_manager.model_version or "unknown",
        }
    except Exception as e:
        return {"error": str(e)}


def _mock_chat(message: str, model_manager) -> ChatResponse:
    """Mock mode: parse the message for obvious passenger fields and call predict directly.

    This lets the frontend work in dev/CI without an LLM API key.
    """
    trace = LLMTrace(
        timestamp=datetime.now(UTC).isoformat(),
        model="mock",
        prompt=message,
    )
    start = time.monotonic()

    # Try a basic prediction if the model is loaded
    if model_manager.is_loaded:
        # Default passenger for mock mode
        result = _execute_tool_call(
            "predict_survival",
            {
                "Pclass": 2,
                "Sex": "female",
                "Age": 29,
                "Fare": 50.0,
                "Embarked": "S",
            },
            model_manager,
        )

        tool_info = ToolCallInfo(
            tool_name="predict_survival",
            arguments={"Pclass": 2, "Sex": "female", "Age": 29, "Fare": 50.0, "Embarked": "S"},
            result=result,
        )

        reply = (
            f"[Mock mode -- no LLM_API_KEY configured]\n\n"
            f"Running a default prediction: a 29-year-old woman in 2nd class.\n\n"
            f"Prediction: {'Survived' if result.get('survived') else 'Did not survive'} "
            f"(probability: {result.get('probability', 0):.1%}).\n\n"
            f"To enable natural-language chat, set the LLM_API_KEY environment variable."
        )

        trace.response = reply
        trace.tool_calls = [{"name": "predict_survival", "arguments": tool_info.arguments, "result": result}]
        trace.latency_ms = round((time.monotonic() - start) * 1000, 1)
        trace_store.add(trace)

        return ChatResponse(
            reply=reply,
            prediction=result,
            model_version=result.get("model_version"),
            tool_calls=[tool_info],
            trace_id=trace.trace_id,
        )
    else:
        reply = (
            "[Mock mode -- no LLM_API_KEY configured]\n\n"
            "No model is currently loaded. Train a model first, then try again.\n\n"
            "To enable natural-language chat, set the LLM_API_KEY environment variable."
        )
        trace.response = reply
        trace.latency_ms = round((time.monotonic() - start) * 1000, 1)
        trace_store.add(trace)
        return ChatResponse(reply=reply, trace_id=trace.trace_id)


async def handle_chat(message: str, model_manager) -> ChatResponse:
    """Process a chat message, calling the LLM or falling back to mock mode."""

    # Mock mode when no API key is set
    if not LLM_API_KEY:
        return _mock_chat(message, model_manager)

    # Real LLM mode
    trace = LLMTrace(
        timestamp=datetime.now(UTC).isoformat(),
        model=LLM_MODEL,
        prompt=message,
    )
    start = time.monotonic()

    try:
        from openai import AsyncOpenAI

        client_kwargs = {"api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            client_kwargs["base_url"] = LLM_BASE_URL

        client = AsyncOpenAI(**client_kwargs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        # First call: let the LLM decide whether to use the tool
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=[PREDICT_TOOL],
            tool_choice="auto",
        )

        choice = response.choices[0]
        tool_calls_info = []
        prediction_result = None
        model_version = None

        # If the LLM wants to call a tool, execute it and send the result back
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # Append the assistant message with tool calls
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = _execute_tool_call(tc.function.name, args, model_manager)

                tool_calls_info.append(
                    ToolCallInfo(
                        tool_name=tc.function.name,
                        arguments=args,
                        result=result,
                    )
                )

                if tc.function.name == "predict_survival" and "error" not in result:
                    prediction_result = result
                    model_version = result.get("model_version")

                trace.tool_calls.append(
                    {
                        "name": tc.function.name,
                        "arguments": args,
                        "result": result,
                    }
                )

                # Send tool result back to the LLM
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

            # Second call: get the final answer with the tool result
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
            )
            choice = response.choices[0]

        reply = choice.message.content or ""

        # Record usage
        if response.usage:
            trace.prompt_tokens = response.usage.prompt_tokens
            trace.completion_tokens = response.usage.completion_tokens
            trace.total_tokens = response.usage.total_tokens

        trace.response = reply
        trace.latency_ms = round((time.monotonic() - start) * 1000, 1)
        trace_store.add(trace)

        return ChatResponse(
            reply=reply,
            prediction=prediction_result,
            model_version=model_version,
            tool_calls=tool_calls_info,
            trace_id=trace.trace_id,
        )

    except Exception as e:
        trace.error = str(e)
        trace.latency_ms = round((time.monotonic() - start) * 1000, 1)
        trace_store.add(trace)
        logger.error("Chat error: %s", e)

        return ChatResponse(
            reply=f"An error occurred while processing your request: {e}",
            trace_id=trace.trace_id,
        )

"""OCI + OpenAI Agents runtime for Streamlit."""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable

from agents import (
    Agent,
    OpenAIResponsesModel,
    Runner,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types.responses import ResponseTextDeltaEvent
from rich.logging import RichHandler
from rich.markup import escape

load_dotenv()
set_tracing_disabled(disabled=True)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        omit_repeated_times=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("AGENT_LOG_LEVEL", "INFO").upper())
logger.propagate = False

OPENAI_TIMEOUT_SECONDS = 90.0
STAGE_STYLES = {
    "runtime": "cyan",
    "conversation": "magenta",
    "chat": "green",
    "mcp": "blue",
}

ASSISTANT_INSTRUCTIONS = (
    "Voce e um agente de suporte ao cliente. Ajude o cliente com perguntas e problemas. "
    "Responda de forma curta, objetiva e clara."
)


AppContext = Any


class AgentConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str
    base_url: str
    model_id: str
    project: str


@dataclass
class RuntimeResources:
    config: RuntimeConfig
    client: AsyncOpenAI
    model: OpenAIResponsesModel
    assistant_agent: Agent[AppContext]


_runtime: RuntimeResources | None = None


def _conversation_label(conversation_id: str) -> str:
    if len(conversation_id) <= 16:
        return conversation_id
    return f"...{conversation_id[-12:]}"


def _log_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, bool):
        return "sim" if value else "nao"
    text = str(value)
    if len(text) > 96:
        text = f"{text[:93].rstrip()}..."
    if any(char.isspace() for char in text):
        text = repr(text)
    return escape(text)


def _log(level: int, stage: str, event: str, **fields: Any) -> None:
    style = STAGE_STYLES.get(stage, "white")
    prefix = f"[bold {style}][{stage.upper()}][/bold {style}] {event}"
    if fields:
        details = " ".join(f"{key}={_log_value(value)}" for key, value in fields.items())
        prefix = f"{prefix} {details}"
    logger.log(level, prefix)


def _input_summary(value: str) -> str:
    preview = " ".join(value.split())
    if len(preview) > 24:
        preview = f"{preview[:24].rstrip()}..."
    return f"{len(value)}:{preview!r}"


def _event_delta(event: Any) -> str:
    if isinstance(event, ResponseTextDeltaEvent):
        return event.delta or ""

    data = getattr(event, "data", None)
    if isinstance(data, ResponseTextDeltaEvent):
        return data.delta or ""
    if getattr(event, "type", None) == "response.output_text.delta":
        return getattr(event, "delta", "") or ""
    if getattr(data, "type", None) == "response.output_text.delta":
        return getattr(data, "delta", "") or ""
    return ""


def _settings() -> RuntimeConfig:
    config = RuntimeConfig(
        api_key=os.getenv("OCI_API_KEY", "").strip(),
        base_url=os.getenv("OCI_BASE_URL", "").strip().rstrip("/"),
        model_id=os.getenv("OCI_MODEL_ID", "").strip(),
        project=os.getenv("OCI_PROJECT", "").strip(),
    )
    missing = []
    if not config.api_key:
        missing.append("OCI_API_KEY")
    if not config.base_url:
        missing.append("OCI_BASE_URL")
    if not config.model_id:
        missing.append("OCI_MODEL_ID")
    if not config.project:
        missing.append("OCI_PROJECT")
    if missing:
        raise AgentConfigError(f"Variaveis ausentes: {', '.join(missing)}")
    return config


def _build_runtime(config: RuntimeConfig) -> RuntimeResources:
    _log(
        logging.INFO,
        "runtime",
        "init",
        base=config.base_url,
        model=config.model_id,
        project=config.project,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        project=config.project,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    model = OpenAIResponsesModel(
        model=config.model_id,
        openai_client=client,
    )
    assistant_agent = Agent[AppContext](
        name="Atendimento",
        instructions=ASSISTANT_INSTRUCTIONS,
        model=model,
    )
    return RuntimeResources(
        config=config,
        client=client,
        model=model,
        assistant_agent=assistant_agent,
    )


def _get_runtime() -> RuntimeResources:
    global _runtime
    config = _settings()
    if _runtime is None or _runtime.config != config:
        _runtime = _build_runtime(config)
    return _runtime


def _assistant_agent() -> Agent[AppContext]:
    return _get_runtime().assistant_agent


def _tavily_mcp_url() -> str:
    return os.getenv("TAVILY_MCP_URL", "").strip()


def _tavily_search_url() -> str:
    return os.getenv("TAVILY_SEARCH_URL", "").strip()


def _build_tavily_mcp_server(tavily_url: str) -> MCPServerStreamableHttp:
    return MCPServerStreamableHttp(
        name="Tavily MCP Streamable HTTP Server",
        params={
            "url": tavily_url,
            "timeout": 15,
            "sse_read_timeout": 300,
            "ignore_initialized_notification_failure": True,
        },
        cache_tools_list=True,
        use_structured_content=True,
        max_retry_attempts=2,
        retry_backoff_seconds_base=2.0,
        client_session_timeout_seconds=15,
    )


def _assistant_agent_with_tavily(mcp_server: MCPServerStreamableHttp) -> Agent[AppContext]:
    search_url = _tavily_search_url()
    source_instruction = (
        f" Priorize buscas e referencias em {search_url}."
        if search_url
        else ""
    )
    return Agent[AppContext](
        name="Atendimento",
        instructions=(
            ASSISTANT_INSTRUCTIONS
            + " Use as ferramentas MCP do Tavily para responder quando a pergunta exigir "
            "informacao atual ou pesquisa externa."
            + source_instruction
        ),
        model=_get_runtime().model,
        mcp_servers=[mcp_server],
    )


async def _create_conversation(memory_subject_id: str) -> str:
    runtime = _get_runtime()
    conversation = await runtime.client.conversations.create(
        metadata={"memory_subject_id": memory_subject_id}
    )
    conversation_id = str(conversation.id)
    _log(
        logging.INFO,
        "conversation",
        "created",
        cid=_conversation_label(conversation_id),
        subject=memory_subject_id,
    )
    return conversation_id


def create_agent_conversation(memory_subject_id: str) -> str:
    with asyncio.Runner() as runner:
        return runner.run(_create_conversation(memory_subject_id))


def stream_agent_reply(
    user_prompt: str,
    conversation_id: str,
    use_tavily: bool = False,
    turn_id: str | None = None,
) -> Iterable[str]:
    turn = turn_id or "-"
    _log(
        logging.INFO,
        "chat",
        "start",
        cid=_conversation_label(conversation_id),
        turn=turn,
        inp=_input_summary(user_prompt),
    )

    async def _stream() -> AsyncIterator[str]:
        emitted_delta = False
        chunk_count = 0
        total_chars = 0
        started_at = asyncio.get_running_loop().time()
        first_token_at: float | None = None

        async def _stream_events(stream: Any) -> AsyncIterator[str]:
            nonlocal emitted_delta, chunk_count, total_chars, first_token_at
            async for event in stream.stream_events():
                delta = _event_delta(event)
                if not delta:
                    continue

                emitted_delta = True
                chunk_count += 1
                total_chars += len(delta)
                if first_token_at is None:
                    first_token_at = asyncio.get_running_loop().time()
                    _log(
                        logging.DEBUG,
                        "chat",
                        "first_token",
                        cid=_conversation_label(conversation_id),
                        turn=turn,
                        t=first_token_at - started_at,
                    )
                yield delta

            if not emitted_delta:
                final_output = getattr(stream, "final_output", None)
                if final_output is not None:
                    final_text = str(final_output).strip()
                    if final_text:
                        _log(
                            logging.DEBUG,
                            "chat",
                            "final_output_fallback",
                            cid=_conversation_label(conversation_id),
                            turn=turn,
                            t=asyncio.get_running_loop().time() - started_at,
                            chars=len(final_text),
                        )
                        total_chars = len(final_text)
                        yield final_text

        mcp_server: MCPServerStreamableHttp | None = None
        if use_tavily:
            tavily_url = _tavily_mcp_url()
            if not tavily_url:
                raise AgentConfigError("Variaveis ausentes: TAVILY_MCP_URL")
            mcp_server = _build_tavily_mcp_server(tavily_url)
            _log(logging.INFO, "mcp", "tavily.enabled", cid=_conversation_label(conversation_id))

        if mcp_server is not None:
            async with mcp_server:
                stream = Runner.run_streamed(
                    _assistant_agent_with_tavily(mcp_server),
                    input=user_prompt,
                    conversation_id=conversation_id,
                )
                async for delta in _stream_events(stream):
                    yield delta
        else:
            stream = Runner.run_streamed(
                _assistant_agent(),
                input=user_prompt,
                conversation_id=conversation_id,
            )
            async for delta in _stream_events(stream):
                yield delta

        _log(
            logging.INFO,
            "chat",
            "done",
            cid=_conversation_label(conversation_id),
            turn=turn,
            t=asyncio.get_running_loop().time() - started_at,
            ch=chunk_count,
            chars=total_chars,
        )

    with asyncio.Runner() as runner:
        iterator = _stream()
        try:
            while True:
                try:
                    chunk = runner.run(anext(iterator))
                except StopAsyncIteration:
                    break
                if chunk:
                    yield chunk
        except RateLimitError as error:
            _log(
                logging.WARNING,
                "chat",
                "rate_limit_error",
                cid=_conversation_label(conversation_id),
                turn=turn,
                err=error,
            )
            raise
        except APIStatusError as error:
            status = getattr(error, "status_code", None)
            if status in (401, 403):
                _log(
                    logging.ERROR,
                    "chat",
                    "auth_error",
                    cid=_conversation_label(conversation_id),
                    turn=turn,
                    st=status,
                )
            elif status == 429:
                _log(
                    logging.WARNING,
                    "chat",
                    "rate_limited",
                    cid=_conversation_label(conversation_id),
                    turn=turn,
                )
            else:
                _log(
                    logging.WARNING,
                    "chat",
                    "status_error",
                    cid=_conversation_label(conversation_id),
                    turn=turn,
                    st=status,
                    err=error,
                )
            raise
        except (APIConnectionError, APITimeoutError) as error:
            _log(
                logging.WARNING,
                "chat",
                "connection_error",
                cid=_conversation_label(conversation_id),
                turn=turn,
                err=error,
            )
            raise
        except Exception as error:  # noqa: BLE001
            _log(
                logging.WARNING,
                "chat",
                "unexpected_error",
                cid=_conversation_label(conversation_id),
                turn=turn,
                typ=type(error).__name__,
                err=error,
            )
            raise

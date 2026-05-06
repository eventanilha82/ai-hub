import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch

from httpx import Request, Response
from openai import APIConnectionError, APIStatusError, RateLimitError

import agent


class FakeStream:
    def __init__(self, events, final_output=None) -> None:
        self._events = events
        self.final_output = final_output

    async def stream_events(self):
        for event in self._events:
            yield event


class FakeCompletedStream:
    def __init__(self) -> None:
        self.final_output = None

    async def stream_events(self):
        yield SimpleNamespace(type="response.output_text.delta", delta="Ola")
        yield SimpleNamespace(type="raw_response_event", data=SimpleNamespace(type="response.completed"))


class BrokenStream:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.final_output = None

    async def stream_events(self):
        raise self._error
        yield


class FakeMCPServer:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        self.tool_calls = []

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.exited = True

    async def call_tool(self, tool_name, arguments):
        self.tool_calls.append((tool_name, arguments))
        return SimpleNamespace(
            structuredContent={
                "results": [
                    {
                        "title": "Filhotes de cachorro - Petz",
                        "url": "https://www.petz.com.br/blog/filhotes-de-cachorro/",
                        "content": "Cuidados basicos com filhotes.",
                    }
                ]
            }
        )


class AgentUnitTests(TestCase):
    def setUp(self) -> None:
        agent._runtime = None

    def test_settings_raise_when_required_env_is_missing(self) -> None:
        with patch.dict(
            os.environ,
            {"OCI_API_KEY": "", "OCI_BASE_URL": "", "OCI_MODEL_ID": "", "OCI_PROJECT": ""},
            clear=False,
        ):
            with self.assertRaises(agent.AgentConfigError):
                agent._settings()

    def test_runtime_reuses_same_config_and_rotates_on_change(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OCI_API_KEY": "k1",
                "OCI_BASE_URL": "https://example-1.test/v1",
                "OCI_MODEL_ID": "model-1",
                "OCI_PROJECT": "project-1",
            },
            clear=False,
        ):
            runtime_a = agent._get_runtime()
            runtime_b = agent._get_runtime()

        with patch.dict(
            os.environ,
            {
                "OCI_API_KEY": "k2",
                "OCI_BASE_URL": "https://example-2.test/v1",
                "OCI_MODEL_ID": "model-2",
                "OCI_PROJECT": "project-2",
            },
            clear=False,
        ):
            runtime_c = agent._get_runtime()

        self.assertIs(runtime_a, runtime_b)
        self.assertIsNot(runtime_a, runtime_c)

    def test_build_runtime_configures_openai_client_with_project(self) -> None:
        config = agent.RuntimeConfig(
            api_key="k1",
            base_url="https://example.test/v1",
            model_id="model-1",
            project="project-1",
        )

        with patch.object(agent, "AsyncOpenAI", return_value=object()) as async_openai, patch.object(
            agent, "OpenAIResponsesModel", return_value=object()
        ), patch.object(agent, "Agent", return_value=Mock()):
            agent._build_runtime(config)

        async_openai.assert_called_once_with(
            api_key="k1",
            base_url="https://example.test/v1",
            project="project-1",
            timeout=agent.OPENAI_TIMEOUT_SECONDS,
        )

    def test_create_agent_conversation_sets_memory_subject_metadata(self) -> None:
        runtime = Mock()
        runtime.config = agent.RuntimeConfig(
            api_key="k1",
            base_url="https://example.test/openai/v1",
            model_id="model-1",
            project="project-1",
        )
        runtime.client.conversations.create = AsyncMock(return_value=SimpleNamespace(id="conv-123"))

        with patch.object(agent, "_get_runtime", return_value=runtime):
            conversation_id = agent.create_agent_conversation("maria")

        self.assertEqual(conversation_id, "conv-123")
        runtime.client.conversations.create.assert_awaited_once_with(
            metadata={"memory_subject_id": "maria"}
        )

    def test_stream_agent_reply_yields_incremental_chunks(self) -> None:
        fake_stream = FakeStream(
            [
                SimpleNamespace(type="response.output_text.delta", delta="Ola"),
                SimpleNamespace(type="response.output_text.delta", delta=" mundo"),
            ]
        )

        with patch.object(agent, "_assistant_agent", return_value=object()), patch.object(
            agent.Runner, "run_streamed", return_value=fake_stream
        ) as run_streamed:
            chunks = list(agent.stream_agent_reply("oi", "conv-1"))

        self.assertEqual(chunks, ["Ola", " mundo"])
        self.assertEqual(run_streamed.call_args.kwargs["conversation_id"], "conv-1")
        self.assertNotIn("context", run_streamed.call_args.kwargs)

    def test_stream_agent_reply_uses_tavily_mcp_when_enabled(self) -> None:
        fake_stream = FakeStream([SimpleNamespace(type="response.output_text.delta", delta="Ola")])
        fake_mcp = FakeMCPServer()

        runtime = SimpleNamespace(model="main-model-test", assistant_agent=object())
        with patch.dict(
            os.environ,
            {
                "TAVILY_MCP_URL": "https://mcp.example.test/mcp",
                "TAVILY_SEARCH_URL": "https://www.petz.com.br/blog/",
            },
            clear=False,
        ), patch.object(agent, "TimedTavilyMCPServer", return_value=fake_mcp) as mcp_server, patch.object(
            agent, "_get_runtime", return_value=runtime
        ), patch.object(
            agent.Runner, "run_streamed", return_value=fake_stream
        ) as run_streamed:
            chunks = list(agent.stream_agent_reply("oi", "conv-1", use_tavily=True))

        self.assertEqual(chunks, ["Ola"])
        self.assertTrue(fake_mcp.entered)
        self.assertTrue(fake_mcp.exited)
        mcp_server.assert_called_once()
        self.assertEqual(mcp_server.call_args.kwargs["name"], "Tavily MCP Streamable HTTP Server")
        self.assertEqual(mcp_server.call_args.kwargs["params"]["url"], "https://mcp.example.test/mcp")
        self.assertEqual(mcp_server.call_args.kwargs["max_retry_attempts"], 2)
        self.assertEqual(mcp_server.call_args.kwargs["retry_backoff_seconds_base"], 2.0)
        self.assertEqual(mcp_server.call_args.kwargs["client_session_timeout_seconds"], 15)
        self.assertTrue(mcp_server.call_args.kwargs["use_structured_content"])
        self.assertEqual(mcp_server.call_args.kwargs["conversation_id"], "conv-1")
        self.assertEqual(mcp_server.call_args.kwargs["turn"], "-")
        streamed_agent = run_streamed.call_args.args[0]
        self.assertEqual(streamed_agent.mcp_servers, [fake_mcp])
        self.assertEqual(streamed_agent.tools, [])
        self.assertIs(streamed_agent.model, runtime.model)
        self.assertEqual(run_streamed.call_args.kwargs["input"], "oi")
        self.assertEqual(run_streamed.call_args.kwargs["conversation_id"], "conv-1")
        self.assertEqual(fake_mcp.tool_calls, [])

    def test_stream_agent_reply_requires_tavily_mcp_url_when_enabled(self) -> None:
        with patch.dict(os.environ, {"TAVILY_MCP_URL": ""}, clear=False):
            with self.assertRaises(agent.AgentConfigError):
                list(agent.stream_agent_reply("oi", "conv-1", use_tavily=True))

    def test_stream_agent_reply_uses_final_output_when_no_deltas_arrive(self) -> None:
        fake_stream = FakeStream([], final_output="Resposta final")

        with patch.object(agent, "_assistant_agent", return_value=object()), patch.object(
            agent.Runner, "run_streamed", return_value=fake_stream
        ):
            chunks = list(agent.stream_agent_reply("oi", "conv-2"))

        self.assertEqual(chunks, ["Resposta final"])

    def test_stream_agent_reply_finishes_cleanly_after_response_completed(self) -> None:
        fake_stream = FakeCompletedStream()

        with patch.object(agent, "_assistant_agent", return_value=object()), patch.object(
            agent.Runner, "run_streamed", return_value=fake_stream
        ):
            chunks = list(agent.stream_agent_reply("oi", "conv-terminal"))

        self.assertEqual(chunks, ["Ola"])

    def test_stream_agent_reply_reraises_auth_status_error(self) -> None:
        error = APIStatusError(
            "unauthorized",
            response=Response(401, request=Request("POST", "https://example.test")),
            body=None,
        )

        with patch.object(agent, "_assistant_agent", return_value=object()), patch.object(
            agent.Runner, "run_streamed", return_value=BrokenStream(error)
        ), patch.object(agent, "_log") as log:
            with self.assertRaises(APIStatusError):
                list(agent.stream_agent_reply("oi", "conv-auth"))

        self.assertTrue(any(call.args[2] == "auth_error" for call in log.call_args_list))

    def test_stream_agent_reply_reraises_rate_limited_status_error(self) -> None:
        error = APIStatusError(
            "too many requests",
            response=Response(429, request=Request("POST", "https://example.test")),
            body=None,
        )

        with patch.object(agent, "_assistant_agent", return_value=object()), patch.object(
            agent.Runner, "run_streamed", return_value=BrokenStream(error)
        ), patch.object(agent, "_log") as log:
            with self.assertRaises(APIStatusError):
                list(agent.stream_agent_reply("oi", "conv-429"))

        self.assertTrue(any(call.args[2] == "rate_limited" for call in log.call_args_list))

    def test_stream_agent_reply_reraises_generic_status_error(self) -> None:
        error = APIStatusError(
            "server error",
            response=Response(500, request=Request("POST", "https://example.test")),
            body=None,
        )

        with patch.object(agent, "_assistant_agent", return_value=object()), patch.object(
            agent.Runner, "run_streamed", return_value=BrokenStream(error)
        ), patch.object(agent, "_log") as log:
            with self.assertRaises(APIStatusError):
                list(agent.stream_agent_reply("oi", "conv-500"))

        self.assertTrue(any(call.args[2] == "status_error" for call in log.call_args_list))

    def test_stream_agent_reply_reraises_rate_limit_error(self) -> None:
        error = RateLimitError(
            "rate limit",
            response=Response(429, request=Request("POST", "https://example.test")),
            body=None,
        )

        with patch.object(agent, "_assistant_agent", return_value=object()), patch.object(
            agent.Runner, "run_streamed", return_value=BrokenStream(error)
        ), patch.object(agent, "_log") as log:
            with self.assertRaises(RateLimitError):
                list(agent.stream_agent_reply("oi", "conv-rate-limit"))

        self.assertTrue(any(call.args[2] == "rate_limit_error" for call in log.call_args_list))

    def test_stream_agent_reply_reraises_connection_error(self) -> None:
        error = APIConnectionError(
            message="offline",
            request=Request("POST", "https://example.test"),
        )

        with patch.object(agent, "_assistant_agent", return_value=object()), patch.object(
            agent.Runner, "run_streamed", return_value=BrokenStream(error)
        ), patch.object(agent, "_log") as log:
            with self.assertRaises(APIConnectionError):
                list(agent.stream_agent_reply("oi", "conv-connection"))

        self.assertTrue(any(call.args[2] == "connection_error" for call in log.call_args_list))

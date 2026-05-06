from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from httpx import Request
from openai import APITimeoutError

import app


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class AppUnitTests(TestCase):
    def setUp(self) -> None:
        self.session_state = SessionState(
            messages=[],
            conversation_id="conv-test",
            memory_subject_id="user_123456",
            user_name="user_123456",
            notice_message=None,
        )
        app.st.session_state = self.session_state
        app.st.sidebar = nullcontext()
        app.st.chat_message = Mock(return_value=nullcontext())
        self.loading_placeholder = SimpleNamespace(caption=Mock(), empty=Mock())
        self.response_placeholder = SimpleNamespace(markdown=Mock(), empty=Mock())
        app.st.empty = Mock(side_effect=[self.loading_placeholder, self.response_placeholder])
        app.st.button = Mock(return_value=False)
        app.st.text_input = Mock(return_value="user_123456")
        app.st.toggle = Mock(return_value=False)
        app.st.subheader = Mock()
        app.st.caption = Mock()
        app.st.divider = Mock()
        app.st.error = Mock()
        app.st.markdown = Mock()
        app.st.toast = Mock()
        app.st.rerun = Mock()

    def test_memory_subject_id_from_user_name_normalizes_spaces(self) -> None:
        self.assertEqual(app._memory_subject_id_from_user_name(" Maria Silva "), "Maria_Silva")
        self.assertEqual(app._memory_subject_id_from_user_name("   "), app.DEFAULT_USER_NAME)

    def test_init_state_creates_single_remote_conversation_when_missing(self) -> None:
        self.session_state.pop("conversation_id")
        self.session_state.pop("memory_subject_id")
        self.session_state.user_name = "Maria Silva"

        with patch.object(app, "create_agent_conversation", return_value="conv-123") as create_conversation:
            app._init_state()

        create_conversation.assert_called_once_with("Maria_Silva")
        self.assertEqual(self.session_state.memory_subject_id, "Maria_Silva")
        self.assertEqual(self.session_state.conversation_id, "conv-123")
        self.assertEqual(self.session_state.messages, [])

    def test_init_state_reuses_existing_conversation(self) -> None:
        with patch.object(app, "create_agent_conversation") as create_conversation:
            app._init_state()

        create_conversation.assert_not_called()
        self.assertEqual(self.session_state.conversation_id, "conv-test")

    def test_set_user_creates_new_conversation_and_clears_messages_when_id_changes(self) -> None:
        self.session_state.messages = [app.ChatMessage(role="user", content="oi")]

        with patch.object(app, "create_agent_conversation", return_value="conv-maria") as create_conversation:
            app._set_user("Maria Silva")

        create_conversation.assert_called_once_with("Maria_Silva")
        self.assertEqual(self.session_state.user_name, "Maria Silva")
        self.assertEqual(self.session_state.memory_subject_id, "Maria_Silva")
        self.assertEqual(self.session_state.conversation_id, "conv-maria")
        self.assertEqual(self.session_state.messages, [])

    def test_set_user_reuses_existing_user(self) -> None:
        with patch.object(app, "create_agent_conversation") as create_conversation:
            app._set_user("user_123456")

        create_conversation.assert_not_called()

    def test_stream_response_clears_loading_on_first_chunk(self) -> None:
        placeholder = SimpleNamespace(markdown=Mock())
        loading_placeholder = SimpleNamespace(empty=Mock())

        with patch.object(app, "stream_agent_reply", return_value=iter(["Ola", " mundo"])) as stream_reply:
            response = app._stream_response(
                "oi",
                "conv-test",
                "turn-1",
                placeholder,
                loading_placeholder,
                [],
                False,
            )

        self.assertEqual(response, "Ola mundo")
        loading_placeholder.empty.assert_called_once()
        placeholder.markdown.assert_called()
        stream_reply.assert_called_once_with("oi", "conv-test", use_tavily=False, turn_id="turn-1")

    def test_stream_response_passes_tavily_toggle(self) -> None:
        placeholder = SimpleNamespace(markdown=Mock())
        loading_placeholder = SimpleNamespace(empty=Mock())

        with patch.object(app, "stream_agent_reply", return_value=iter(["Ola"])) as stream_reply:
            app._stream_response(
                "oi",
                "conv-test",
                "turn-1",
                placeholder,
                loading_placeholder,
                [],
                True,
            )

        stream_reply.assert_called_once_with("oi", "conv-test", use_tavily=True, turn_id="turn-1")

    def test_handle_prompt_appends_messages_after_successful_response(self) -> None:
        with patch.object(app, "_stream_response", return_value="Resposta final"):
            app._handle_prompt("oi")

        self.assertEqual(
            self.session_state.messages,
            [
                {"role": "user", "content": "oi"},
                {"role": "assistant", "content": "Resposta final"},
            ],
        )
        app.st.rerun.assert_called_once()

    def test_handle_prompt_appends_user_after_runtime_error(self) -> None:
        def broken_stream(prompt, conversation_id, turn_id, placeholder, loading_placeholder, chunks, use_tavily):
            chunks.extend(["resposta", " parcial"])
            raise RuntimeError("falhou no meio")

        with patch.object(app, "_stream_response", side_effect=broken_stream):
            app._handle_prompt("oi")

        self.assertEqual(self.session_state.messages, [{"role": "user", "content": "oi"}])
        self.assertEqual(
            self.session_state.notice_message,
            {"text": app.RUNTIME_ERROR_MESSAGE},
        )
        app.st.rerun.assert_called_once()

    def test_handle_prompt_appends_user_after_config_error(self) -> None:
        def broken_stream(prompt, conversation_id, turn_id, placeholder, loading_placeholder, chunks, use_tavily):
            raise app.AgentConfigError("faltou env")

        with patch.object(app, "_stream_response", side_effect=broken_stream):
            app._handle_prompt("oi")

        self.assertEqual(self.session_state.messages, [{"role": "user", "content": "oi"}])
        self.assertEqual(
            self.session_state.notice_message,
            {"text": app.CONFIG_ERROR_MESSAGE},
        )
        app.st.rerun.assert_called_once()

    def test_handle_prompt_appends_user_after_timeout(self) -> None:
        def stalled_stream(prompt, conversation_id, turn_id, placeholder, loading_placeholder, chunks, use_tavily):
            chunks.extend(["resposta"])
            raise APITimeoutError(request=Request("POST", "https://example.test"))

        with patch.object(app, "_stream_response", side_effect=stalled_stream):
            app._handle_prompt("oi")

        self.assertEqual(self.session_state.messages, [{"role": "user", "content": "oi"}])
        self.assertEqual(
            self.session_state.notice_message,
            {"text": app.TIMEOUT_ERROR_MESSAGE},
        )
        app.st.rerun.assert_called_once()

    def test_handle_prompt_ignores_blank_input(self) -> None:
        with patch.object(app, "_stream_response") as stream_response:
            app._handle_prompt("   ")

        self.assertEqual(self.session_state.messages, [])
        stream_response.assert_not_called()

    def test_handle_prompt_replaces_empty_response_with_runtime_message(self) -> None:
        with patch.object(app, "_stream_response", return_value="   "):
            app._handle_prompt("oi")

        self.response_placeholder.markdown.assert_called_with(app.RUNTIME_ERROR_MESSAGE)
        self.assertEqual(
            self.session_state.messages,
            [
                {"role": "user", "content": "oi"},
                {"role": "assistant", "content": app.RUNTIME_ERROR_MESSAGE},
            ],
        )
        app.st.rerun.assert_called_once()

    def test_render_notice_message_keeps_error_in_memory(self) -> None:
        self.session_state.notice_message = {"text": "atenção"}

        app._render_notice_message()

        app.st.error.assert_called_once_with("atenção")
        self.assertEqual(self.session_state.notice_message, {"text": "atenção"})

    def test_render_sidebar_applies_user_name(self) -> None:
        app.st.text_input.return_value = "Maria Silva"
        app.st.button.side_effect = [True, False]
        app.st.toggle.return_value = False

        with patch.object(app, "create_agent_conversation", return_value="conv-maria") as create_conversation:
            app._render_sidebar()

        create_conversation.assert_called_once_with("Maria_Silva")
        self.assertEqual(self.session_state.user_name, "Maria Silva")
        self.assertEqual(self.session_state.memory_subject_id, "Maria_Silva")
        self.assertEqual(self.session_state.conversation_id, "conv-maria")
        app.st.rerun.assert_called_once()

    def test_render_sidebar_updates_tavily_toggle(self) -> None:
        app.st.toggle.return_value = True

        app._render_sidebar()

        self.assertTrue(self.session_state.use_tavily)

    def test_render_sidebar_clears_chat_and_creates_new_conversation_for_current_user(self) -> None:
        self.session_state.messages = [
            app.ChatMessage(role="user", content="oi"),
        ]
        app.st.button.side_effect = [False, True]
        app.st.toggle.return_value = False

        with patch.object(app, "create_agent_conversation", return_value="conv-new") as create_conversation:
            app._render_sidebar()

        create_conversation.assert_called_once_with("user_123456")
        self.assertEqual(self.session_state.messages, [])
        self.assertEqual(self.session_state.conversation_id, "conv-new")
        self.assertEqual(
            self.session_state.toast_message,
            "Conversa limpa. Voce pode comecar uma nova agora.",
        )
        app.st.rerun.assert_called_once()

    def test_render_history_keeps_session_order(self) -> None:
        self.session_state.messages = [
            {"role": "user", "content": "primeira"},
            {"role": "assistant", "content": "segunda"},
            {"role": "user", "content": "terceira"},
        ]
        app.st.chat_message = Mock(side_effect=[nullcontext(), nullcontext(), nullcontext()])

        app._render_history()

        self.assertEqual(
            [call.args[0] for call in app.st.chat_message.call_args_list],
            ["user", "assistant", "user"],
        )
        self.assertEqual(
            [call.args[0] for call in app.st.markdown.call_args_list[-3:]],
            ["primeira", "segunda", "terceira"],
        )

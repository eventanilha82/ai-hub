import logging
from typing import TypedDict
from uuid import uuid4

import streamlit as st
from openai import APITimeoutError

from agent import AgentConfigError, create_agent_conversation, stream_agent_reply

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = False

RUNTIME_ERROR_MESSAGE = "Nao consegui completar a resposta agora. Tente novamente em instantes."
TIMEOUT_ERROR_MESSAGE = "A resposta demorou demais e foi interrompida. Tente novamente."
CONFIG_ERROR_MESSAGE = "A configuracao do agente esta invalida. Revise o .env e tente novamente."
DEFAULT_USER_NAME = "user_123456"
USER_NAME_INPUT_KEY = "user_name_input"


class ChatMessage(TypedDict):
    role: str
    content: str


def _append_visible_message(messages: list[ChatMessage], message: ChatMessage) -> None:
    if messages:
        previous = messages[-1]
        if previous["role"] == message["role"] and previous["content"] == message["content"]:
            return
    messages.append(message)


def _memory_subject_id_from_user_name(user_name: str) -> str:
    normalized = "_".join(user_name.strip().split())
    return normalized or DEFAULT_USER_NAME


def _current_user_name() -> str:
    value = st.session_state.get("user_name")
    return value if isinstance(value, str) and value.strip() else DEFAULT_USER_NAME


def _set_user(user_name: str) -> None:
    memory_subject_id = _memory_subject_id_from_user_name(user_name)
    if st.session_state.get("memory_subject_id") == memory_subject_id:
        return

    st.session_state.user_name = user_name.strip() or DEFAULT_USER_NAME
    st.session_state.memory_subject_id = memory_subject_id
    st.session_state.conversation_id = create_agent_conversation(memory_subject_id)
    st.session_state.messages = []
    st.session_state.notice_message = None
    st.session_state.toast_message = f"Usuario ativo: {st.session_state.user_name}."


def _ensure_conversation() -> str:
    conversation_id = st.session_state.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id.strip():
        return conversation_id

    memory_subject_id = st.session_state.get("memory_subject_id")
    if not isinstance(memory_subject_id, str) or not memory_subject_id.strip():
        memory_subject_id = _memory_subject_id_from_user_name(_current_user_name())
        st.session_state.memory_subject_id = memory_subject_id

    conversation_id = create_agent_conversation(memory_subject_id)
    st.session_state.conversation_id = conversation_id
    logger.info(
        "UI: conversation criada conversation_id=%s memory_subject_id=%s",
        conversation_id,
        memory_subject_id,
    )
    return conversation_id


def _init_state() -> None:
    st.session_state.setdefault("toast_message", None)
    st.session_state.setdefault("notice_message", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("use_tavily", False)
    st.session_state.setdefault("user_name", DEFAULT_USER_NAME)
    st.session_state.setdefault(USER_NAME_INPUT_KEY, _current_user_name())
    st.session_state.setdefault("memory_subject_id", _memory_subject_id_from_user_name(_current_user_name()))
    _ensure_conversation()


def _avatar_for(role: str) -> str:
    return ":material/person:" if role == "user" else ":material/robot_2:"


def _stream_response(
    prompt: str,
    conversation_id: str,
    turn_id: str,
    placeholder: st.delta_generator.DeltaGenerator,
    chunks: list[str],
    use_tavily: bool,
) -> str:
    logger.info(
        "UI: iniciando renderizacao incremental conversation_id=%s prompt_len=%s",
        conversation_id,
        len(prompt),
    )
    for chunk in stream_agent_reply(prompt, conversation_id, use_tavily=use_tavily, turn_id=turn_id):
        if not chunk:
            continue
        chunks.append(chunk)
        placeholder.markdown("".join(chunks))
    response = "".join(chunks)
    logger.info(
        "UI: renderizacao incremental concluida conversation_id=%s chunks=%s chars=%s",
        conversation_id,
        len(chunks),
        len(response),
    )
    return response


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=_avatar_for(message["role"])):
            st.markdown(message["content"])


def _set_error_message(text: str) -> None:
    st.session_state.notice_message = {"text": text}


def _render_notice_message() -> None:
    notice_message = st.session_state.get("notice_message")
    if not isinstance(notice_message, dict):
        return

    text = notice_message.get("text")
    if not isinstance(text, str) or not text.strip():
        st.session_state.notice_message = None
        return

    st.error(text)


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.memory_subject_id = _memory_subject_id_from_user_name(_current_user_name())
    st.session_state.conversation_id = create_agent_conversation(st.session_state.memory_subject_id)
    st.session_state.notice_message = None
    st.session_state.toast_message = "Conversa limpa. Voce pode comecar uma nova agora."
    st.rerun()


def _render_sidebar() -> None:
    with st.sidebar:
        st.subheader(":material/chat: Assistente")
        st.caption("Memoria remota por usuario e resposta em streaming.")
        st.divider()
        st.markdown("**Usuario**")
        user_name = st.text_input("Nome do usuario", key=USER_NAME_INPUT_KEY)
        selected_memory_subject_id = _memory_subject_id_from_user_name(user_name)
        active_memory_subject_id = st.session_state.get("memory_subject_id")
        st.caption(f"ID: `{selected_memory_subject_id}`")
        should_apply_user = selected_memory_subject_id != active_memory_subject_id
        if st.button("Usar usuario", type="primary", icon=":material/person:") or should_apply_user:
            _set_user(user_name)
            st.rerun()
        st.divider()
        st.markdown("**Ferramentas**")
        st.session_state.use_tavily = st.toggle(
            "Tavily MCP",
            value=bool(st.session_state.get("use_tavily", False)),
            help="Habilita pesquisa externa via MCP Tavily nesta conversa.",
        )
        st.divider()
        st.markdown("**Conversa atual**")
        st.caption(f"Mensagens visiveis: {len(st.session_state.messages)}")
        conversation_id = st.session_state.get("conversation_id", "")
        if conversation_id:
            st.caption(f"Conversation: `{conversation_id[-12:]}`")
        if st.button("Limpar conversa", type="secondary", icon=":material/delete:"):
            _clear_chat()
        st.divider()
        st.markdown("**Como funciona**")
        st.caption("1. Informe o usuario.")
        st.caption("2. Pergunte normalmente.")
        st.caption("3. O Runner usa o conversation_id remoto para manter contexto.")


def _handle_prompt(prompt: str) -> None:
    prompt = prompt.strip()
    if not prompt:
        return

    try:
        conversation_id = _ensure_conversation()
    except AgentConfigError as error:
        logger.exception("UI: falha ao criar conversation detalhe=%s", error)
        _set_error_message(CONFIG_ERROR_MESSAGE)
        st.rerun()
        return

    turn_id = uuid4().hex[:8]
    use_tavily = bool(st.session_state.get("use_tavily", False))
    logger.info("UI: recebida nova pergunta conversation_id=%s prompt_len=%s", conversation_id, len(prompt))

    with st.chat_message("user", avatar=_avatar_for("user")):
        st.markdown(prompt)

    response_chunks: list[str] = []
    with st.chat_message("assistant", avatar=_avatar_for("assistant")):
        placeholder = st.empty()
        try:
            with st.spinner("Preparando resposta..."):
                response_text = _stream_response(
                    prompt,
                    conversation_id,
                    turn_id,
                    placeholder,
                    response_chunks,
                    use_tavily,
                )
        except AgentConfigError as error:
            logger.exception(
                "UI: falha de configuracao/runtime conversation_id=%s tipo=%s parcial_len=%s detalhe=%s",
                conversation_id,
                type(error).__name__,
                len("".join(response_chunks)),
                error,
            )
            _append_visible_message(st.session_state.messages, ChatMessage(role="user", content=prompt))
            _set_error_message(CONFIG_ERROR_MESSAGE)
            st.rerun()
            return
        except APITimeoutError:
            _append_visible_message(st.session_state.messages, ChatMessage(role="user", content=prompt))
            _set_error_message(TIMEOUT_ERROR_MESSAGE)
            st.rerun()
            return
        except Exception as error:
            logger.exception(
                "UI: falha inesperada conversation_id=%s tipo=%s parcial_len=%s detalhe=%s",
                conversation_id,
                type(error).__name__,
                len("".join(response_chunks)),
                error,
            )
            _append_visible_message(st.session_state.messages, ChatMessage(role="user", content=prompt))
            _set_error_message(RUNTIME_ERROR_MESSAGE)
            st.rerun()
            return

        if not response_text.strip():
            response_text = RUNTIME_ERROR_MESSAGE
            placeholder.markdown(response_text)
            logger.warning("UI: resposta vazia conversation_id=%s", conversation_id)

    _append_visible_message(st.session_state.messages, ChatMessage(role="user", content=prompt))
    _append_visible_message(st.session_state.messages, ChatMessage(role="assistant", content=response_text))
    st.session_state.notice_message = None
    st.rerun()
    logger.info("UI: resposta final exibida conversation_id=%s len=%s", conversation_id, len(response_text))


def main() -> None:
    st.set_page_config(page_title="AI Hub", page_icon=":material/chat:")
    st.title("AI Hub")
    st.caption("Suporte conversacional com conversation remota por usuario e resposta em streaming.")
    try:
        _init_state()
    except AgentConfigError:
        st.error(CONFIG_ERROR_MESSAGE)
        return
    if st.session_state.toast_message:
        st.toast(st.session_state.toast_message, icon=":material/check_circle:")
        st.session_state.toast_message = None
    _render_sidebar()
    _render_history()
    _render_notice_message()
    if prompt := st.chat_input("Digite sua pergunta"):
        _handle_prompt(prompt)


if __name__ == "__main__":
    main()

import os
from unittest import TestCase, skipUnless
from uuid import uuid4

import agent


RUN_LIVE_TESTS = os.getenv("RUN_LIVE_AGENT_TESTS") == "1"


@skipUnless(RUN_LIVE_TESTS, "Defina RUN_LIVE_AGENT_TESTS=1 para chamar a API real.")
class AgentLiveTests(TestCase):
    def test_live_stream_returns_text_for_safe_prompt(self) -> None:
        conversation_id = agent.create_agent_conversation(f"live-safe-{uuid4().hex}")
        response = "".join(
            agent.stream_agent_reply(
                "Responda em uma frase curta: o que voce faz?",
                conversation_id,
            )
        )

        self.assertTrue(response.strip())

    def test_live_conversation_keeps_two_turn_context(self) -> None:
        conversation_id = agent.create_agent_conversation(f"live-history-{uuid4().hex}")
        response_one = "".join(agent.stream_agent_reply("oi", conversation_id))
        response_two = "".join(agent.stream_agent_reply("o que voce faz?", conversation_id))

        self.assertTrue(response_one.strip())
        self.assertTrue(response_two.strip())

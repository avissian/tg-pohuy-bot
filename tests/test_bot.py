import importlib.util
import os
import pathlib
import unittest
from unittest import mock

import httpx


def load_bot(**overrides):
    env = {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "ALLOWED_CHAT_IDS": "1",
        "LLM_API_BASE": "https://llm.example/v1",
        "LLM_API_KEY": "test-key",
        **overrides,
    }
    path = pathlib.Path(__file__).resolve().parents[1] / "bot.py"
    spec = importlib.util.spec_from_file_location("bot_under_test", path)
    bot = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, env, clear=True):
        spec.loader.exec_module(bot)
    return bot


class TimeoutTests(unittest.TestCase):
    def test_llm_read_timeout_default_and_override(self):
        for env, expected in (({}, 600.0), ({"REQUEST_TIMEOUT": "900"}, 900.0)):
            with self.subTest(env=env):
                bot = load_bot(**env)
                requests = []

                def respond(request):
                    requests.append(request)
                    return httpx.Response(
                        200, json={"choices": [{"message": {"content": " ответ "}}]}
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    answer = bot.llm_chat(client, [{"role": "user", "content": "тест"}])

                self.assertEqual(answer, "ответ")
                self.assertEqual(len(requests), 1)
                self.assertEqual(
                    requests[0].extensions["timeout"],
                    {"connect": 10, "read": expected, "write": 30, "pool": 30},
                )

    def test_telegram_timeout_does_not_follow_llm_timeout(self):
        bot = load_bot(REQUEST_TIMEOUT="900")
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            self.assertEqual(
                bot.tg_call(client, "sendChatAction", chat_id=1, action="typing"),
                {"ok": True},
            )

        self.assertEqual(
            requests[0].extensions["timeout"],
            {"connect": 30, "read": 30, "write": 30, "pool": 30},
        )

    def test_timeout_is_reported_without_retry_and_stops_typing(self):
        for error in (httpx.ReadTimeout, httpx.ConnectTimeout):
            with self.subTest(error=error.__name__):
                bot = load_bot()
                requests = []

                def respond(request):
                    requests.append(request)
                    raise error("timed out", request=request)

                conn = mock.sentinel.connection
                msg = {"chat": {"id": 1}, "from": {"id": 2}, "text": "тест"}
                with (
                    httpx.Client(transport=httpx.MockTransport(respond)) as client,
                    mock.patch.object(bot, "history_append") as append,
                    mock.patch.object(bot, "history_load", return_value=[]),
                    mock.patch.object(bot, "send_text") as send,
                    mock.patch.object(bot.threading, "Thread") as thread,
                    self.assertLogs("bot", level="ERROR") as logs,
                ):
                    bot.handle_message(conn, client, msg)

                self.assertEqual(len(requests), 1)
                append.assert_called_once_with(conn, 1, "user", "тест")
                send.assert_called_once_with(
                    client, 1, "LLM API не ответил вовремя. Попробуй ещё раз."
                )
                thread.return_value.start.assert_called_once_with()
                self.assertTrue(thread.call_args.kwargs["args"][2].is_set())
                self.assertIn("LLM timeout after", logs.output[0])
                self.assertIn(error.__name__, logs.output[0])
                self.assertIn("read_timeout=600.0s", logs.output[0])


if __name__ == "__main__":
    unittest.main()

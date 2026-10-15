#!/usr/bin/env python3
"""Telegram-чат-бот — тонкий клиент до LLM API со скиллом pohuy.

Архитектура "тонкого клиента":
  - whitelist chat_id проверяется ДО обращения к модели (тупой код);
  - у модели нет никаких tool-вызовов: это чистый текстовый диалог;
  - история диалогов в sqlite, системный промпт собирается из файлов скилла;
  - исходящие соединения только к api.telegram.org и LLM API.

Никакого исполнения команд, файловых операций и сети по желанию модели.
"""
from __future__ import annotations

import html
import json
import logging
import os
import pathlib
import sqlite3
import sys
import time

import httpx
import threading

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_CHATS = {
    int(x) for x in os.environ["ALLOWED_CHAT_IDS"].split(",") if x.strip()
}
LLM_BASE = os.environ["LLM_API_BASE"].rstrip("/")
LLM_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
HISTORY_DEPTH = int(os.environ.get("HISTORY_DEPTH", "16"))
MAX_REPLY_CHARS = int(os.environ.get("MAX_REPLY_CHARS", "3800"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))
DATA_DIR = pathlib.Path(os.environ.get("DATA_DIR", "/data"))

TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
SKILL_DIR = pathlib.Path(__file__).parent / "skills" / "pohuy"

log = logging.getLogger("bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

# ---------------------------------------------------------------- system prompt


def build_system_prompt() -> str:
    parts = [
        "Ты — чат-бот в Telegram. Общаешься с инженером на русском.",
        "Ниже — твой языковой скилл: следуй его правилам дословно.",
        "",
        "=== SKILL.md ===",
        (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"),
    ]
    refs = sorted((SKILL_DIR / "references").glob("*.md"))
    for ref in refs:
        parts += ["", f"=== references/{ref.name} ===", ref.read_text(encoding="utf-8")]
    parts += [
        "",
        "Конец скилла. Дополнительно от бота:",
        "- длинные ответы нарезай на абзацы, Telegram-читабельно;",
        "- ты не исполняешь команды и не имеешь доступа к каким-либо системам:",
        "  если просят что-то выполнить — честно скажи, что ты собеседник без рук.",
        "",
        "=== Форматирование для Telegram ===",
        "Твои ответы рендерятся в Telegram как Markdown. Доступно:",
        "  *жирный*  _курсив_  `моно`  ```блок кода```  [текст](url)",
        "НЕДОСТУПНО (не используй): ###, ##, таблицы, LaTeX, ~~зачёркнутый~~, ||спойлеры||.",
        "Списки: обычный дефис на новой строке. Тире — просто дефис.",
        "Пиши компактно, без лишнего форматирования: чистый текст чаще лучше.",
    ]
    return "\n".join(parts)


SYSTEM_PROMPT = build_system_prompt()

# ---------------------------------------------------------------------- storage


def db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATA_DIR / "history.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        " chat_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,"
        " ts REAL NOT NULL)"
    )
    return conn


def history_load(conn: sqlite3.Connection, chat_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE chat_id=? "
        "ORDER BY ts DESC LIMIT ?",
        (chat_id, HISTORY_DEPTH),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def history_append(
    conn: sqlite3.Connection, chat_id: int, role: str, content: str
) -> None:
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, ts) VALUES (?,?,?,?)",
        (chat_id, role, content, time.time()),
    )
    conn.commit()


def history_reset(conn: sqlite3.Connection, chat_id: int) -> None:
    conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.commit()


# ---------------------------------------------------------------------- telegram


def tg_call(client: httpx.Client, method: str, **kwargs) -> dict:
    r = client.post(f"{TG}/{method}", json=kwargs, timeout=30)
    r.raise_for_status()
    return r.json()


def send_text(client: httpx.Client, chat_id: int, text: str) -> None:
    """Markdown → Telegram, с fallback на чистый текст при ошибке парсинга."""
    for i in range(0, min(len(text), MAX_REPLY_CHARS), 4096):
        chunk = text[i : i + 4096]
        try:
            tg_call(client, "sendMessage", chat_id=chat_id, text=chunk, parse_mode="Markdown")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 and "parse" in exc.response.text.lower():
                tg_call(client, "sendMessage", chat_id=chat_id, text=chunk)
            else:
                raise
    if len(text) > MAX_REPLY_CHARS:
        tg_call(client, "sendMessage", chat_id=chat_id, text="… (обрезано)")


def typing_loop(client: httpx.Client, chat_id: int, done: threading.Event) -> None:
    """Держит индикатор «печатает…» пока LLM думает (Telegram гасит его за ~5с)."""
    while not done.is_set():
        try:
            tg_call(client, "sendChatAction", chat_id=chat_id, action="typing")
        except Exception:  # noqa: BLE001 — индикатор не критичен
            pass
        done.wait(4)


# -------------------------------------------------------------------------- llm


def llm_chat(client: httpx.Client, messages: list[dict]) -> str:
    r = client.post(
        f"{LLM_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_KEY}"},
        json={"model": LLM_MODEL, "messages": messages, "temperature": 0.7},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ------------------------------------------------------------------------ main


def handle_message(conn: sqlite3.Connection, client: httpx.Client, msg: dict) -> None:
    chat_id = msg["chat"]["id"]
    text = (msg.get("text") or "").strip()
    if not text:
        return  # фото, войсы, стикеры, файлы — всё мимо
    who = msg["from"].get("username") or msg["from"].get("id", "?")

    if chat_id not in ALLOWED_CHATS:
        log.warning("DROP chat_id=%s from @%s: not in allowlist", chat_id, who)
        return

    if text == "/reset":
        history_reset(conn, chat_id)
        send_text(client, chat_id, "История диалога очищена.")
        return
    if text in ("/start", "/help"):
        send_text(
            client,
            chat_id,
            "Чат-бот с характером. Просто пиши. /reset — очистить контекст.",
        )
        return

    history_append(conn, chat_id, "user", text)
    typing_done = threading.Event()
    typing_thread = threading.Thread(
        target=typing_loop, args=(client, chat_id, typing_done), daemon=True
    )
    typing_thread.start()
    try:
        answer = llm_chat(
            client, [{"role": "system", "content": SYSTEM_PROMPT}]
            + history_load(conn, chat_id)
        )
    except httpx.HTTPStatusError as exc:
        log.error("LLM HTTP %s: %s", exc.response.status_code, exc.response.text[:300])
        send_text(client, chat_id, f"LLM API вернул ошибку {exc.response.status_code}.")
        return
    except httpx.HTTPError as exc:
        log.error("LLM network error: %s", exc)
        send_text(client, chat_id, "Не дозвонился до LLM API. Попробуй ещё раз.")
        return
    finally:
        typing_done.set()

    history_append(conn, chat_id, "assistant", answer)
    send_text(client, chat_id, answer)
    log.info("chat=%s @%s: %d chars in, %d chars out", chat_id, who, len(text), len(answer))


def main() -> None:
    log.info(
        "bot starting: model=%s, allowlist=%s, history=%d",
        LLM_MODEL,
        sorted(ALLOWED_CHATS),
        HISTORY_DEPTH,
    )
    conn = db()
    client = httpx.Client()
    offset = 0
    with client:
        while True:
            try:
                resp = client.post(
                    f"{TG}/getUpdates",
                    json={"timeout": 60, "offset": offset, "allowed_updates": ["message"]},
                    timeout=90,
                )
                resp.raise_for_status()
                for upd in resp.json().get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message")
                    if not msg:
                        continue
                    try:
                        handle_message(conn, client, msg)
                    except Exception:  # noqa: BLE001 — бот не должен умирать
                        log.exception("handler error on update %s", upd["update_id"])
            except httpx.HTTPError as exc:
                log.error("telegram poll error: %s — retry in 5s", exc)
                time.sleep(5)


if __name__ == "__main__":
    main()

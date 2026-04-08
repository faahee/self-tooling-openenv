"""Email Agent — Send, read, summarize, and auto-reply to emails via SMTP/IMAP."""
from __future__ import annotations

import asyncio
import email as email_lib
import imaplib
import logging
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from core.llm_core import TaskType

logger = logging.getLogger(__name__)


class EmailAgent:
    """Send and receive emails using standard Python SMTP/IMAP.

    Config (config.yaml):
        email:
          smtp_host: smtp.gmail.com
          smtp_port: 587
          imap_host: imap.gmail.com
          imap_port: 993
          address: yourmail@gmail.com
          password: your_app_password     # Gmail app password
    """

    def __init__(self, config: dict, llm_core, ui_layer) -> None:
        self.config = config.get("email", {})
        self.llm = llm_core
        self.ui = ui_layer
        self._smtp_host = self.config.get("smtp_host", "smtp.gmail.com")
        self._smtp_port = int(self.config.get("smtp_port", 587))
        self._imap_host = self.config.get("imap_host", "imap.gmail.com")
        self._imap_port = int(self.config.get("imap_port", 993))
        self._address  = self.config.get("address", "")
        self._password = self.config.get("password", "")

    @property
    def _configured(self) -> bool:
        return bool(self._address and self._password)

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle(self, user_input: str, context: dict) -> str:
        """Route email requests.

        Args:
            user_input: Natural-language email command.
            context: Orchestrator context dict.

        Returns:
            Result string.
        """
        if not self._configured:
            return (
                "Email not configured. Add to config.yaml:\n"
                "  email:\n"
                "    smtp_host: smtp.gmail.com\n"
                "    smtp_port: 587\n"
                "    imap_host: imap.gmail.com\n"
                "    imap_port: 993\n"
                "    address: you@gmail.com\n"
                "    password: your_app_password"
            )

        low = user_input.lower()

        if any(w in low for w in ("send", "compose", "write email", "email to")):
            return await self._send(user_input)

        if any(w in low for w in ("read", "check", "inbox", "unread", "latest email", "show email")):
            return await self._read(user_input)

        if any(w in low for w in ("summarize", "summary of", "what emails")):
            return await self._summarize(user_input)

        if any(w in low for w in ("reply", "respond to")):
            return await self._reply(user_input)

        if any(w in low for w in ("search email", "find email", "look for email")):
            return await self._search(user_input)

        return await self._send(user_input)  # default: try to send

    # ── Send ──────────────────────────────────────────────────────────────────

    async def _send(self, user_input: str) -> str:
        # Extract: to, subject, body using LLM
        prompt = (
            f"Extract from this instruction:\n\"{user_input}\"\n\n"
            "Return EXACTLY this JSON (no extra text):\n"
            '{"to": "email@example.com", "subject": "Subject here", "body": "Email body here"}'
        )
        raw = await self.llm.generate(prompt, task_type=TaskType.EMAIL_COMPOSE)
        try:
            m = re.search(r'\{[^}]+\}', raw, re.S)
            data = __import__("json").loads(m.group(0)) if m else {}
            to_addr = data.get("to", "")
            subject = data.get("subject", "Message from JARVIS")
            body    = data.get("body", user_input)
        except Exception:
            # Fallback: parse manually
            to_match = re.search(r'\b[\w.+-]+@[\w-]+\.\w+\b', user_input)
            to_addr  = to_match.group(0) if to_match else ""
            subject  = "Message from JARVIS"
            body     = user_input

        if not to_addr:
            return "Could not extract recipient email address. Please include it explicitly."

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._smtp_send, to_addr, subject, body)

    def _smtp_send(self, to_addr: str, subject: str, body: str) -> str:
        try:
            msg = MIMEMultipart()
            msg["From"]    = self._address
            msg["To"]      = to_addr
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self._smtp_host, self._smtp_port) as s:
                s.ehlo()
                s.starttls()
                s.login(self._address, self._password)
                s.sendmail(self._address, to_addr, msg.as_string())
            return f"Email sent to {to_addr} — Subject: {subject}"
        except Exception as e:
            return f"Failed to send email: {e}"

    # ── Read ──────────────────────────────────────────────────────────────────

    async def _read(self, user_input: str) -> str:
        n_match = re.search(r'\b(\d+)\b', user_input)
        n = int(n_match.group(1)) if n_match else 5
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._imap_fetch, n)

    def _imap_fetch(self, n: int = 5) -> str:
        try:
            mail = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
            mail.login(self._address, self._password)
            mail.select("inbox")
            _, data = mail.search(None, "UNSEEN")
            ids = data[0].split()[-n:]
            emails: list[str] = []
            for uid in reversed(ids):
                _, msg_data = mail.fetch(uid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw)
                sender  = msg.get("From", "?")
                subject = msg.get("Subject", "(no subject)")
                date    = msg.get("Date", "")
                body    = self._extract_body(msg)[:300]
                emails.append(f"From: {sender}\nSubject: {subject}\nDate: {date}\n{body}\n---")
            mail.logout()
            if not emails:
                return "No unread emails."
            return f"Unread emails ({len(emails)}):\n\n" + "\n".join(emails)
        except Exception as e:
            return f"IMAP error: {e}"

    def _extract_body(self, msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode(errors="replace")
        return msg.get_payload(decode=True).decode(errors="replace") if isinstance(msg.get_payload(), bytes) else str(msg.get_payload())

    # ── Summarize ─────────────────────────────────────────────────────────────

    async def _summarize(self, user_input: str) -> str:
        emails_text = self._imap_fetch(10)
        prompt = f"Here are recent emails:\n\n{emails_text}\n\nSummarize the key points and action items."
        return await self.llm.generate(prompt, task_type=TaskType.SIMPLE_CHAT)

    # ── Reply ─────────────────────────────────────────────────────────────────

    async def _reply(self, user_input: str) -> str:
        return "To auto-reply: read your emails first, then say 'reply to <sender> saying <message>'"

    # ── Search ────────────────────────────────────────────────────────────────

    async def _search(self, user_input: str) -> str:
        kw_match = re.search(r'(?:search|find|look for)\s+(?:email\s+)?(?:about\s+)?["\']?(.+?)["\']?$', user_input, re.I)
        keyword = kw_match.group(1).strip() if kw_match else user_input
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._imap_search, keyword)

    def _imap_search(self, keyword: str) -> str:
        try:
            mail = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
            mail.login(self._address, self._password)
            mail.select("inbox")
            _, data = mail.search(None, f'(SUBJECT "{keyword}")')
            ids = data[0].split()[-5:]
            results = []
            for uid in reversed(ids):
                _, msg_data = mail.fetch(uid, "(RFC822)")
                msg = email_lib.message_from_bytes(msg_data[0][1])
                results.append(f"- {msg.get('Subject','')} from {msg.get('From','')} on {msg.get('Date','')}")
            mail.logout()
            return f"Found {len(results)} emails matching '{keyword}':\n" + "\n".join(results) if results else f"No emails found matching '{keyword}'"
        except Exception as e:
            return f"Search error: {e}"

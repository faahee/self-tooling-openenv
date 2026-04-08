from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .models import SupportTriageAction, SupportTriageState, SupportTriageStepResult


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class SupportTriageEnv:
    def __init__(self, base_url: str, process: subprocess.Popen | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._process = process
        self._client = httpx.AsyncClient(timeout=60)

    @classmethod
    async def from_docker_image(cls, image_name: str | None = None) -> "SupportTriageEnv":
        explicit_url = os.getenv("OPENENV_LOCAL_URL")
        if explicit_url:
            env = cls(explicit_url)
            await env._wait_until_ready()
            return env

        image_name = image_name or os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
        if image_name:
            port = _free_port()
            proc = subprocess.Popen(
                ["docker", "run", "--rm", "-p", f"{port}:7860", image_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            env = cls(f"http://127.0.0.1:{port}", process=proc)
            await env._wait_until_ready()
            return env

        port = _free_port()
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "support_triage_env.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        env = cls(f"http://127.0.0.1:{port}", process=proc)
        await env._wait_until_ready()
        return env

    async def _wait_until_ready(self) -> None:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                resp = await self._client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    return
            except Exception:
                await asyncio.sleep(0.5)
                continue
            await asyncio.sleep(0.5)
        raise RuntimeError(f"Environment did not become ready: {self.base_url}")

    async def reset(self, task_name: str | None = None) -> SupportTriageStepResult:
        payload = {"task_name": task_name} if task_name else {}
        resp = await self._client.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return SupportTriageStepResult.model_validate(resp.json())

    async def step(self, action: SupportTriageAction) -> SupportTriageStepResult:
        resp = await self._client.post(f"{self.base_url}/step", json=action.model_dump())
        resp.raise_for_status()
        return SupportTriageStepResult.model_validate(resp.json())

    async def state(self) -> SupportTriageState:
        resp = await self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return SupportTriageState.model_validate(resp.json())

    async def close(self) -> None:
        await self._client.aclose()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

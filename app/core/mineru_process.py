from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import IO

import httpx

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class MinerUProcessManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.process: subprocess.Popen | None = None
        self.log_handle: IO[bytes] | None = None
        self.started_by_manager = False

    async def start(self) -> None:
        if not self.settings.manage_mineru_process:
            return

        if await self._is_healthy():
            logger.info("MinerU is already healthy; FastAPI will not manage that external process.")
            return

        try:
            command = self._build_command()
            self.log_handle = self._open_log_file()
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self.log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            self.started_by_manager = True
            logger.info("Started MinerU with command: %s", " ".join(command))
            await self._wait_until_healthy()
        except Exception:
            self._close_log_file()
            if self.settings.mineru_fail_fast_on_startup:
                raise
            logger.exception(
                "MinerU did not become healthy during startup. FastAPI will continue; "
                "document conversion will fail until MinerU is healthy."
            )

    async def stop(self) -> None:
        if not self.started_by_manager or self.process is None:
            self._close_log_file()
            return

        if self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, timeout=self.settings.mineru_shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
        self._close_log_file()
        logger.info("Stopped MinerU process managed by FastAPI.")

    def _build_command(self) -> list[str]:
        command = shlex.split(self.settings.mineru_api_command, posix=os.name != "nt")
        command[0] = self._resolve_executable(command[0])
        return [
            *command,
            "--host",
            self.settings.mineru_api_host,
            "--port",
            str(self.settings.mineru_api_port),
        ]

    def _resolve_executable(self, executable: str) -> str:
        if Path(executable).exists():
            return executable

        resolved = which(executable)
        if resolved:
            return resolved

        suffix = ".exe" if os.name == "nt" and not executable.endswith(".exe") else ""
        interpreter_sibling = Path(sys.executable).with_name(f"{executable}{suffix}")
        if interpreter_sibling.exists():
            return str(interpreter_sibling)

        raise FileNotFoundError(
            f"MinerU command '{executable}' was not found. "
            f"Install MinerU into this environment or set MANAGE_MINERU_PROCESS=false. "
            f"Expected executable near: {Path(sys.executable).parent}"
        )

    def _open_log_file(self) -> IO[bytes] | None:
        if not self.settings.mineru_log_file:
            return None
        log_path = Path(self.settings.mineru_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path.open("ab")

    async def _wait_until_healthy(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.settings.mineru_startup_timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if self.process and self.process.poll() is not None:
                self._close_log_file()
                raise RuntimeError(
                    "MinerU process exited before becoming healthy. "
                    f"Check {self.settings.mineru_log_file or 'process output'}."
                )
            if await self._is_healthy():
                return
            await asyncio.sleep(1)
        raise RuntimeError(
            "Timed out waiting for MinerU to become healthy. "
            f"Check {self.settings.mineru_log_file or 'process output'}."
        )

    async def _is_healthy(self) -> bool:
        url = self.settings.mineru_parse_api_url.rstrip("/") + "/health"
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                response = await client.get(url)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _close_log_file(self) -> None:
        with contextlib.suppress(Exception):
            if self.log_handle:
                self.log_handle.close()
        self.log_handle = None

from __future__ import annotations

import select
import sys
import termios
import tty


class KeyCommandReader:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        self.buffer = ""

    def __enter__(self) -> "KeyCommandReader":
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def poll(self) -> list[str]:
        commands: list[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                break
            char = sys.stdin.read(1)
            if not char:
                break
            if char in {"s", "q", "d"}:
                commands.append(char)
            self.buffer = (self.buffer + char)[-8:]
            if self.buffer.endswith("exit"):
                commands.append("exit")
                self.buffer = ""
        return commands

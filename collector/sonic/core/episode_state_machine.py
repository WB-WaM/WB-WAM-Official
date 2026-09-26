from __future__ import annotations


class EpisodeStateMachine:
    def __init__(self) -> None:
        self.state = "idle"

    @property
    def is_idle(self) -> bool:
        return self.state == "idle"

    @property
    def is_waiting(self) -> bool:
        return self.state == "waiting_for_first_valid_frame"

    @property
    def is_recording(self) -> bool:
        return self.state == "recording"

    def start_waiting(self) -> None:
        if not self.is_idle:
            raise RuntimeError("recording start can only be requested from idle")
        self.state = "waiting_for_first_valid_frame"

    def start_recording(self) -> None:
        if not self.is_waiting:
            raise RuntimeError("recording can only begin after waiting for the first valid frame")
        self.state = "recording"

    def finish(self) -> None:
        self.state = "idle"

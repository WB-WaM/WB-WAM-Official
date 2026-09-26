from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .action_schema import FSQ_GRID_STEP, TOKEN_DIM
from .encoder_input import SONIC_ENCODER_INPUT_DIM


class SonicEncoder:
    """ONNX runtime wrapper for the released SONIC motion encoder."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
        snap_fsq_grid: bool = True,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SONIC encoder model not found: {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for physical-action WAM deployment") from exc
        self._session = ort.InferenceSession(
            str(path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input = self._session.get_inputs()[0]
        self._output = self._session.get_outputs()[0]
        self._snap_fsq_grid = bool(snap_fsq_grid)

    @property
    def input_shape(self) -> list[Any]:
        return list(self._input.shape)

    def encode(self, encoder_inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(encoder_inputs, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != SONIC_ENCODER_INPUT_DIM:
            raise ValueError(f"encoder_inputs must have shape [H,{SONIC_ENCODER_INPUT_DIM}], got {values.shape}")
        outputs: list[np.ndarray] = []
        fixed_batch_one = bool(self._input.shape and self._input.shape[0] == 1)
        batches = (values[index : index + 1] for index in range(values.shape[0])) if fixed_batch_one else (values,)
        for batch in batches:
            encoded = self._session.run(
                [self._output.name],
                {self._input.name: np.ascontiguousarray(batch)},
            )[0]
            outputs.append(np.asarray(encoded, dtype=np.float32).reshape(batch.shape[0], -1))
        tokens = np.concatenate(outputs, axis=0)
        if tokens.shape != (values.shape[0], TOKEN_DIM):
            raise ValueError(f"SONIC encoder returned {tokens.shape}, expected {(values.shape[0], TOKEN_DIM)}")
        if self._snap_fsq_grid:
            tokens = np.round(tokens / FSQ_GRID_STEP) * FSQ_GRID_STEP
        if not np.isfinite(tokens).all():
            raise ValueError("SONIC encoder returned NaN/Inf")
        return tokens.astype(np.float32, copy=False)

from typing import Iterable

import torch


class TrainHookManager:
    def __init__(self, *, log_every: int):
        self.log_every = int(log_every)
        self._hooks = []

    def append(self, hook) -> None:
        self._hooks.append(hook)

    def begin_step(self, *, step: int, sync_gradients: bool):
        active = bool(sync_gradients) and self.log_every > 0 and step % self.log_every == 0
        return TrainHookStep(self._hooks, step=step, active=active)


class TrainHookStep:
    def __init__(self, hooks: Iterable, *, step: int, active: bool):
        self._hooks = list(hooks)
        self.step = int(step)
        self.active = bool(active)
        self._metrics = {}

    def forward_start(self) -> None:
        self._call("on_forward_start", step=self.step, sync_gradients=True)

    def forward_end(self) -> None:
        self._call("on_forward_end")

    def before_gradient_clip(self, *, model, max_grad_norm: float) -> None:
        self._call(
            "on_before_gradient_clip",
            model=model,
            step=self.step,
            max_grad_norm=max_grad_norm,
        )

    def optimizer_step_end(self, *, model, optimizer) -> None:
        self._call(
            "on_optimizer_step_end",
            model=model,
            optimizer=optimizer,
            step=self.step,
        )

    def gather_metrics(self, accelerator, *, device) -> dict:
        if not self._metrics:
            return {}
        gathered = {}
        for key, value in self._metrics.items():
            tensor = torch.tensor(float(value), device=device, dtype=torch.float32).reshape(1)
            gathered[key] = float(accelerator.gather(tensor).mean().item())
        return gathered

    def _call(self, hook_name: str, **kwargs) -> None:
        if not self.active:
            return
        for hook in self._hooks:
            fn = getattr(hook, hook_name, None)
            if fn is None:
                continue
            result = fn(**kwargs)
            if result is None:
                continue
            if not isinstance(result, dict):
                raise TypeError(f"Train hook `{hook_name}` must return dict or None, got {type(result)}")
            self._metrics.update(result)

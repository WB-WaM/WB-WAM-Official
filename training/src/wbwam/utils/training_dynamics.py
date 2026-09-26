import math
from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from wbwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class TrainingDynamicsMonitor:
    """Lightweight training dynamics metrics for WBWAM training.

    The monitor is gated by step, so hooks do almost no work on non-logging steps.
    """

    def __init__(self, cfg: DictConfig, *, default_interval: int):
        self.enabled = bool(OmegaConf.select(cfg, "enabled", default=False))
        self.interval = int(default_interval)
        self.collect_activations = bool(OmegaConf.select(cfg, "activation.enabled", default=True))
        self.collect_grad = bool(OmegaConf.select(cfg, "grad.enabled", default=True))
        self.collect_optimizer = bool(OmegaConf.select(cfg, "optimizer.enabled", default=True))
        self.collect_weight = bool(OmegaConf.select(cfg, "weight.enabled", default=True))

        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._activation_metrics: Dict[str, float] = {}
        self._should_collect = False
        self._mot: Optional[nn.Module] = None
        self._clip_window: deque[float] = deque(maxlen=100)
        self._param_to_component: dict[int, str] = {}
        self._param_to_layer: dict[int, tuple[str, int]] = {}
        self._param_to_weight_kind: dict[int, str] = {}

    def install(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        self._index_model_params(model)
        self._install_activation_hooks(model)
        self._mot = getattr(model, "mot", None)
        if self.collect_activations and self._mot is not None:
            setattr(self._mot, "_training_dynamics_monitor", self)
        logger.info(
            "TrainingDynamicsMonitor enabled: interval=%d activation=%s grad=%s optimizer=%s weight=%s hooks=%d",
            self.interval,
            self.collect_activations,
            self.collect_grad,
            self.collect_optimizer,
            self.collect_weight,
            len(self._handles),
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._mot is not None and getattr(self._mot, "_training_dynamics_monitor", None) is self:
            setattr(self._mot, "_training_dynamics_monitor", None)
        self._mot = None

    def on_forward_start(self, step: int, *, sync_gradients: bool) -> None:
        self._activation_metrics.clear()
        self._should_collect = (
            self.enabled
            and sync_gradients
            and self.interval > 0
            and step % self.interval == 0
        )

    def on_forward_end(self) -> None:
        self._should_collect = False

    def record_residual(self, expert_name: str, layer_idx: int, hidden: torch.Tensor) -> None:
        if not self._should_collect or not self.collect_activations:
            return
        self._record_residual_stats(expert_name, layer_idx, hidden)

    def get_backward_metrics(
        self,
        model: nn.Module,
        *,
        step: int,
        max_grad_norm: float,
    ) -> Dict[str, float]:
        if not self.enabled or not self.collect_grad or self.interval <= 0 or step % self.interval != 0:
            return {}
        return self._grad_metrics(model, max_grad_norm=max_grad_norm)

    def on_before_gradient_clip(
        self,
        *,
        model: nn.Module,
        step: int,
        max_grad_norm: float,
    ) -> Dict[str, float]:
        return self.get_backward_metrics(
            model=model,
            step=step,
            max_grad_norm=max_grad_norm,
        )

    def get_step_metrics(self, model: nn.Module, optimizer, step: int) -> Dict[str, float]:
        if not self.enabled or self.interval <= 0 or step % self.interval != 0:
            return {}

        metrics: Dict[str, float] = {}
        metrics.update(self._activation_metrics)
        if self.collect_optimizer:
            metrics.update(self._optimizer_metrics(model, optimizer))
        if self.collect_weight:
            metrics.update(self._weight_metrics(model))
        return metrics

    def on_optimizer_step_end(
        self,
        *,
        model: nn.Module,
        optimizer,
        step: int,
    ) -> Dict[str, float]:
        return self.get_step_metrics(
            model=model,
            optimizer=optimizer,
            step=step,
        )

    def _install_activation_hooks(self, model: nn.Module) -> None:
        if not self.collect_activations:
            return
        for expert_name in ("video", "action"):
            expert = getattr(model, f"{expert_name}_expert", None)
            if expert is None:
                continue
            blocks = getattr(expert, "blocks", None)
            if blocks is None:
                continue
            for layer_idx, block in enumerate(blocks):
                self._handles.append(
                    block.register_forward_pre_hook(
                        self._make_residual_hook(expert_name, layer_idx)
                    )
                )

    def _make_residual_hook(self, expert_name: str, layer_idx: int):
        def hook(module, args):
            if not self._should_collect or not self.collect_activations:
                return
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                return
            self._record_residual_stats(expert_name, layer_idx, hidden)

        return hook

    def _record_residual_stats(self, expert_name: str, layer_idx: int, hidden: torch.Tensor) -> None:
        with torch.no_grad():
            x = hidden.detach().float()
            prefix = f"act/{expert_name}"
            self._activation_metrics[f"{prefix}/residual_norm_layer_{layer_idx}"] = float(
                x.norm(dim=-1).mean().item()
            )
            self._activation_metrics[f"{prefix}/nan_count_layer_{layer_idx}"] = float(
                torch.isnan(x).sum().item()
            )
            self._activation_metrics[f"{prefix}/inf_count_layer_{layer_idx}"] = float(
                torch.isinf(x).sum().item()
            )

            flat = x.reshape(-1, x.shape[-1])
            mean = flat.mean(dim=0)
            var = flat.var(dim=0, unbiased=False).clamp(min=1e-12)
            z = (flat - mean) / var.sqrt()
            kurtosis = z.pow(4).mean(dim=0).mean() - 3.0
            self._activation_metrics[f"activation/kurtosis_{expert_name}_layer_{layer_idx}"] = float(
                kurtosis.item()
            )

    def _grad_metrics(self, model: nn.Module, *, max_grad_norm: float) -> Dict[str, float]:
        component_sq: dict[str, torch.Tensor] = {}
        layer_sq: dict[tuple[str, int], torch.Tensor] = {}
        global_sq = None

        for param in model.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad_sq = param.grad.detach().float().square().sum()
            param_id = id(param)
            component = self._param_to_component.get(param_id, "other")
            component_sq[component] = self._add_square(component_sq, component, grad_sq)
            layer_key = self._param_to_layer.get(param_id)
            if layer_key is not None:
                layer_sq[layer_key] = self._add_square(layer_sq, layer_key, grad_sq)
            global_sq = grad_sq if global_sq is None else global_sq + grad_sq

        if global_sq is None:
            return {}

        metrics: Dict[str, float] = {}
        global_norm = float(global_sq.sqrt().item())
        metrics["grad/norm_global"] = global_norm
        metrics["grad/clip_ratio"] = global_norm / max(float(max_grad_norm), 1e-12)
        if global_norm > max_grad_norm and max_grad_norm > 0:
            metrics["grad/effective_utilization"] = float(max_grad_norm) / global_norm
            self._clip_window.append(1.0)
        else:
            metrics["grad/effective_utilization"] = 1.0
            self._clip_window.append(0.0)
        metrics["grad/frac_clipped_100"] = sum(self._clip_window) / max(len(self._clip_window), 1)

        for component, value in sorted(component_sq.items()):
            metrics[f"grad/norm_{component}"] = float(value.sqrt().item())
        for (component, layer_idx), value in sorted(layer_sq.items()):
            metrics[f"grad/norm_{component}_layer_{layer_idx}"] = float(value.sqrt().item())
        return metrics

    def _optimizer_metrics(self, model: nn.Module, optimizer) -> Dict[str, float]:
        param_sq: dict[str, torch.Tensor] = {}
        update_sq: dict[str, torch.Tensor] = {}
        param_count: dict[str, int] = {}
        v_sum: dict[str, torch.Tensor] = {}
        v_count: dict[str, int] = {}
        snr_sum: dict[str, torch.Tensor] = {}
        snr_count: dict[str, int] = {}
        second_moment_sum = None
        second_moment_count = 0
        second_moment_max = None
        effective_lr_max = None
        momentum_abs_sum = None
        momentum_abs_max = None
        momentum_count = 0

        for group in optimizer.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                if param not in optimizer.state:
                    continue
                state = optimizer.state[param]
                if "exp_avg" not in state or "exp_avg_sq" not in state:
                    continue

                step_t = state["step"]
                step_value = int(step_t.item()) if hasattr(step_t, "item") else int(step_t)
                if step_value < 1:
                    continue

                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()
                bias_correction1 = 1.0 - beta1 ** step_value
                bias_correction2 = 1.0 - beta2 ** step_value
                update = lr * (
                    (exp_avg / bias_correction1)
                    / ((exp_avg_sq / bias_correction2).sqrt() + eps)
                    + weight_decay * param.detach().float()
                )
                component = self._param_to_component.get(id(param), "other")
                update_norm_sq = update.square().sum()
                param_norm_sq = param.detach().float().square().sum()
                update_sq[component] = self._add_square(update_sq, component, update_norm_sq)
                param_sq[component] = self._add_square(param_sq, component, param_norm_sq)
                update_sq["global"] = self._add_square(update_sq, "global", update_norm_sq)
                param_sq["global"] = self._add_square(param_sq, "global", param_norm_sq)
                param_count[component] = self._add_count(param_count, component, param.numel())
                param_count["global"] = self._add_count(param_count, "global", param.numel())

                v_total = exp_avg_sq.sum()
                v_sum[component] = self._add_square(v_sum, component, v_total)
                v_sum["global"] = self._add_square(v_sum, "global", v_total)
                v_count[component] = self._add_count(v_count, component, exp_avg_sq.numel())
                v_count["global"] = self._add_count(v_count, "global", exp_avg_sq.numel())
                second_moment_sum = v_total if second_moment_sum is None else second_moment_sum + v_total
                second_moment_count += exp_avg_sq.numel()
                v_max = exp_avg_sq.max()
                second_moment_max = v_max if second_moment_max is None else torch.maximum(second_moment_max, v_max)
                effective_lr = lr / (v_max.sqrt() + eps)
                effective_lr_max = effective_lr if effective_lr_max is None else torch.maximum(effective_lr_max, effective_lr)

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                snr = (m_hat.abs() / (v_hat.sqrt() + eps)).sum()
                snr_sum[component] = self._add_square(snr_sum, component, snr)
                snr_sum["global"] = self._add_square(snr_sum, "global", snr)
                snr_count[component] = self._add_count(snr_count, component, param.numel())
                snr_count["global"] = self._add_count(snr_count, "global", param.numel())

                momentum_abs = exp_avg.abs()
                momentum_total = momentum_abs.sum()
                momentum_max = momentum_abs.max()
                momentum_abs_sum = momentum_total if momentum_abs_sum is None else momentum_abs_sum + momentum_total
                momentum_abs_max = momentum_max if momentum_abs_max is None else torch.maximum(momentum_abs_max, momentum_max)
                momentum_count += exp_avg.numel()

        metrics: Dict[str, float] = {}
        for component in sorted(param_sq):
            ratio = update_sq[component].sqrt() / param_sq[component].clamp(min=1e-24).sqrt()
            key = "optim/update_ratio" if component == "global" else f"optim/update_ratio_{component}"
            metrics[key] = float(ratio.item())
            if component != "global":
                theta_rms = param_sq[component].sqrt() / max(math.sqrt(param_count[component]), 1e-12)
                nominal = self._mean_lr(optimizer) / max(float(theta_rms.item()), 1e-12)
                metrics[f"optim/implicit_lr_drift_{component}"] = metrics[key] / max(nominal, 1e-12)

        for component in sorted(v_sum):
            mean = v_sum[component] / max(v_count[component], 1)
            key = "optim/v_mean" if component == "global" else f"optim/v_mean_{component}"
            metrics[key] = float(mean.item())
        for component in sorted(snr_sum):
            mean = snr_sum[component] / max(snr_count[component], 1)
            key = "optim/snr" if component == "global" else f"optim/snr_{component}"
            metrics[key] = float(mean.item())
        if second_moment_sum is not None and second_moment_max is not None:
            metrics["optim/second_moment_mean"] = float((second_moment_sum / max(second_moment_count, 1)).item())
            metrics["optim/second_moment_max"] = float(second_moment_max.item())
        if effective_lr_max is not None:
            metrics["optim/effective_lr_max"] = float(effective_lr_max.item())
        if momentum_abs_sum is not None and momentum_abs_max is not None and momentum_count > 0:
            momentum_mean = momentum_abs_sum / momentum_count
            metrics["spike/adam_momentum_spike"] = float((momentum_abs_max / momentum_mean.clamp(min=1e-12)).item())
        return metrics

    def _weight_metrics(self, model: nn.Module) -> Dict[str, float]:
        component_sq: dict[str, torch.Tensor] = {}
        layer_kind_sq: dict[tuple[str, int, str], torch.Tensor] = {}

        for param in model.parameters():
            if not param.requires_grad:
                continue
            value = param.detach().float()
            sq = value.square().sum()
            param_id = id(param)
            component = self._param_to_component.get(param_id, "other")
            component_sq[component] = self._add_square(component_sq, component, sq)

            layer_key = self._param_to_layer.get(param_id)
            kind = self._param_to_weight_kind.get(param_id)
            if layer_key is not None and kind is not None:
                metric_key = (layer_key[0], layer_key[1], kind)
                layer_kind_sq[metric_key] = self._add_square(layer_kind_sq, metric_key, sq)

        metrics: Dict[str, float] = {}
        for component, value in sorted(component_sq.items()):
            metrics[f"weight/norm_{component}_global"] = float(value.sqrt().item())
        for (component, layer_idx, kind), value in sorted(layer_kind_sq.items()):
            metrics[f"weight/norm_{component}_layer_{layer_idx}_{kind}"] = float(value.sqrt().item())
        return metrics

    def _index_model_params(self, model: nn.Module) -> None:
        self._param_to_component.clear()
        self._param_to_layer.clear()
        self._param_to_weight_kind.clear()

        for component in ("video", "action"):
            expert = getattr(model, f"{component}_expert", None)
            if expert is None:
                continue
            for param in expert.parameters():
                self._param_to_component[id(param)] = component

            blocks = getattr(expert, "blocks", None)
            if blocks is None:
                continue
            for layer_idx, block in enumerate(blocks):
                for param in block.parameters():
                    self._param_to_layer[id(param)] = (component, layer_idx)

                self_attn = getattr(block, "self_attn", None)
                if self_attn is not None:
                    for param in self_attn.parameters():
                        self._param_to_weight_kind[id(param)] = "attn"

                ffn = getattr(block, "ffn", None)
                if ffn is not None:
                    for param in ffn.parameters():
                        self._param_to_weight_kind[id(param)] = "ffn"

        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            for param in proprio_encoder.parameters():
                self._param_to_component[id(param)] = "proprio"

    @staticmethod
    def _add_square(bucket: dict, key, value: torch.Tensor) -> torch.Tensor:
        if key in bucket:
            return bucket[key] + value
        return value

    @staticmethod
    def _add_count(bucket: dict, key, value: int) -> int:
        if key in bucket:
            return bucket[key] + int(value)
        return int(value)

    @staticmethod
    def _mean_lr(optimizer) -> float:
        total = 0.0
        count = 0
        for group in optimizer.param_groups:
            total += float(group["lr"])
            count += 1
        return total / max(count, 1)

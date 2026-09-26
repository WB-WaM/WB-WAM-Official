import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as rt

_CUDA_DLLS_PRELOADED = False


def preload_onnx_cuda_dependencies() -> None:
    """Load pip-installed CUDA/cuDNN libraries before creating ORT sessions.

    ONNX Runtime can discover NVIDIA wheels in ``site-packages`` even when a
    direct ``conda run`` invocation does not populate ``LD_LIBRARY_PATH``.
    The operation is process-global and therefore only needs to run once.
    """
    global _CUDA_DLLS_PRELOADED
    if _CUDA_DLLS_PRELOADED:
        return
    preload = getattr(rt, "preload_dlls", None)
    if preload is not None:
        preload(cuda=True, cudnn=True, directory="")
    _CUDA_DLLS_PRELOADED = True


def _parse_device_id(device: str) -> int:
    if "cuda" in device and ":" in device:
        return int(device.split(":")[-1])
    return 0


def _get_onnx_providers(device: str):
    available = rt.get_available_providers()
    if "cuda" in device:
        preload_onnx_cuda_dependencies()
        device_id = _parse_device_id(device)
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        providers.append("CPUExecutionProvider")
        return providers
    return ["CPUExecutionProvider"]


# For small policy MLPs (~few hundred input dims), parallelism HURTS: thread-pool
# sync between layers dominates the actual matmul cost, and the workers get
# parked between calls when the loop sleeps to its control frequency.  Using a
# single intra-op thread + spinning eliminates wake-up jitter (from ~2 ms back
# to ~0.4 ms on Jetson Orin in a paced 50 Hz loop).  Override via env var
# ``HGPT_ORT_INTRA_THREADS`` if a larger model needs more parallelism.
_DEFAULT_INTRA = int(os.environ.get("HGPT_ORT_INTRA_THREADS", "1"))
_DEFAULT_INTER = int(os.environ.get("HGPT_ORT_INTER_THREADS", "1"))


def _make_session_options(
    intra_op: int = _DEFAULT_INTRA,
    inter_op: int = _DEFAULT_INTER,
) -> rt.SessionOptions:
    sess_opts = rt.SessionOptions()
    sess_opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.intra_op_num_threads = intra_op
    sess_opts.inter_op_num_threads = inter_op
    # Keep worker threads spinning between calls so a periodic 50 Hz loop does
    # not pay the futex_wake cost on every step.
    try:
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
        # Pre-empt the dynamic block scheduler that yields cores aggressively.
        sess_opts.add_session_config_entry("session.dynamic_block_base", "0")
    except Exception:
        pass
    return sess_opts


def _make_tensorrt_session(
    onnx_path: str,
    strict: bool = False,
    device_id: int = 0,
) -> rt.InferenceSession:
    """Build an ONNX Runtime TensorRT session. strict=True forbids fallback."""
    preload_onnx_cuda_dependencies()
    sess_opts = _make_session_options()

    trt_cache_dir = Path("storage/logs/trt_cache") / Path(onnx_path).stem
    trt_cache_dir.mkdir(parents=True, exist_ok=True)
    trt_provider = (
        "TensorrtExecutionProvider",
        {
            "device_id": device_id,
            "trt_fp16_enable": False,
            "trt_max_workspace_size": 4 * 1024**3,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(trt_cache_dir),
            "trt_builder_optimization_level": 3,
        },
    )

    available = rt.get_available_providers()
    if strict:
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                "strict_trt=True but TensorrtExecutionProvider is unavailable. "
                f"available={available}"
            )
        providers = [trt_provider]
    else:
        providers = []
        if "TensorrtExecutionProvider" in available:
            providers.append(trt_provider)
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        providers.append("CPUExecutionProvider")

    session = rt.InferenceSession(
        onnx_path, sess_options=sess_opts, providers=providers
    )
    actual = session.get_providers()
    print(f"[TensorRT] Loaded {Path(onnx_path).name}  providers={actual}")
    if strict and (len(actual) == 0 or actual[0] != "TensorrtExecutionProvider"):
        raise RuntimeError(
            f"strict_trt=True but TensorRT is not the active provider order: {actual}"
        )
    return session


def _warmup_onnx(session: rt.InferenceSession, n: int = 20):
    inputs = session.get_inputs()
    feeds = {}
    for inp in inputs:
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
    for _ in range(n):
        session.run(None, feeds)
    print(f"[ONNX] Warmup ({n} iters) done.")


@dataclass
class Args:
    policy_type: str = "mlp"  # "mlp" | "transformer"
    device: str = "cuda:0"
    load_path: str = ""


class _ONNXFastInfer:
    """Mixin that adds an IOBinding-based fast inference path.

    ``infer(obs)`` keeps the original behaviour (returns a fresh ndarray each
    call).  When the caller can re-use the same input buffer every step (the
    common case in deploy loops), call :py:meth:`bind_input_buffer` once, then
    use :py:meth:`infer_bound`, which skips ORT's per-call input copy +
    output allocation and writes into a stable output array.
    """

    onnx_model: rt.InferenceSession  # populated by subclass
    supports_host_iobinding = True

    def _post_init(self) -> None:
        # Cache I/O metadata
        self._inp_name = self.onnx_model.get_inputs()[0].name  # "obs"
        self._out_name = "continuous_actions"
        self._iobinding = None
        self._obs_buf: np.ndarray | None = None
        self._out_buf: np.ndarray | None = None

    def bind_input_buffer(self, obs_shape: tuple[int, ...]) -> np.ndarray:
        """Allocate a persistent (CPU) input buffer + IO binding.

        Returns the input buffer.  Caller is expected to write observations
        in-place into it before each :py:meth:`infer_bound` call.
        """
        out_shape = self.onnx_model.get_outputs()[0].shape
        out_shape = tuple(
            d if isinstance(d, int) and d > 0 else s
            for d, s in zip(out_shape, obs_shape, strict=False)
        )
        # Fallback: use known action dim 29 if shape is dynamic
        if any(
            not isinstance(d, int) or d <= 0
            for d in self.onnx_model.get_outputs()[0].shape
        ):
            out_shape = (obs_shape[0], 29)

        self._obs_buf = np.zeros(obs_shape, dtype=np.float32)
        self._out_buf = np.zeros(out_shape, dtype=np.float32)

        self._iobinding = self.onnx_model.io_binding()
        self._iobinding.bind_input(
            name=self._inp_name,
            device_type="cpu",
            device_id=0,
            element_type=np.float32,
            shape=obs_shape,
            buffer_ptr=self._obs_buf.ctypes.data,
        )
        self._iobinding.bind_output(
            name=self._out_name,
            device_type="cpu",
            device_id=0,
            element_type=np.float32,
            shape=out_shape,
            buffer_ptr=self._out_buf.ctypes.data,
        )
        return self._obs_buf

    def infer_bound(self) -> np.ndarray:
        """Run inference using the pre-bound IOBinding buffers.

        ``self._obs_buf`` must already contain the observation when this is
        called.  Returns the output buffer (do NOT mutate before next infer).
        """
        self.onnx_model.run_with_iobinding(self._iobinding)
        return self._out_buf


class MLP_Policy_ONNX(_ONNXFastInfer):
    def __init__(self, config):
        providers = _get_onnx_providers(getattr(config, "device", "cuda"))
        sess_opts = _make_session_options()
        self.onnx_model = rt.InferenceSession(
            config.load_path,
            sess_options=sess_opts,
            providers=providers,
        )
        print(
            f"[ONNX] Loaded {config.load_path}  "
            f"providers={self.onnx_model.get_providers()}  "
            f"intra={sess_opts.intra_op_num_threads} "
            f"inter={sess_opts.inter_op_num_threads}"
        )
        _warmup_onnx(self.onnx_model)
        self._post_init()

    def infer(self, obs: np.ndarray) -> np.ndarray:
        nn_action = self.onnx_model.run(["continuous_actions"], {"obs": obs})[0]
        return nn_action


class MLP_Policy_TensorRT(_ONNXFastInfer):
    # ORT TensorRT 1.23 can cache the first value of a persistently bound
    # CPU input buffer. The observation changes in memory, but subsequent
    # run_with_iobinding() calls return the first action. Keep TensorRT as
    # the execution provider while using session.run(), which recopies the
    # current host observation every cycle.
    supports_host_iobinding = False

    def __init__(self, config, strict_trt: bool = False):
        self.onnx_model = _make_tensorrt_session(
            config.load_path,
            strict=strict_trt,
            device_id=_parse_device_id(getattr(config, "device", "cuda")),
        )
        _warmup_onnx(self.onnx_model)
        self._post_init()

    def infer(self, obs: np.ndarray) -> np.ndarray:
        nn_action = self.onnx_model.run(["continuous_actions"], {"obs": obs})[0]
        return nn_action


# Transformer ONNX shares the "obs" -> "continuous_actions" contract; only
# obs shape differs (B, K, D) vs (B, D), so we reuse the MLP plumbing.
class Transformer_Policy_ONNX(MLP_Policy_ONNX):
    pass


class Transformer_Policy_TensorRT(MLP_Policy_TensorRT):
    pass


def get_policy_onnx(
    args,
    use_trt: bool = False,
    strict_trt: bool = False,
):
    if args.policy_type == "mlp":
        if use_trt:
            return MLP_Policy_TensorRT(args, strict_trt=strict_trt)
        return MLP_Policy_ONNX(args)
    if args.policy_type == "transformer":
        if use_trt:
            return Transformer_Policy_TensorRT(args, strict_trt=strict_trt)
        return Transformer_Policy_ONNX(args)
    raise ValueError(f"Unknown policy type: {args.policy_type}")

"""Extract hidden states from intermediate layers during prefill using vLLM."""

import inspect
import uuid
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_kv_cache_groups,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from speculators.utils.util import empty_cache, is_npu_available, mem_get_info

from .cuda_ipc import CudaIpcImporter
from .logging_utils import PipelineLogger

__all__ = ["VllmHiddenStatesGenerator"]

# Constants
CACHE_MEMORY_FRACTION = 0.2  # Fraction of GPU memory for KV cache
VLLM_BLOCK_SIZE = 128 if is_npu_available() else 16  # Block size for KV cache
MAX_NUM_SEQS = 32  # Maximum sequences for prefill-only workload
MIN_MAX_BATCHED_TOKENS = 8192  # Minimum batched tokens threshold
MAX_DECODE_TOKENS = 1  # Maximum tokens to generate (prefill only)
SAMPLING_TEMPERATURE = 0.0  # Temperature for sampling (greedy)
INITIAL_ARRIVAL_TIME = 0.0  # Initial request arrival time

log = PipelineLogger(__name__)

_REQUEST_ACCEPTS_EOS_TOKEN_ID = "eos_token_id" in inspect.signature(
    Request
).parameters
_SAMPLING_PARAMS_ACCEPTS_PRIVATE_EOS_TOKEN_ID = "_eos_token_id" in inspect.signature(
    SamplingParams
).parameters
_VLLM_CONFIG_OVERRIDE_TARGETS = {
    "model": ModelConfig,
    "cache": CacheConfig,
    "parallel": ParallelConfig,
    "scheduler": SchedulerConfig,
    "device": DeviceConfig,
    "load": LoadConfig,
    "vllm": VllmConfig,
    "scheduler_init": Scheduler,
}
_PROTECTED_VLLM_CONFIG_OVERRIDE_KEYS = {
    "model": {"model", "tokenizer", "max_model_len"},
    "cache": {"enable_prefix_caching"},
    "parallel": {
        "tensor_parallel_size",
        "enable_expert_parallel",
        "worker_extension_cls",
    },
    "scheduler": {"max_model_len", "is_encoder_decoder"},
    "vllm": {
        "model_config",
        "cache_config",
        "parallel_config",
        "scheduler_config",
        "device_config",
        "load_config",
    },
    "scheduler_init": {
        "vllm_config",
        "kv_cache_config",
        "structured_output_manager",
        "block_size",
    },
}


def _normalize_vllm_config_overrides(
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if overrides is None:
        return {}

    unexpected_targets = set(overrides) - set(_VLLM_CONFIG_OVERRIDE_TARGETS)
    if unexpected_targets:
        raise ValueError(
            "Unexpected vLLM config override target(s): "
            f"{sorted(unexpected_targets)}. Expected one of "
            f"{sorted(_VLLM_CONFIG_OVERRIDE_TARGETS)}."
        )

    normalized = {}
    for target, kwargs in overrides.items():
        if not isinstance(kwargs, dict):
            raise TypeError(
                "vLLM config override values must be dictionaries. "
                f"Got {type(kwargs).__name__} for target {target!r}."
            )
        normalized[target] = dict(kwargs)
    return normalized


def _validate_vllm_config_override_kwargs(
    target: str,
    kwargs: dict[str, Any],
) -> None:
    target_cls = _VLLM_CONFIG_OVERRIDE_TARGETS[target]
    protected_keys = _PROTECTED_VLLM_CONFIG_OVERRIDE_KEYS.get(target, set()) & set(
        kwargs
    )
    if protected_keys:
        raise ValueError(
            "vLLM config override target "
            f"{target!r} cannot override hidden-state generator owned key(s): "
            f"{sorted(protected_keys)}."
        )

    valid_keys = set(inspect.signature(target_cls).parameters)
    unexpected_keys = set(kwargs) - valid_keys
    if unexpected_keys:
        raise ValueError(
            "Unexpected vLLM config override key(s) for "
            f"{target!r}: {sorted(unexpected_keys)}. Expected one of "
            f"{sorted(valid_keys)}."
        )


def _make_sampling_params(eos_token_id: int | None) -> SamplingParams:
    sampling_kwargs = {
        "max_tokens": MAX_DECODE_TOKENS,
        "temperature": SAMPLING_TEMPERATURE,
    }
    if eos_token_id is not None and _SAMPLING_PARAMS_ACCEPTS_PRIVATE_EOS_TOKEN_ID:
        sampling_kwargs["_eos_token_id"] = eos_token_id
    return SamplingParams(**sampling_kwargs)


def _make_prefill_request(
    request_id: str,
    prompt_token_ids: list[int],
    eos_token_id: int | None,
    block_hasher,
) -> Request:
    req_kwargs = {
        "request_id": request_id,
        "prompt_token_ids": prompt_token_ids,
        "sampling_params": _make_sampling_params(eos_token_id),
        "pooling_params": None,
        "arrival_time": INITIAL_ARRIVAL_TIME,
        "block_hasher": block_hasher,
    }
    if eos_token_id is not None and _REQUEST_ACCEPTS_EOS_TOKEN_ID:
        req_kwargs["eos_token_id"] = eos_token_id
    return Request(**req_kwargs)


def _get_kv_cache_groups_for_scheduler(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    return get_kv_cache_groups(vllm_config, kv_cache_spec)


def _get_kv_cache_configs_for_scheduler(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> tuple[Any, Any]:
    kv_cache_configs = get_kv_cache_configs(
        vllm_config=vllm_config,
        kv_cache_specs=kv_cache_specs,
        available_memory=available_memory,
    )
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    return kv_cache_configs, scheduler_kv_cache_config


class VllmHiddenStatesGenerator:
    """Extracts hidden states from intermediate layers during prefill only.

    This module provides a generator for extracting hidden states from
    transformer models during the prefill phase using VLLM's inference engine.
    It is designed for generating training data for speculative decoding models
    like EAGLE3.

    The generator:
    - Uses VLLM's multiprocess executor for efficient batch inference
    - Patches model forward pass to capture intermediate layer hidden states
    - Operates in prefill-only mode (max_tokens=1) for data generation
    - Supports tensor parallelism for large models
    - Automatically manages KV cache and memory allocation

    Example:
        generator = VllmHiddenStatesGenerator(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            layer_ids=[10, 20, 30],
            tensor_parallel_size=2
        )

        results = generator.generate(token_ids)
        for result in results:
            input_ids = result["input_ids"]
            hidden_states = result["hidden_states"]  # List of tensors per layer`
    """

    def __init__(  # noqa: PLR0915
        self,
        model_path: str,
        layer_ids: list[int] | None = None,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
        kv_cache_dtype: str = "auto",
        expert_parallel_size: int | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int = MAX_NUM_SEQS,
        max_batched_tokens: int = MIN_MAX_BATCHED_TOKENS,
        vllm_config_overrides: dict[str, dict[str, Any]] | None = None,
        output_device: str = "cpu",
    ):
        self.vllm_config_overrides = _normalize_vllm_config_overrides(
            vllm_config_overrides
        )
        self._validate_parallel_sizes(
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
        )
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.kv_cache_dtype = kv_cache_dtype
        self.expert_parallel_size = expert_parallel_size
        self._request_counter = 0
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.output_device = output_device
        self._use_torch_cuda_ipc = torch.device(output_device).type == "cuda"

        log.info(f"Initializing hidden states generator for {model_path}")
        log.info(f"Tensor parallel size: {tensor_parallel_size}")
        log.info(f"KV cache dtype: {kv_cache_dtype}")
        if expert_parallel_size is None:
            log.info("Expert parallel size: disabled")
        else:
            log.info(f"Expert parallel size: {expert_parallel_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, "num_hidden_layers"):
            num_layers = config.num_hidden_layers
        elif hasattr(config, "text_config"):
            num_layers = config.text_config.num_hidden_layers
        else:
            raise ValueError("Cannot determine num_layers from config")

        log.info(f"Model has {num_layers} layers")

        if layer_ids is None:
            self.layer_ids = [2, num_layers // 2, num_layers - 3, num_layers - 1]
            log.info(
                f"Auto-selected layers: {self.layer_ids} "
                f"(from {num_layers} total layers)"
            )
        else:
            self.layer_ids = layer_ids
            log.info(f"Using specified layers: {layer_ids}")

        for layer_id in self.layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(
                    f"Layer index {layer_id} out of bounds [0, {num_layers - 1}]"
                )

        self.vllm_config = self._create_vllm_config(
            model_path=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            kv_cache_dtype=kv_cache_dtype,
            expert_parallel_size=expert_parallel_size,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        self.block_size = self.vllm_config.cache_config.block_size

        log.info("Initializing executor...")
        self.executor = MultiprocExecutor(vllm_config=self.vllm_config)

        log.info("Setting up hidden states capture...")
        self._setup_capture()

        log.info("Creating scheduler...")
        kv_cache_spec_list = self.executor.collective_rpc("get_kv_cache_spec")

        free_memory, _ = mem_get_info()
        cache_memory = int(free_memory * gpu_memory_utilization * CACHE_MEMORY_FRACTION)

        kv_cache_configs, scheduler_kv_cache_config = (
            _get_kv_cache_configs_for_scheduler(
                vllm_config=self.vllm_config,
                kv_cache_specs=kv_cache_spec_list,
                available_memory=[cache_memory] * len(kv_cache_spec_list),
            )
        )

        self.vllm_config.cache_config.num_gpu_blocks = (
            scheduler_kv_cache_config.num_blocks
        )
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            self.vllm_config.cache_config.block_size = min(
                group.kv_cache_spec.block_size for group in kv_cache_groups
            )
        self.block_size = self.vllm_config.cache_config.block_size
        self.vllm_config.validate_block_size()

        structured_output_manager = StructuredOutputManager(
            vllm_config=self.vllm_config
        )

        scheduler_kwargs = {
            "vllm_config": self.vllm_config,
            "kv_cache_config": scheduler_kv_cache_config,
            "structured_output_manager": structured_output_manager,
            "block_size": self.vllm_config.cache_config.block_size,
        }
        scheduler_kwargs.update(self._get_vllm_config_override_kwargs("scheduler_init"))
        self.scheduler = Scheduler(**scheduler_kwargs)

        log.info("Initializing KV cache on all workers...")
        self.executor.initialize_from_config(kv_cache_configs)

        # Create block hasher for request KV cache management
        # Following vLLM's pattern in v1/engine/core.py
        caching_hash_fn = get_hash_fn_by_name(
            self.vllm_config.cache_config.prefix_caching_hash_algo
        )
        init_none_hash(caching_hash_fn)

        self.block_hasher = get_request_block_hasher(
            self.vllm_config.cache_config.block_size,
            caching_hash_fn,
        )

    @staticmethod
    def _validate_parallel_sizes(
        tensor_parallel_size: int,
        expert_parallel_size: int | None,
    ) -> None:
        if tensor_parallel_size < 1:
            raise ValueError(
                "tensor_parallel_size must be >= 1. "
                f"Got {tensor_parallel_size}."
            )
        if expert_parallel_size is None:
            return
        if expert_parallel_size < 1:
            raise ValueError(
                "expert_parallel_size must be >= 1 when provided. "
                f"Got {expert_parallel_size}."
            )
        if expert_parallel_size > 1 and expert_parallel_size != tensor_parallel_size:
            raise ValueError(
                "VllmHiddenStatesGenerator currently maps expert parallelism "
                "onto the tensor-parallel worker group, so expert_parallel_size "
                "must match tensor_parallel_size when it is greater than 1. "
                f"Got expert_parallel_size={expert_parallel_size} and "
                f"tensor_parallel_size={tensor_parallel_size}."
            )

    def _get_vllm_config_override_kwargs(self, target: str) -> dict[str, Any]:
        overrides = getattr(self, "vllm_config_overrides", {})
        kwargs = dict(overrides.get(target, {}))
        _validate_vllm_config_override_kwargs(target, kwargs)
        return kwargs

    def _create_vllm_config(
        self,
        model_path: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        kv_cache_dtype: str,
        expert_parallel_size: int | None,
        max_num_batched_tokens: int | None = None,
    ) -> VllmConfig:
        """Create VllmConfig with hidden states worker extension"""
        cache_config_kwargs = {
            "block_size": VLLM_BLOCK_SIZE,
            "gpu_memory_utilization": gpu_memory_utilization,
            "cache_dtype": kv_cache_dtype,
            "enable_prefix_caching": False,
        }
        cache_config_kwargs.update(self._get_vllm_config_override_kwargs("cache"))
        cache_config = CacheConfig(**cache_config_kwargs)

        # For prefill-only workloads, use conservative scheduler limits
        # to reduce warmup memory allocation. max_num_seqs controls the
        # warmup allocation size (see gpu_worker.py:441-444).
        # We set it to a small value since we only do prefill in batches.
        max_num_seqs = self.max_num_seqs
        if not max_num_batched_tokens:
            max_num_batched_tokens = max(self.max_batched_tokens, max_model_len)

        model_config_kwargs = {
            "model": model_path,
            "tokenizer": model_path,
            "trust_remote_code": True,
            "dtype": "auto",
            "max_model_len": max_model_len,
            "enforce_eager": True,
        }
        model_config_kwargs.update(self._get_vllm_config_override_kwargs("model"))

        parallel_config_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "enable_expert_parallel": expert_parallel_size not in (None, 1),
            "worker_extension_cls": (
                "speculators.data_generation.custom_worker."
                "HiddenStatesWorkerExtension"
            ),
        }
        parallel_config_kwargs.update(
            self._get_vllm_config_override_kwargs("parallel")
        )

        scheduler_config_kwargs = {
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "is_encoder_decoder": False,
        }
        scheduler_config_kwargs.update(
            self._get_vllm_config_override_kwargs("scheduler")
        )

        vllm_config_kwargs = {
            "model_config": ModelConfig(**model_config_kwargs),
            "cache_config": cache_config,
            "parallel_config": ParallelConfig(**parallel_config_kwargs),
            "scheduler_config": SchedulerConfig(**scheduler_config_kwargs),
            "device_config": DeviceConfig(
                **self._get_vllm_config_override_kwargs("device")
            ),
            "load_config": LoadConfig(
                **self._get_vllm_config_override_kwargs("load")
            ),
        }
        vllm_config_kwargs.update(self._get_vllm_config_override_kwargs("vllm"))
        return VllmConfig(**vllm_config_kwargs)

    def _setup_capture(self):
        self.executor.collective_rpc(
            "_setup_hidden_states_capture",
            args=(self.layer_ids,),
        )

    def generate(self, token_ids: list[list[int]] | torch.Tensor) -> list[dict]:  # noqa: PLR0912, PLR0915
        """Extract hidden states from prefill phase only.

        Args:
            token_ids: Batch of token ID sequences as list[list[int]] or Tensor

        Returns:
            List of dicts with keys: input_ids, hidden_states, loss_mask
        """
        if isinstance(token_ids, torch.Tensor):
            input_ids_list = token_ids.tolist()
        else:
            if not token_ids:
                raise ValueError("token_ids cannot be empty")
            input_ids_list = token_ids

        log.debug(f"Generating hidden states for {len(input_ids_list)} sequences")
        # Account for max_tokens=1 in sampling params
        # (vLLM enforces: len(prompt) + max_tokens <= max_model_len)
        max_len = self.vllm_config.model_config.max_model_len - MAX_DECODE_TOKENS
        input_ids_list = [ids[:max_len] for ids in input_ids_list]

        # Track request IDs and prompt lengths for proper token attribution
        request_id_to_idx = {}
        request_id_to_prompt_len = {}

        for i, ids in enumerate(input_ids_list):
            # Ensure ids is a list (not tensor) for vLLM Request
            ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else ids
            req_id = f"req_{self._request_counter}_{i}"
            request_id_to_idx[req_id] = i
            request_id_to_prompt_len[req_id] = len(ids_list)

            req = _make_prefill_request(
                request_id=req_id,
                prompt_token_ids=ids_list,
                eos_token_id=self.tokenizer.eos_token_id,
                block_hasher=self.block_hasher,
            )
            self.scheduler.add_request(req)

        # Increment to ensure unique request IDs across calls
        # (prevents KV cache corruption with delayed block freeing)
        self._request_counter += 1
        self.executor.collective_rpc("_reset_capture")

        # Track progress for each request to distinguish prefill from decode
        request_num_computed = dict.fromkeys(request_id_to_idx, 0)
        schedule_iterations = 0
        all_prefill_complete = False

        while (
            scheduler_output := self.scheduler.schedule()
        ).total_num_scheduled_tokens > 0 and not all_prefill_complete:
            schedule_iterations += 1

            # Calculate prefill tokens for each request (ignore decode tokens)
            prefill_metadata = {}
            for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
                num_already_computed = request_num_computed[req_id]
                num_prompt = request_id_to_prompt_len[req_id]
                num_prefill = max(0, min(num_tokens, num_prompt - num_already_computed))

                if num_prefill > 0:
                    prefill_metadata[req_id] = num_prefill

                request_num_computed[req_id] += num_tokens

            all_prefill_complete = all(
                request_num_computed[req_id] >= request_id_to_prompt_len[req_id]
                for req_id in request_id_to_idx
            )

            if prefill_metadata:
                self.executor.collective_rpc(
                    "_set_request_metadata", args=(prefill_metadata,)
                )

            model_output = self.executor.execute_model(scheduler_output)
            self.executor.sample_tokens(model_output)

        # Abort all requests (prefill complete, don't need decode)
        self.scheduler.finish_requests(
            list(request_id_to_idx.keys()), RequestStatus.FINISHED_ABORTED
        )

        capture_token = None
        imported_devices: set[int] = set()
        if self._use_torch_cuda_ipc:
            # Get captured states organized by request ID using torch CUDA IPC.
            capture_token = (
                f"capture_{self._request_counter - 1}_{uuid.uuid4().hex[:8]}"
            )
            request_states_payload = self.executor.collective_rpc(
                "_get_captured_states",
                args=(capture_token,),
                unique_reply_rank=0,
            )
            if not request_states_payload:
                raise RuntimeError("Failed to capture hidden states from worker")
            if not (
                isinstance(request_states_payload, dict)
                and request_states_payload.get("transport") == "torch_cuda_ipc"
            ):
                raise RuntimeError(
                    "Unexpected captured states transport payload for CUDA output: "
                    f"{type(request_states_payload).__name__}"
                )
            request_states_dict, imported_devices = CudaIpcImporter.open_capture(
                request_states_payload
            )
        else:
            # Non-CUDA output devices use vLLM collective_rpc's standard transport.
            request_states_dict = self.executor.collective_rpc(
                "_get_captured_states",
                unique_reply_rank=0,
            )
            if not request_states_dict:
                raise RuntimeError("Failed to capture hidden states from worker")
            if not isinstance(request_states_dict, dict):
                raise RuntimeError(
                    "Unexpected captured states payload for non-CUDA output: "
                    f"{type(request_states_dict).__name__}"
                )

        log.debug(f"Captured states for {len(request_states_dict)} requests")
        try:
            # Map results back to original input order.
            results = [None] * len(input_ids_list)
            for req_id, i in request_id_to_idx.items():

                if req_id not in request_states_dict:
                    raise RuntimeError(
                        f"Request {req_id} not found in captured states. "
                        f"Available: {list(request_states_dict.keys())}"
                    )

                layer_states = []
                for h in request_states_dict[req_id]:
                    if h.device == torch.device(self.output_device):
                        # Clone when staying on the same device to decouple from
                        # IPC-backed or shared storage.
                        layer_states.append(h.clone())
                    else:
                        # Avoid an extra clone when transferring between devices.
                        layer_states.append(h.to(self.output_device))
                input_ids_tensor = torch.as_tensor(
                    input_ids_list[i],
                    dtype=torch.long,
                ).to(self.output_device)

                results[i] = {
                    "input_ids": input_ids_tensor,
                    "hidden_states": layer_states,
                    "loss_mask": None,
                }

            if any(result is None for result in results):
                missing_indices = [
                    idx for idx, result in enumerate(results) if result is None
                ]
                raise RuntimeError(
                    f"Missing hidden-state results for batch indices: {missing_indices}"
                )
        finally:
            if imported_devices:
                for device_index in imported_devices:
                    torch.cuda.synchronize(device=device_index)
            if capture_token is not None:
                self.executor.collective_rpc(
                    "_release_ipc_capture", args=(capture_token,), unique_reply_rank=0
                )

        empty_cache()
        return results

    def __del__(self):
        if hasattr(self, "executor"):
            try:
                self.executor.shutdown()
            except Exception:
                log.warning("Exception during executor shutdown")

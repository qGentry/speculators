"""Tests for vLLM hidden states generator accuracy against HuggingFace baseline."""

import gc
import logging
import os
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_spec,
    unify_hybrid_kv_cache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from speculators.data_generation import VllmHiddenStatesGenerator, custom_worker
from speculators.data_generation.custom_worker import HiddenStatesWorkerExtension
from speculators.data_generation.vllm_hidden_states_generator import (
    _REQUEST_ACCEPTS_EOS_TOKEN_ID,
    _SAMPLING_PARAMS_ACCEPTS_PRIVATE_EOS_TOKEN_ID,
    _get_kv_cache_configs_for_scheduler,
    _get_kv_cache_groups_for_scheduler,
    _make_prefill_request,
)

logger = logging.getLogger(__name__)

# Set vLLM multiprocessing method to spawn for CUDA compatibility
# Must be set before vLLM imports to avoid CUDA re-initialization errors
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def test_make_prefill_request_is_compatible_with_installed_vllm():
    req = _make_prefill_request(
        request_id="req_test",
        prompt_token_ids=[1, 2, 3],
        eos_token_id=42,
        block_hasher=None,
    )

    assert req.request_id == "req_test"
    assert req.prompt_token_ids == [1, 2, 3]
    assert req.max_tokens == 1
    if _SAMPLING_PARAMS_ACCEPTS_PRIVATE_EOS_TOKEN_ID:
        assert req.sampling_params._eos_token_id == 42
    if not _REQUEST_ACCEPTS_EOS_TOKEN_ID:
        assert "eos_token_id" not in req.__dict__


def test_validate_parallel_sizes_accepts_matching_expert_parallel_size():
    VllmHiddenStatesGenerator._validate_parallel_sizes(
        tensor_parallel_size=4,
        expert_parallel_size=4,
    )


@pytest.mark.parametrize("expert_parallel_size", [0, -1])
def test_validate_parallel_sizes_rejects_invalid_expert_parallel_size(
    expert_parallel_size,
):
    with pytest.raises(
        ValueError, match="expert_parallel_size must be >= 1 when provided"
    ):
        VllmHiddenStatesGenerator._validate_parallel_sizes(
            tensor_parallel_size=4,
            expert_parallel_size=expert_parallel_size,
        )


def test_validate_parallel_sizes_rejects_mismatched_expert_parallel_size():
    with pytest.raises(
        ValueError, match="must match tensor_parallel_size when it is greater than 1"
    ):
        VllmHiddenStatesGenerator._validate_parallel_sizes(
            tensor_parallel_size=8,
            expert_parallel_size=4,
        )


def test_create_vllm_config_forwards_kv_cache_dtype_and_expert_parallel():
    generator = object.__new__(VllmHiddenStatesGenerator)
    generator.max_num_seqs = 8
    generator.max_batched_tokens = 512

    with (
        mock.patch(
            "speculators.data_generation.vllm_hidden_states_generator.CacheConfig"
        ) as cache_config_cls,
        mock.patch(
            "speculators.data_generation.vllm_hidden_states_generator.ModelConfig"
        ) as model_config_cls,
        mock.patch(
            "speculators.data_generation.vllm_hidden_states_generator.ParallelConfig"
        ) as parallel_config_cls,
        mock.patch(
            "speculators.data_generation.vllm_hidden_states_generator.SchedulerConfig"
        ) as scheduler_config_cls,
        mock.patch(
            "speculators.data_generation.vllm_hidden_states_generator.DeviceConfig"
        ) as device_config_cls,
        mock.patch(
            "speculators.data_generation.vllm_hidden_states_generator.LoadConfig"
        ) as load_config_cls,
        mock.patch(
            "speculators.data_generation.vllm_hidden_states_generator.VllmConfig"
        ) as vllm_config_cls,
    ):
        cache_config = mock.sentinel.cache_config
        model_config = mock.sentinel.model_config
        parallel_config = mock.sentinel.parallel_config
        scheduler_config = mock.sentinel.scheduler_config
        device_config = mock.sentinel.device_config
        load_config = mock.sentinel.load_config
        vllm_config = mock.sentinel.vllm_config

        cache_config_cls.return_value = cache_config
        model_config_cls.return_value = model_config
        parallel_config_cls.return_value = parallel_config
        scheduler_config_cls.return_value = scheduler_config
        device_config_cls.return_value = device_config
        load_config_cls.return_value = load_config
        vllm_config_cls.return_value = vllm_config

        result = generator._create_vllm_config(
            model_path="Qwen/Qwen2-0.5B",
            max_model_len=2048,
            gpu_memory_utilization=0.3,
            tensor_parallel_size=2,
            kv_cache_dtype="fp8",
            expert_parallel_size=2,
        )

    assert result is vllm_config
    assert cache_config_cls.call_args.kwargs["cache_dtype"] == "fp8"
    assert parallel_config_cls.call_args.kwargs["tensor_parallel_size"] == 2
    assert parallel_config_cls.call_args.kwargs["enable_expert_parallel"] is True


def test_hidden_states_extension_uses_deepseek_v4_post_norm_forward(monkeypatch):
    class FakeLayer(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment

        def forward(self, hidden_states, positions, input_ids):
            return hidden_states + self.increment

    class DeepseekV4Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="deepseek_v4")
            self.hc_mult = 1
            self.start_layer = 0
            self.end_layer = 3
            self.layers = torch.nn.ModuleList(
                [FakeLayer(10.0), FakeLayer(20.0), FakeLayer(30.0)]
            )
            self._mtp_hidden_buffer = torch.zeros(8, 1)

        def embed_input_ids(self, input_ids):
            return input_ids.to(torch.float32).unsqueeze(-1)

    base_model = DeepseekV4Model()
    extension = HiddenStatesWorkerExtension()
    extension.model_runner = SimpleNamespace(model=SimpleNamespace(model=base_model))
    monkeypatch.setattr(
        custom_worker,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    monkeypatch.setattr(
        custom_worker,
        "_deepseek_v4_post_norm_hidden_states",
        lambda _base_model, hidden_states: hidden_states.squeeze(-2) + 1000.0,
    )

    extension._setup_hidden_states_capture([0, 2])
    output = base_model.forward(
        input_ids=torch.tensor([1.0, 2.0]),
        positions=torch.tensor([0, 1]),
    )

    assert base_model.forward.__func__ is custom_worker._patched_deepseek_v4_forward
    assert torch.equal(output, torch.tensor([[1061.0], [1062.0]]))
    assert len(extension._captured_states) == 2
    assert torch.equal(
        extension._captured_states[0][0],
        torch.tensor([[1011.0], [1012.0]]),
    )
    assert torch.equal(
        extension._captured_states[1][0],
        torch.tensor([[1061.0], [1062.0]]),
    )


def _make_attention_mamba_kv_cache_spec():
    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
    )
    return {
        "layers.0.self_attn": full_attention_spec,
        "layers.1.mamba": MambaSpec(
            block_size=16,
            shapes=((16, 128),),
            dtypes=(torch.float16,),
            page_size_padded=full_attention_spec.page_size_bytes,
        ),
    }


def _force_uniform_kv_cache_grouping(kv_cache_spec):
    unify_hybrid_kv_cache_specs(kv_cache_spec)
    return _get_kv_cache_groups_uniform_spec(kv_cache_spec)


def test_forced_uniform_kv_cache_grouping_reproduces_attention_mamba_failure():
    with pytest.raises(ValueError, match="failed to convert the KV cache specs"):
        _force_uniform_kv_cache_grouping(_make_attention_mamba_kv_cache_spec())


def test_kv_cache_grouping_supports_attention_mamba_without_loading_weights():
    vllm_config = VllmConfig(
        cache_config=CacheConfig(block_size=16),
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=False,
        ),
    )

    kv_cache_groups = _get_kv_cache_groups_for_scheduler(
        vllm_config,
        _make_attention_mamba_kv_cache_spec(),
    )

    assert len(kv_cache_groups) == 2
    assert {type(group.kv_cache_spec) for group in kv_cache_groups} == {
        FullAttentionSpec,
        MambaSpec,
    }
    assert {group.kv_cache_spec.page_size_bytes for group in kv_cache_groups} == {
        65536
    }


def test_scheduler_kv_cache_config_projects_uniform_type_specs():
    vllm_config = VllmConfig(
        cache_config=CacheConfig(block_size=16),
        parallel_config=ParallelConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=False,
        ),
    )
    vllm_config.model_config = SimpleNamespace(
        original_max_model_len=128,
        max_model_len=128,
    )

    kv_cache_specs = [
        {
            "layers.0.self_attn": FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.float16,
            ),
            "layers.1.self_attn": FullAttentionSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=16,
                dtype=torch.float16,
            ),
        }
    ]

    kv_cache_configs, scheduler_kv_cache_config = _get_kv_cache_configs_for_scheduler(
        vllm_config=vllm_config,
        kv_cache_specs=kv_cache_specs,
        available_memory=[1024 * 1024 * 1024],
    )

    assert isinstance(
        kv_cache_configs[0].kv_cache_groups[0].kv_cache_spec,
        UniformTypeKVCacheSpecs,
    )
    assert isinstance(
        scheduler_kv_cache_config.kv_cache_groups[0].kv_cache_spec,
        FullAttentionSpec,
    )


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Fixture to clean up GPU memory before and after each test."""
    # Cleanup before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    yield  # Run the test

    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1)  # Give time for cleanup


@pytest.mark.regression
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("model_path", "tensor_parallel_size"),
    [
        ("Qwen/Qwen2-0.5B", 1),
    ],
)
def test_vllm_vs_huggingface_accuracy(model_path, tensor_parallel_size):
    """Test vLLM hidden states match HuggingFace baseline within tolerance."""

    test_prompts = [
        (
            "The future of artificial intelligence is bright and full "
            "of possibilities that will transform humanity."
        ),
        (
            "In a world where technology advances rapidly, we must "
            "carefully consider the ethical implications."
        ),
    ]

    logger.info("=" * 80)
    logger.info(f"Testing {model_path}")
    logger.info(f"Prompts: {len(test_prompts)}")
    logger.info("=" * 80)

    # HuggingFace baseline Implementation, adapted from research/eagle3/ge_data
    logger.info("[1/2] Running HuggingFace baseline...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")  # type: ignore[arg-type]
    num_layers = len(hf_model.model.layers)
    logger.info(f"Model has {num_layers} layers")

    inputs = tokenizer(
        test_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(hf_model.device)
    logger.info(f"Input shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        hf_output = hf_model(**inputs, output_hidden_states=True)

    # Extract layers using EAGLE3 pattern
    # Feature fusion: layers 2, num_layers//2, num_layers-3 (before norm)
    # Excluding the last layer (after norm) which has different behavior
    expected_layer_ids = [2, num_layers // 2, num_layers - 3]
    hf_layers = [
        hf_output.hidden_states[3],  # layer 2 (before norm)
        hf_output.hidden_states[
            num_layers // 2 + 1
        ],  # layer num_layers//2 (before norm)
        hf_output.hidden_states[num_layers - 2],  # layer num_layers-3 (before norm)
    ]

    hf_concat = torch.cat(hf_layers, dim=-1).cpu()
    logger.info(f"HuggingFace layers {expected_layer_ids}: {hf_concat.shape}")

    # Cleanup HuggingFace model - aggressive cleanup
    del hf_model, hf_output, hf_layers, inputs, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    time.sleep(3)

    logger.info(
        f"GPU memory freed, available: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB"
    )

    # 2. vLLM implementation
    logger.info("[2/2] Running vLLM implementation...")
    # Only test feature fusion layers (before norm), exclude the last layer (after norm)
    test_layer_ids = [2, num_layers // 2, num_layers - 3]
    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
        layer_ids=test_layer_ids,
        max_model_len=2048,
        gpu_memory_utilization=0.3,  # Conservative to avoid OOM after HF cleanup
        tensor_parallel_size=tensor_parallel_size,
    )

    try:
        # Tokenize prompts for vLLM (current implementation expects token_ids)
        # IMPORTANT: Use the SAME tokenizer that was used for HuggingFace
        # to ensure identical tokenization
        vllm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if vllm_tokenizer.pad_token is None:
            vllm_tokenizer.pad_token = vllm_tokenizer.eos_token

        # Tokenize with padding to match HuggingFace behavior
        vllm_inputs = vllm_tokenizer(
            test_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        token_ids_batch = vllm_inputs["input_ids"].tolist()

        vllm_results = generator.generate(token_ids=token_ids_batch)
        if not isinstance(vllm_results, list):
            vllm_results = [vllm_results]

        vllm_concat_per_seq = []
        for r in vllm_results:
            seq_concat = torch.cat(r["hidden_states"], dim=-1)
            vllm_concat_per_seq.append(seq_concat)
        vllm_concat = torch.stack(vllm_concat_per_seq).cpu()
        logger.info(f"vLLM layers {expected_layer_ids}: {vllm_concat.shape}")

        # Check layer IDs before cleanup
        actual_layer_ids = generator.layer_ids
    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

    # Verify layer IDs
    assert actual_layer_ids == expected_layer_ids, (
        f"Layer IDs mismatch! Got {actual_layer_ids}, expected {expected_layer_ids}"
    )

    # Verify shapes
    assert hf_concat.shape == vllm_concat.shape, (
        f"Shape mismatch! HF: {hf_concat.shape}, vLLM: {vllm_concat.shape}"
    )

    # Verify EAGLE3 output format
    for result in vllm_results:
        assert "input_ids" in result
        assert "hidden_states" in result
        assert "loss_mask" in result
        assert isinstance(result["hidden_states"], list)
        for layer_state in result["hidden_states"]:
            assert layer_state.shape[0] == result["input_ids"].shape[0], (
                "Sequence length mismatch"
            )

    # Numerical comparison
    max_diff = torch.abs(hf_concat - vllm_concat).max().item()
    mean_diff = torch.abs(hf_concat - vllm_concat).mean().item()
    logger.info(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    assert mean_diff < 0.02, (
        f"Mean difference {mean_diff} too large. "
        f"Expected layer_ids={expected_layer_ids}"
    )


@pytest.mark.regression
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("model_path", "tensor_parallel_size"),
    [
        ("Qwen/Qwen2-0.5B", 1),
    ],
)
def test_batch_vs_individual_consistency(  # noqa: C901
    model_path, tensor_parallel_size
):
    """Test that batch processing matches individual processing.

    Regression test for GitHub issue #279: VllmHiddenStatesGenerator returns
    silently wrong hidden states with batch_size > 1 or repeated calls.

    This test verifies:
    1. No KV cache state leakage between calls (Bug 1)
    2. Correct token ordering in chunked prefill (Bug 2)
    """
    # 8 distinct prompts of varying length to trigger chunked prefill
    test_prompts = [
        "What is 2+2?",
        "Explain the theory of relativity in simple terms.",
        "Write a haiku about the ocean.",
        "What are the main differences between Python and JavaScript?",
        "Hello!",
        "Translate 'good morning' to French, Spanish, and German.",
        "What is the capital of Brazil?",
        "Describe the process of photosynthesis step by step.",
    ]

    logger.info(f"Testing batch vs individual consistency: {model_path}")

    # Initialize generator with aggressive chunking to properly test the fix
    # This forces multi-iteration chunked prefill which exposes token ordering bugs
    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
        layer_ids=[10],  # Single layer for faster testing
        max_model_len=2048,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=tensor_parallel_size,
        max_num_batched_tokens=100,  # Force chunking: ~212 tokens / 100 = 3 iterations
    )

    try:
        # Tokenize prompts
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use chat template if available, otherwise just tokenize
        all_ids = []
        for text in test_prompts:
            try:
                # Try chat template first (for instruct models)
                msgs = [{"role": "user", "content": text}]
                ids = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=False,
                )
                if isinstance(ids, dict):
                    ids = ids["input_ids"]
                all_ids.append(ids.squeeze(0).tolist())
            except (ValueError, AttributeError):
                # Fallback for base models without chat template
                ids = tokenizer(text, return_tensors="pt")["input_ids"]
                all_ids.append(ids.squeeze(0).tolist())

        seq_lens = [len(ids) for ids in all_ids]
        logger.info(f"Sequence lengths: {seq_lens}")
        logger.info(f"Total tokens: {sum(seq_lens)}")

        # --- Ground truth: process each sequence individually ---
        logger.info("Processing sequences individually...")
        individual_results = []
        for i, ids in enumerate(all_ids):
            results = generator.generate([ids])
            individual_results.append(results[0])
            hs = results[0]["hidden_states"][0]
            logger.info(
                f"  Seq {i}: input_len={seq_lens[i]:3d}, hs_shape={list(hs.shape)}"
            )

        # --- Batch processing ---
        logger.info("Processing all sequences as batch...")
        batch_results = generator.generate(all_ids)

        # --- Verify results match ---
        misaligned = 0
        empty = 0
        for i in range(len(all_ids)):
            individual_hs = individual_results[i]["hidden_states"][0]
            batch_hs = batch_results[i]["hidden_states"][0]

            expected_shape = list(individual_hs.shape)
            got_shape = list(batch_hs.shape)

            # Check for empty results
            if batch_hs.numel() == 0:
                empty += 1
                logger.error(f"  Seq {i}: EMPTY (bug reproduced)")
                continue

            # Check for shape mismatch
            if got_shape != expected_shape:
                misaligned += 1
                logger.error(
                    f"  Seq {i}: WRONG SHAPE "
                    f"(got {got_shape}, expected {expected_shape})"
                )
                continue

            # Check for value mismatch
            if individual_hs.shape[0] > 0 and batch_hs.shape[0] > 0:
                mean_diff = torch.abs(individual_hs - batch_hs).mean().item()

                if mean_diff > 0.01:  # Tolerance for numerical differences
                    misaligned += 1
                    logger.error(f"  Seq {i}: WRONG VALUES (mean_diff={mean_diff:.6f})")
                    continue

        # Assert no errors
        total_errors = empty + misaligned
        assert total_errors == 0, (
            f"Batch processing returned wrong hidden states: "
            f"{empty} empty, {misaligned} misaligned out of {len(all_ids)} sequences. "
            f"This indicates bug #279 regression."
        )

        logger.info(
            f"SUCCESS: All {len(all_ids)} sequences matched between "
            f"individual and batch processing"
        )

    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)


@pytest.mark.regression
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("model_path", "tensor_parallel_size"),
    [
        ("Qwen/Qwen2-0.5B", 1),
    ],
)
def test_output_device_cuda_matches_cpu(model_path, tensor_parallel_size):
    """Regression test for CUDA output_device transport and device placement."""
    test_prompts = [
        "Summarize why deterministic inference helps regression testing.",
        "List two properties of stable API contracts.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(
        test_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )["input_ids"].tolist()

    common_kwargs = {
        "model_path": model_path,
        "layer_ids": [2],
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.3,
        "tensor_parallel_size": tensor_parallel_size,
    }

    cpu_generator = VllmHiddenStatesGenerator(output_device="cpu", **common_kwargs)
    try:
        cpu_results = cpu_generator.generate(input_ids)
    finally:
        del cpu_generator
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1)

    cuda_generator = VllmHiddenStatesGenerator(output_device="cuda:0", **common_kwargs)
    try:
        cuda_results = cuda_generator.generate(input_ids)
    finally:
        del cuda_generator
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1)

    assert len(cpu_results) == len(cuda_results)
    for cpu_result, cuda_result in zip(cpu_results, cuda_results, strict=True):
        assert cpu_result["input_ids"].device.type == "cpu"
        assert cuda_result["input_ids"].device.type == "cuda"
        assert torch.equal(cpu_result["input_ids"], cuda_result["input_ids"].cpu())

        assert len(cpu_result["hidden_states"]) == len(cuda_result["hidden_states"])
        for cpu_layer, cuda_layer in zip(
            cpu_result["hidden_states"], cuda_result["hidden_states"], strict=True
        ):
            assert cpu_layer.device.type == "cpu"
            assert cuda_layer.device.type == "cuda"
            assert cpu_layer.shape == cuda_layer.shape
            assert torch.allclose(
                cpu_layer.float(), cuda_layer.cpu().float(), atol=1e-3, rtol=1e-3
            )

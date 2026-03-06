import torch

from nanochat.gpt import GPT, GPTConfig


def test_moe_forward_and_loss_backward():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
        num_experts=4,
        top_k_experts=2,
        num_shared_experts=1,
    )
    model = GPT(config)
    model.init_weights()

    idx = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    loss = model(idx, targets)
    loss.backward()

    assert torch.isfinite(loss)
    moe_blocks = [b for b in model.transformer.h if hasattr(b.mlp, "router")]
    assert moe_blocks, "Expected MoE-enabled blocks"
    for block in moe_blocks:
        assert block.mlp.experts_fc.grad is not None
        assert block.mlp.experts_proj.grad is not None


def test_moe_scaling_param_count_reports_active_subset():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        num_experts=8,
        top_k_experts=2,
        num_shared_experts=1,
    )
    model = GPT(config)
    counts = model.num_scaling_params()
    assert counts["active_transformer_matrices"] < counts["transformer_matrices"]

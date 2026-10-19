import torch

from diff_gpt.model import block as blk
from diff_gpt.model.gpt import GPT


def _trace(embed_once: bool):
    calls = []
    orig = blk.block_attn_res

    def spy(blocks, partial_block, proj):
        calls.append(sum(b is blocks[0] for b in blocks) if blocks else 0)
        return orig(blocks, partial_block, proj)

    blk.block_attn_res = spy
    try:
        torch.manual_seed(0)
        m = GPT(vocab_size=64, n_embd=16, block_size=8, n_head=2, n_layer=3,
                use_checkpoint=False, attn_res_commit_embed_once=embed_once).eval()
        with torch.no_grad():
            logits, _ = m(torch.randint(0, 64, (1, 8)))
    finally:
        blk.block_attn_res = orig
    return calls, logits


def test_default_keeps_two_copies_of_the_embedding():
    calls, _ = _trace(False)
    assert max(calls) == 2


def test_embed_once_commits_the_embedding_exactly_once():
    calls, logits = _trace(True)
    assert max(calls) == 1
    assert torch.isfinite(logits).all()


def test_embed_once_has_same_parameters_and_loads_old_weights():
    torch.manual_seed(0)
    old = GPT(vocab_size=64, n_embd=16, block_size=8, n_head=2, n_layer=2, use_checkpoint=False)
    new = GPT(vocab_size=64, n_embd=16, block_size=8, n_head=2, n_layer=2, use_checkpoint=False,
              attn_res_commit_embed_once=True)
    new.load_state_dict(old.state_dict())
    idx = torch.randint(0, 64, (2, 8))
    tgt = torch.randint(0, 64, (2, 8))
    _, loss = new(idx, tgt)
    assert torch.isfinite(loss)
    loss.backward()
    assert new.blocks[1].attn_res_proj.weight.grad is not None

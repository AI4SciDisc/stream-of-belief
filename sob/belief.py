"""Belief tokens (Eq. 1 of the paper).

A belief token is the probability-weighted mixture of vocabulary embeddings under the
model's next-token distribution:  z = sum_v p(v) E[v].  In practice p is restricted to
its top-k tokens and renormalised.  Belief tokens live in the convex hull of the input
embeddings, so they can be fed back into the model without any learned projection.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def belief_token(h: torch.Tensor, lm_head: torch.nn.Module, emb_weight: torch.Tensor,
                 topk: int = 50, temperature: float = 1.0, scale: float = 1.0) -> torch.Tensor:
    """Map last-layer hidden states ``h`` (B, d) to belief tokens (B, d)."""
    logits = lm_head(h).float() / temperature
    top, idx = logits.topk(topk, dim=-1)
    p = torch.softmax(top, dim=-1).to(emb_weight.dtype).to(emb_weight.device)   # (B, k)
    E = emb_weight[idx.to(emb_weight.device)]                                      # (B, k, d)
    return (p.unsqueeze(-1) * E).sum(1) * scale


@torch.no_grad()
def nearest_tokens(z: torch.Tensor, emb_weight: torch.Tensor, tokenizer, scale: float = 1.0):
    """Nearest vocabulary token of every belief token in a block ``z`` (L, d).

    Returns ``(tokens, cos)``: the decoded nearest tokens and the cosine similarity to them.
    This is how the paper reads a belief ``in its own latent words`` (Figure 7).
    """
    E = torch.nn.functional.normalize(emb_weight.float(), dim=-1)
    q = torch.nn.functional.normalize(z.float().to(E.device) / scale, dim=-1)
    sims = q @ E.T
    cos, idx = sims.max(-1)
    toks = [tokenizer.decode([i]) for i in idx.tolist()]
    return toks, cos.tolist()


def latent_words(z: torch.Tensor, emb_weight: torch.Tensor, tokenizer, scale: float = 1.0,
                 max_tokens: int | None = None) -> str:
    """Belief block rendered as text through its nearest tokens (diagnostic only)."""
    toks, _ = nearest_tokens(z if max_tokens is None else z[:max_tokens], emb_weight, tokenizer, scale)
    return "".join(toks).replace("\n", " ").strip()

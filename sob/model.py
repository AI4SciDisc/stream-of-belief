"""A frozen causal language model that accepts mixed text / belief-token inputs."""
from __future__ import annotations

import copy
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .belief import belief_token

_SYS = "\x00SYS\x00"
_BODY = "\x00USERBODY\x00"


class FrozenLM:
    """Wraps a HuggingFace causal LM.

    * ``embed_pieces`` turns a list of ``("text", str)`` / ``("belief", Tensor[L, d])`` pieces into one
      input-embedding sequence, so belief blocks are inserted at the embedding layer.
    * ``generate_beliefs`` autoregressively produces ``L`` belief tokens (Eq. 1), feeding each one back
      as the next input.
    * ``decode`` / ``stream_chat`` produce text (used for probes, final solutions and the CoT baseline).
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-8B", device: str = "cuda", dtype=torch.bfloat16,
                 device_map: str | dict | None = None, topk: int = 50, temperature: float = 1.0):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dm = device_map if device_map is not None else {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=dm, attn_implementation="sdpa").eval()
        self.emb = self.model.get_input_embeddings()
        self.device = self.emb.weight.device
        self.dtype = self.emb.weight.dtype
        self.d = self.emb.weight.shape[1]
        self.topk, self.temperature = topk, temperature
        # Gemma-3 scales input embeddings by sqrt(d) inside forward; belief tokens must match.
        self.embed_scale = float(getattr(self.emb, "embed_scale", 1.0))
        g = self.model.generation_config.eos_token_id
        ids = {self.tok.eos_token_id} | set(g if isinstance(g, (list, tuple)) else [g] if g is not None else [])
        for t in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != self.tok.unk_token_id:
                ids.add(i)
        self.eos_ids = torch.tensor(sorted(i for i in ids if i is not None), device=self.device)
        # chat template split into head (system + start of user turn) and tail (end of user turn + assistant header)
        kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
        full = self.tok.apply_chat_template([{"role": "system", "content": _SYS}, {"role": "user", "content": _BODY}],
                                            tokenize=False, add_generation_prompt=True, **kw)
        head, self.chat_tail = full.split(_BODY)
        self._bos_text = ""
        if self.tok.bos_token and head.startswith(self.tok.bos_token):
            head = head[len(self.tok.bos_token):]
            self._bos_text = self.tok.bos_token
        self._chat_head = head
        self._text_cache: dict[str, torch.Tensor] = {}
        self.cost = {"prefill": 0, "decode": 0, "belief": 0, "text": 0}

    # ---------------------------------------------------------------- prompt pieces
    def chat_head(self, system: str, user_body: str) -> str:
        """BOS + system turn + start of the user turn + ``user_body`` (family-specific chat template)."""
        return self._bos_text + self._chat_head.replace(_SYS, system) + user_body

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        t = self._text_cache.get(text)
        if t is None:
            ids = self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            t = self.emb(ids)[0]
            if len(self._text_cache) < 4096:
                self._text_cache[text] = t
        return t

    @torch.no_grad()
    def embed_pieces(self, pieces) -> torch.Tensor:
        """``[("text", str) | ("belief", Tensor[L, d]), ...]`` -> ``Tensor[S, d]``."""
        return torch.cat([self.embed_text(v) if k == "text" else v.to(self.device, self.dtype) for k, v in pieces], 0)

    @torch.no_grad()
    def pad_batch(self, seqs):
        """Left-pad a list of ``(S_i, d)`` sequences -> ``(B, S, d)`` embeddings and ``(B, S)`` mask."""
        S = max(s.shape[0] for s in seqs)
        out = torch.zeros(len(seqs), S, self.d, device=self.device, dtype=self.dtype)
        mask = torch.zeros(len(seqs), S, device=self.device, dtype=torch.long)
        for b, s in enumerate(seqs):
            out[b, S - s.shape[0]:] = s
            mask[b, S - s.shape[0]:] = 1
        return out, mask

    # ---------------------------------------------------------------- belief generation
    def _belief(self, h: torch.Tensor) -> torch.Tensor:
        return belief_token(h, self.model.lm_head, self.emb.weight, self.topk, self.temperature, self.embed_scale)

    @torch.no_grad()
    def _belief_loop(self, past, mask, h, L: int, nxt=None) -> torch.Tensor:
        """Shared inner loop: starting from hidden state ``h`` after a prefill, emit ``L`` belief tokens."""
        base = self.model.model
        B = h.shape[0]
        one = torch.ones(B, 1, device=self.device, dtype=mask.dtype)
        nl = self.embed_text("\n")[0]
        ended = torch.zeros(B, dtype=torch.bool, device=self.device)
        zs = []
        e = None
        for k in range(L):
            e = self._belief(h)
            top = self.model.lm_head(h).argmax(-1).to(self.device)
            ended |= torch.isin(top, self.eos_ids)          # once the model would stop, pad the block with newlines
            e = torch.where(ended.unsqueeze(1), nl.unsqueeze(0).expand_as(e), e)
            zs.append(e)
            if k == L - 1:
                self.cost["decode"] += B * L; self.cost["belief"] += B * L
                break
            if ended.all():
                zs += [nl.unsqueeze(0).expand_as(e)] * (L - 1 - k)
                self.cost["decode"] += B * (k + 1); self.cost["belief"] += B * (k + 1)
                break
            mask = torch.cat([mask, one], 1)
            kw = {} if nxt is None else {"position_ids": nxt}
            out = base(inputs_embeds=e.unsqueeze(1), attention_mask=mask, past_key_values=past, use_cache=True, **kw)
            if nxt is not None:
                nxt = nxt + 1
            past, h = out.past_key_values, out.last_hidden_state[:, -1]
        return torch.stack(zs, 1)

    @torch.no_grad()
    def generate_beliefs(self, embeds: torch.Tensor, mask: torch.Tensor, L: int) -> torch.Tensor:
        """Prefill ``embeds`` (B, S, d) and generate a belief block ``(B, L, d)``."""
        self.cost["prefill"] += int(mask.sum())
        out = self.model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
        return self._belief_loop(out.past_key_values, mask, out.last_hidden_state[:, -1], L)

    @torch.no_grad()
    def encode_prefix(self, embeds: torch.Tensor, mask: torch.Tensor):
        """Run a shared prefix (system turn + problem) once; the KV cache is reused by every cycle."""
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model.model(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, use_cache=True)
        self.cost["prefill"] += int(mask.sum())
        return out.past_key_values, mask

    @torch.no_grad()
    def generate_beliefs_cached(self, prefix, suf_embeds: torch.Tensor, suf_mask: torch.Tensor, L: int) -> torch.Tensor:
        """Like ``generate_beliefs`` but continues a cached prefix (copied, so the prefix survives the call)."""
        cache, pmask = prefix
        past = copy.deepcopy(cache)
        mask = torch.cat([pmask, suf_mask], 1)
        pos_all = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model.model(inputs_embeds=suf_embeds, attention_mask=mask, position_ids=pos_all[:, pmask.shape[1]:],
                               past_key_values=past, use_cache=True)
        self.cost["prefill"] += int(suf_mask.sum())
        return self._belief_loop(out.past_key_values, mask, out.last_hidden_state[:, -1], L, nxt=pos_all[:, -1:] + 1)

    # ---------------------------------------------------------------- text decoding
    @torch.no_grad()
    def decode(self, embeds: torch.Tensor, mask: torch.Tensor, max_new_tokens: int = 32) -> list[str]:
        """Greedy text continuation of mixed text/belief inputs."""
        self.cost["prefill"] += int(mask.sum())
        out = self.model.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens,
                                  do_sample=False, temperature=None, top_p=None, top_k=None,
                                  pad_token_id=self.tok.pad_token_id)
        n = int((out != self.tok.pad_token_id).sum())
        self.cost["decode"] += n; self.cost["text"] += n
        return self.tok.batch_decode(out, skip_special_tokens=True)

    @torch.no_grad()
    def stream_chat(self, messages: list[dict], max_new_tokens: int = 512, enable_thinking: bool = False) -> Iterator[str]:
        """Greedy chat completion, yielding decoded text piece by piece (used by ``sob compare``)."""
        kw = {"enable_thinking": enable_thinking} if "qwen3" in self.model_id.lower() else {}
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        self.cost["prefill"] += ids.shape[1]
        out = self.model(input_ids=ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
        emitted: list[int] = []
        shown = 0
        for _ in range(max_new_tokens):
            tok = int(logits.argmax(-1))
            self.cost["decode"] += 1; self.cost["text"] += 1
            if tok in set(self.eos_ids.tolist()):
                break
            emitted.append(tok)
            text = self.tok.decode(emitted, skip_special_tokens=True)
            if len(text) > shown and not text.endswith("�"):
                yield text[shown:]
                shown = len(text)
            out = self.model(input_ids=torch.tensor([[tok]], device=self.device), past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1]

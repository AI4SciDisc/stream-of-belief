"""Stream of Belief: belief iteration Z^(k+1) = Phi_m(Z^(k)),  Phi_m = L^m o G.

* ``G`` (global interaction): every agent regenerates its belief after reading the beliefs of the others.
* ``L`` (local refinement):  every agent regenerates its belief from its own belief only.
* Each update replaces the agent's belief block (fixed size L x d); nothing accumulates across cycles.
* After each cycle every agent decodes a short probe answer; inference stops when the majority probe
  is unchanged for ``P`` consecutive cycles (or after ``K_max`` cycles). The final solutions are decoded
  from the final beliefs and combined by majority vote.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import torch

from . import agents as A
from . import tasks as T
from .belief import latent_words
from .model import FrozenLM


@dataclass
class BeliefConfig:
    N: int = 4            # agents
    L: int = 384          # belief tokens per agent
    m: int = 1            # local refinements per cycle (Phi_m = L^m o G); m=0: interaction only
    K_max: int = 16       # maximum cycles
    P: int = 2            # patience: stop when the majority probe is unchanged for P cycles
    topk: int = 50        # top-k truncation of the next-token distribution (Eq. 1)
    temperature: float = 1.0
    topology: str = "full"      # full | ring | chain | star | sequential
    personas: bool = True
    shuffle: bool = True        # agent-specific order of options / edges
    distributed: bool = False   # each agent sees only a partition of the evidence
    probe_tokens: int = 24
    probe_style: str = "answer"  # "answer": answer-only probe (paper); "brief": short solution, then the answer
    answer_tokens: int = 512
    cache_prefix: bool = True   # reuse the KV cache of the system+problem prefix across cycles
    readout_comm: bool = False  # probes / final solutions may read the other agents' beliefs too
    seed: int = 0


def neighbours(topology: str, N: int, seed: int = 0) -> list[list[int]]:
    """``nb[i]``: the agents whose beliefs agent ``i`` reads during global interaction."""
    if N == 1 or topology == "none":
        return [[] for _ in range(N)]
    if topology == "full":
        return [[j for j in range(N) if j != i] for i in range(N)]
    if topology == "ring":
        return [[(i - 1) % N, (i + 1) % N] if N > 2 else [1 - i] for i in range(N)]
    if topology == "chain":
        return [[j for j in (i - 1, i + 1) if 0 <= j < N] for i in range(N)]
    if topology == "star":
        return [[j for j in range(1, N)]] + [[0] for _ in range(1, N)]
    if topology == "sequential":
        return [[j for j in range(i)] for i in range(N)]
    raise ValueError(topology)


@dataclass
class Cycle:
    k: int
    residual: float                      # r_k = ||Z^(k+1) - Z^(k)||_F / (||Z^(k)||_F + delta)
    per_agent_residual: list[float]
    disagreement: float                  # 1 - mean pairwise cosine of the agents' pooled beliefs
    probe_answers: list[str]
    probe_majority: str
    probe_texts: list[str]               # the decoded probe of every agent
    latent_words: list[str]              # each belief read through its nearest vocabulary tokens
    pooled: np.ndarray = field(repr=False, default=None)   # (N, d) mean belief per agent, for plots
    def to_dict(self):
        d = asdict(self); d["pooled"] = None if self.pooled is None else self.pooled.tolist(); return d


@dataclass
class Result:
    answer: str                          # majority over the agents' final solutions
    agent_answers: list[str]
    agent_solutions: list[str]
    cycles: list[Cycle]
    stopped_cycle: int
    settled: bool                        # stopped by the patience rule (True) or by K_max (False)
    correct: bool | None
    cost: dict
    seconds: float
    def to_dict(self):
        d = asdict(self); d["cycles"] = [c.to_dict() for c in self.cycles]; return d


class StreamOfBelief:
    """``StreamOfBelief(model).solve(question)`` or ``.run(problem_dict)``."""

    def __init__(self, model: FrozenLM | str = "Qwen/Qwen3-8B", config: BeliefConfig | None = None, **kw):
        self.cfg = config or BeliefConfig(**kw)
        if isinstance(model, str):
            model = FrozenLM(model, topk=self.cfg.topk, temperature=self.cfg.temperature, device_map=kw.pop("device_map", None))
        self.lm = model
        self.lm.topk, self.lm.temperature = self.cfg.topk, self.cfg.temperature
        self.nb = neighbours(self.cfg.topology, self.cfg.N, self.cfg.seed)
        self._prefix = None
        self._prefix_id = None

    # ---------------------------------------------------------------- contexts
    def _problem_text(self, inst: dict, i: int) -> str:
        rng = random.Random(f"{inst['id']}-agent{i}") if self.cfg.shuffle else None
        subset = T.partition(inst, i, self.cfg.N) if self.cfg.distributed else None
        return T.render(inst, rng=rng, subset=subset)

    def _system(self, i: int) -> str:
        return A.system_prompt(i, self.cfg.N, self.cfg.personas)

    def _belief_pieces(self, inst, i, Z, mode):
        return A.belief_pieces(self.lm, self._system(i), self._problem_text(inst, i), i, Z, self.nb[i], mode)

    # ---------------------------------------------------------------- one operator application
    @torch.no_grad()
    def step(self, inst: dict, Z, mode: str) -> torch.Tensor:
        """Apply G (``mode="global"``), L (``"local"``) or the initialisation (``"initial"``) to all agents."""
        c = self.cfg
        if c.cache_prefix:
            if self._prefix_id != inst["id"]:
                pre = [self.lm.embed_pieces(A.prefix_pieces(self.lm, self._system(i), self._problem_text(inst, i)))
                       for i in range(c.N)]
                self._prefix = self.lm.encode_prefix(*self.lm.pad_batch(pre))
                self._prefix_id = inst["id"]
            suf = [self.lm.embed_pieces(self._belief_pieces(inst, i, Z, mode)[1:]) for i in range(c.N)]
            return self.lm.generate_beliefs_cached(self._prefix, *self.lm.pad_batch(suf), c.L)
        seqs = [self.lm.embed_pieces(self._belief_pieces(inst, i, Z, mode)) for i in range(c.N)]
        return self.lm.generate_beliefs(*self.lm.pad_batch(seqs), c.L)

    def global_interaction(self, inst, Z):
        return self.step(inst, Z, "global")

    def local_refinement(self, inst, Z):
        return self.step(inst, Z, "local")

    def cycle(self, inst, Z):
        """One application of Phi_m = L^m o G."""
        Z = self.global_interaction(inst, Z)
        for _ in range(self.cfg.m):
            Z = self.local_refinement(inst, Z)
        return Z

    # ---------------------------------------------------------------- decoding
    @torch.no_grad()
    def decode(self, inst: dict, Z, final: bool) -> list[str]:
        c = self.cfg
        brief = final or c.probe_style == "brief"
        seqs = [self.lm.embed_pieces(self._decode_pieces(inst, i, Z, brief)) for i in range(c.N)]
        texts = self.lm.decode(*self.lm.pad_batch(seqs), c.answer_tokens if final else c.probe_tokens)
        if brief:                     # agents without a parsable answer continue with an explicit cue
            texts = self._force_answer(inst, Z, texts, brief)
        else:
            texts = ["Answer:" + t for t in texts]
        return texts

    def _decode_pieces(self, inst, i, Z, final):
        others = [(j, Z[j]) for j in self.nb[i]] if self.cfg.readout_comm else None
        return A.decode_pieces(self.lm, self._system(i), self._problem_text(inst, i), Z[i], final, others)

    def _force_answer(self, inst, Z, texts, final=True):
        bad = [i for i, t in enumerate(texts) if T.extract(inst, t) in ("", None)]
        if not bad:
            return texts
        seqs = [self.lm.embed_pieces(self._decode_pieces(inst, i, Z, final) + [("text", texts[i] + A.PROBE_CUE)]) for i in bad]
        extra = self.lm.decode(*self.lm.pad_batch(seqs), self.cfg.probe_tokens)
        texts = list(texts)
        for i, y in zip(bad, extra):
            texts[i] = texts[i] + A.PROBE_CUE + y
        return texts

    @staticmethod
    def majority(answers: list[str]) -> str:
        valid = [a for a in answers if a not in ("", None)]
        pool = valid if valid else answers
        counts = {}
        for a in pool:
            counts[a] = counts.get(a, 0) + 1
        return max(counts, key=counts.get)

    # ---------------------------------------------------------------- metrics
    @staticmethod
    def residual(Z_old, Z_new, delta: float = 1e-8):
        a, b = Z_old.float(), Z_new.float()
        rel = ((b - a).norm() / (a.norm() + delta)).item()
        per = ((b - a).flatten(1).norm(dim=1) / (a.flatten(1).norm(dim=1) + delta)).tolist()
        return rel, per

    @staticmethod
    def disagreement(Z) -> float:
        m = torch.nn.functional.normalize(Z.float().mean(1), dim=-1)
        N = m.shape[0]
        if N < 2:
            return 0.0
        S = m @ m.T
        return (1 - (S.sum() - S.diagonal().sum()) / (N * (N - 1))).item()

    # ---------------------------------------------------------------- the stream
    @torch.no_grad()
    def run(self, inst: dict, on_cycle: Callable[[Cycle], None] | None = None, words: int = 60) -> Result:
        """Run belief iteration on one problem. ``on_cycle`` is called after every cycle (live displays)."""
        c = self.cfg
        t0 = time.time()
        self.lm.cost = {"prefill": 0, "decode": 0, "belief": 0, "text": 0}
        self._prefix_id = None
        Z = self.step(inst, None, "initial")                      # z_i <- A_i(x)
        cycles: list[Cycle] = []
        ans, texts = self._probe(inst, Z)
        rec0 = Cycle(k=0, residual=float("nan"), per_agent_residual=[float("nan")] * c.N,
                     disagreement=self.disagreement(Z), probe_answers=ans, probe_majority=self.majority(ans),
                     probe_texts=texts, latent_words=self._words(Z, words), pooled=Z.float().mean(1).cpu().numpy())
        cycles.append(rec0)
        if on_cycle:
            on_cycle(rec0)
        settled = False
        k_stop = 0
        majorities = [rec0.probe_majority]
        for k in range(1, c.K_max + 1):
            Z_new = self.cycle(inst, Z)
            rel, per = self.residual(Z, Z_new)
            Z = Z_new
            ans, texts = self._probe(inst, Z)
            rec = Cycle(k=k, residual=rel, per_agent_residual=per, disagreement=self.disagreement(Z),
                        probe_answers=ans, probe_majority=self.majority(ans), probe_texts=texts,
                        latent_words=self._words(Z, words), pooled=Z.float().mean(1).cpu().numpy())
            cycles.append(rec)
            majorities.append(rec.probe_majority)
            if on_cycle:
                on_cycle(rec)
            k_stop = k
            last = majorities[-(c.P + 1):]
            if len(last) == c.P + 1 and last[0] not in ("", None) and all(x == last[0] for x in last):
                settled = True
                break
        solutions = self.decode(inst, Z, final=True)
        answers = [T.extract(inst, s) for s in solutions]
        maj = self.majority(answers)
        correct = T.grade(inst, solutions[answers.index(maj)]) if inst.get("answer") is not None else None
        return Result(answer=maj, agent_answers=answers, agent_solutions=solutions, cycles=cycles,
                      stopped_cycle=k_stop, settled=settled, correct=correct, cost=dict(self.lm.cost),
                      seconds=time.time() - t0)

    def _probe(self, inst, Z):
        texts = self.decode(inst, Z, final=False)
        return [T.extract(inst, t) for t in texts], texts

    def _words(self, Z, n: int) -> list[str]:
        if n <= 0:
            return [""] * Z.shape[0]
        return [latent_words(Z[i], self.lm.emb.weight, self.lm.tok, self.lm.embed_scale, n) for i in range(Z.shape[0])]

    def solve(self, question: str, instruction: str = "End with 'Answer: <answer>'.", **kw) -> Result:
        """Collective answer to a free-form question."""
        return self.run(T.free(question, instruction=instruction), **kw)

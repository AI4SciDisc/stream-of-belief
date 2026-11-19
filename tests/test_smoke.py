"""End-to-end smoke test with a small model (CPU is fine): one cycle, two agents, four belief tokens."""
import os

import pytest
import torch

from sob import BeliefConfig, StreamOfBelief, tasks

MODEL = os.environ.get("SOB_TEST_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def sob():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = BeliefConfig(N=2, L=4, m=1, K_max=1, P=1, probe_tokens=4, answer_tokens=8)
    from sob import FrozenLM
    lm = FrozenLM(MODEL, device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    return StreamOfBelief(lm, cfg)


def test_belief_block_shape(sob):
    inst = tasks.make_synthetic("connectivity", 6, 1, 0)[0]
    Z = sob.step(inst, None, "initial")
    assert Z.shape == (2, 4, sob.lm.d)
    Z2 = sob.cycle(inst, Z)
    assert Z2.shape == Z.shape and torch.isfinite(Z2).all()


def test_run_returns_result(sob):
    inst = tasks.make_synthetic("connectivity", 6, 1, 1)[0]
    res = sob.run(inst)
    assert len(res.cycles) == 2 and res.stopped_cycle == 1
    assert isinstance(res.answer, str) and len(res.agent_solutions) == 2
    assert res.cost["belief"] > 0 and res.cost["text"] > 0


def test_extraction():
    inst = dict(family="connectivity", answer="yes")
    assert tasks.extract(inst, "blah\nAnswer: Yes.") == "yes"
    assert tasks.grade(dict(family="gsm8k", answer="42"), "so the answer is 42") is True
    assert tasks.extract(dict(family="arc_challenge"), "Answer: (B) because") == "B"

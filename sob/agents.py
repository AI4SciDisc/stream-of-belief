"""Agents = the frozen model + a persona, and the prompt templates of Appendix A.

Every belief update and every decoding step is one chat-formatted call: a system turn, a user
turn that may contain belief blocks inserted as soft embeddings, and an assistant turn in which
the model either writes its new belief block or text.
"""
from __future__ import annotations

PERSONAS = [
    "You reason carefully and check each step.",
    "You look for the most direct chain of evidence.",
    "You are skeptical and try to rule out wrong options.",
    "You organise the facts systematically before concluding.",
    "You focus on tracing relationships one hop at a time.",
    "You look for contradictions and edge cases.",
    "You summarise what is certain before deciding.",
    "You verify the answer by working backwards from the options.",
]

INITIAL = "\nConcisely work through the key reasoning steps and end with your conclusion."
GLOBAL_HEAD = "\nConclusions shared by other agents (in their own latent words):\n"
PREVIOUS = "\nYour previous conclusion: "
GLOBAL_CUE = ("\nReconsider the problem and the conclusions above. Concisely work through "
              "the key reasoning steps and end with your revised conclusion.")
LOCAL_CUE = ("\nReconsider the problem and your previous conclusion. Concisely work through "
             "the key reasoning steps and end with your revised conclusion.")
PROBE_USER = "\nWork through the problem, then give the final answer."
PROBE_CUE = "\n\nTherefore, the final answer is:\nAnswer:"
FINAL_USER = "\nWork through the problem, then end with 'Answer: <answer>'."
FINAL_CUE = "\n\nFinal solution (brief):"


def system_prompt(i: int, N: int, personas: bool = True) -> str:
    s = f"You are Agent {i + 1} in a team of {N} agents solving a reasoning problem together."
    if personas:
        s += " " + PERSONAS[i % len(PERSONAS)]
    return s


def prefix_pieces(lm, system: str, problem: str):
    """The cacheable prefix: system turn + problem text (identical for every cycle)."""
    return [("text", lm.chat_head(system, problem + "\n"))]


def belief_pieces(lm, system: str, problem: str, i: int, Z, neighbours: list[int], mode: str):
    """Context of agent ``i`` for a belief update.

    mode: ``"initial"`` (no beliefs yet), ``"global"`` (own belief + neighbours' beliefs) or
    ``"local"`` (own belief only). The assistant turn is left open: the model writes belief tokens.
    """
    p = prefix_pieces(lm, system, problem)
    if mode == "global":
        p.append(("text", GLOBAL_HEAD))
        for j in neighbours:
            p += [("text", f"[Agent {j + 1}]: "), ("belief", Z[j]), ("text", "\n")]
        p += [("text", PREVIOUS), ("belief", Z[i]), ("text", "\n"), ("text", GLOBAL_CUE)]
    elif mode == "local":
        p += [("text", PREVIOUS), ("belief", Z[i]), ("text", "\n"), ("text", LOCAL_CUE)]
    elif mode == "initial":
        p.append(("text", INITIAL))
    else:
        raise ValueError(mode)
    p.append(("text", lm.chat_tail))
    return p


def decode_pieces(lm, system: str, problem: str, z_i, final: bool, others=None):
    """Context for decoding text from an agent's belief (probe or final solution).

    ``others``: optional ``[(j, z_j), ...]``; when given, the other agents' beliefs are shown in the user turn
    (``readout_comm=True``), otherwise the decoder sees only the agent's own belief.
    """
    user = FINAL_USER if final else PROBE_USER
    cue = FINAL_CUE if final else PROBE_CUE
    p = prefix_pieces(lm, system, problem)
    if others:
        p.append(("text", GLOBAL_HEAD))
        for j, z_j in others:
            p += [("text", f"[Agent {j + 1}]: "), ("belief", z_j), ("text", "\n")]
    return p + [("text", user + lm.chat_tail), ("belief", z_i), ("text", cue)]

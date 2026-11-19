"""Command line: ``sob demo`` | ``sob compare`` | ``sob run`` | ``sob gif``."""
from __future__ import annotations

import argparse
import json
import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import tasks as T
from .dynamics import BeliefConfig, StreamOfBelief
from .model import FrozenLM

AGENT_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
DEFAULT_MODEL = "Qwen/Qwen3-8B"


def _add_config_args(p: argparse.ArgumentParser, L: int = 384):
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--N", type=int, default=4, help="agents")
    p.add_argument("--L", type=int, default=L, help="belief tokens per agent")
    p.add_argument("--m", type=int, default=1, help="local refinements per cycle")
    p.add_argument("--K_max", type=int, default=16, help="maximum cycles")
    p.add_argument("--P", type=int, default=2, help="patience (cycles with an unchanged majority probe)")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--topology", default="full")
    p.add_argument("--no-personas", action="store_true")
    p.add_argument("--readout-comm", action="store_true", help="probes and final solutions also read the other agents' beliefs")
    p.add_argument("--probe-tokens", type=int, default=24)
    p.add_argument("--probe-style", default="answer", choices=["answer", "brief"], help="answer-only probe (paper) or a short solution")
    p.add_argument("--device-map", default=None, help="e.g. 'auto' to shard a large model over all GPUs")
    p.add_argument("--seed", type=int, default=0)


def _config(a, **over) -> BeliefConfig:
    return BeliefConfig(N=a.N, L=a.L, m=a.m, K_max=a.K_max, P=a.P, topk=a.topk, topology=a.topology,
                        personas=not a.no_personas, readout_comm=a.readout_comm, probe_tokens=a.probe_tokens,
                        probe_style=a.probe_style, seed=a.seed, **over)


def _problem(a) -> dict:
    if a.question:
        return T.free(a.question, instruction="End with 'Answer: <answer>'.")
    if a.task in T.SYNTHETIC:
        return T.make_synthetic(a.task, a.size, 1, a.seed, start=a.index)[0]
    return T.load_benchmark(a.task, 1, a.seed)[0]


# ----------------------------------------------------------------------------------- live display
class StreamView:
    """Rich renderable of a running stream: one row per agent, one column per cycle."""

    def __init__(self, inst: dict, N: int, views: list[str] | None, title: str = "Stream of Belief", words: bool = True):
        self.inst, self.N, self.views, self.title, self.words = inst, N, views, title, words
        self.cycles = []
        self.final = None

    def _mark(self, ans: str) -> Text:
        gold = self.inst.get("answer")
        if ans in ("", None):
            return Text("·", style="dim")
        if gold is None:
            return Text(ans)
        ok = T.grade(self.inst, "Answer: " + ans)
        return Text(ans, style="bold green" if ok else "red")

    def __rich__(self):
        parts = []
        if self.views:
            t = Table.grid(padding=(0, 1))
            for i, v in enumerate(self.views):
                t.add_row(Text(f"Agent {i + 1}", style=f"bold {AGENT_COLORS[i % 8]}"), Text(v, style="dim"))
            parts.append(Panel(t, title="what each agent sees", border_style="dim"))
        tab = Table(title=self.title, expand=False, title_style="bold")
        tab.add_column("agent", style="bold")
        for c in self.cycles:
            tab.add_column(f"k={c.k}", justify="center")
        tail = []
        for i in range(self.N):
            row = [Text(f"Agent {i + 1}", style=AGENT_COLORS[i % 8])]
            row += [self._mark(c.probe_answers[i]) for c in self.cycles]
            tab.add_row(*row)
        tab.add_row(*([Text("majority", style="bold")] + [self._mark(c.probe_majority) for c in self.cycles] + tail))
        tab.add_row(*([Text("residual r_k", style="bold")] + [Text("–" if c.k == 0 else f"{c.residual:.3f}") for c in self.cycles] + tail))
        parts.append(tab)
        if self.words and self.cycles:
            c = self.cycles[-1]
            g = Table.grid(padding=(0, 1), expand=True)
            g.add_column(width=9); g.add_column(ratio=1)
            for i in range(self.N):
                g.add_row(Text(f"Agent {i + 1}", style=f"bold {AGENT_COLORS[i % 8]}"),
                          Text(c.latent_words[i], style="italic", overflow="ellipsis", no_wrap=True))
            parts.append(Panel(g, title=f"belief stream · cycle {c.k} · each belief read through its nearest tokens",
                               border_style="dim"))
        if self.final is not None:
            parts.append(self.final)
        return Group(*parts)


def _final_panel(inst, res, cfg) -> Panel:
    gold = inst.get("answer")
    style = "green" if res.correct else ("red" if res.correct is False else "white")
    lines = [Text.assemble(("Collective answer: ", "bold"), (str(res.answer), f"bold {style}"),
                           ("" if gold is None else f"   (reference: {gold})", "dim"))]
    lines.append(Text(f"agents' final answers: {res.agent_answers}", style="dim"))
    how = "majority probe unchanged for P=%d cycles" % cfg.P if res.settled else "K_max reached"
    ops = 1 + (1 + cfg.m) * res.stopped_cycle
    lines.append(Text(f"stopped after {res.stopped_cycle} cycle(s): {how}; {ops} belief updates of {cfg.L} tokens per agent; "
                      f"text tokens: {res.cost.get('text', 0)}; {res.seconds:.0f}s", style="dim"))
    return Panel(Group(*lines), border_style=style)


def _views(sob: StreamOfBelief, inst, width=110):
    return [sob._problem_text(inst, i).replace("\n", " ")[:width] + ("…" if len(sob._problem_text(inst, i)) > width else "")
            for i in range(sob.cfg.N)]


# ----------------------------------------------------------------------------------- commands
def cmd_demo(a):
    console = Console()
    inst = _problem(a)
    cfg = _config(a, distributed=not a.question and a.task in T.SYNTHETIC and not a.shared)
    console.print(Panel(Text(inst["question"], style="bold"), title=f"[bold]{inst['family']}[/bold]"
                        + ("  ·  distributed evidence: each agent sees only part of the problem" if cfg.distributed else "")))
    with console.status(f"loading {a.model}"):
        sob = StreamOfBelief(a.model, cfg, device_map=a.device_map)
    view = StreamView(inst, cfg.N, _views(sob, inst))
    with Live(view, console=console, refresh_per_second=4):
        res = sob.run(inst, on_cycle=lambda c: view.cycles.append(c))
        view.final = _final_panel(inst, res, cfg)
    if a.trace:
        json.dump(dict(problem={k: v for k, v in inst.items() if k != "paragraphs"}, config=vars(cfg), result=res.to_dict()),
                  open(a.trace, "w"))
        console.print(f"trace written to {a.trace}  (render it with: sob gif {a.trace} belief.gif)")


def cmd_compare(a):
    console = Console()
    inst = _problem(a)
    cfg = _config(a)
    console.print(Panel(Text(inst["question"], style="bold"), title=f"[bold]{inst['family']}[/bold]"))
    with console.status(f"loading {a.model}"):
        lm = FrozenLM(a.model, topk=a.topk, device_map=a.device_map)
    sob = StreamOfBelief(lm, cfg)
    # ---- left: chain of thought (one model, text) ---------------------------------------------------
    cot_text = Text("")
    cot_panel = Panel(cot_text, title="[bold]Chain of Thought[/bold]  one model · text", border_style="yellow")
    view = StreamView(inst, cfg.N, None, title=f"Stream of Belief  ·  {cfg.N} agents · latent beliefs", words=False)
    sob_panel = Panel(view, border_style="cyan")
    layout = Table.grid(expand=True)
    layout.add_column(ratio=1)
    layout.add_column(ratio=1)
    layout.add_row(cot_panel, sob_panel)
    msgs = [{"role": "user", "content": T.render(inst) + "\nThink step by step, then end with a line 'Answer: <answer>'."}]
    lm.cost = {"prefill": 0, "decode": 0, "text": 0}
    t0 = time.time()
    with Live(layout, console=console, refresh_per_second=8):
        for piece in lm.stream_chat(msgs, max_new_tokens=a.cot_tokens):
            cot_text.append(piece)
        cot_tokens, cot_s = lm.cost["decode"], time.time() - t0
        cot_ans = T.extract(inst, cot_text.plain)
        cot_ok = T.grade(inst, cot_text.plain)
        cot_panel.subtitle = f"answer: {cot_ans}  ·  {cot_tokens} sequential text tokens  ·  {cot_s:.0f}s"
        cot_panel.border_style = "green" if cot_ok else ("red" if cot_ok is False else "yellow")
        res = sob.run(inst, on_cycle=lambda c: view.cycles.append(c))
        view.final = _final_panel(inst, res, cfg)
        sob_panel.border_style = "green" if res.correct else ("red" if res.correct is False else "cyan")
    ops = 1 + (1 + cfg.m) * res.stopped_cycle
    summary = Table(title="same frozen model, two ways of reasoning", show_header=True)
    summary.add_column(""); summary.add_column("Chain of Thought"); summary.add_column("Stream of Belief")
    summary.add_row("answer", f"{cot_ans}", f"{res.answer}")
    summary.add_row("correct", str(cot_ok), str(res.correct))
    summary.add_row("text tokens generated", str(cot_tokens), str(res.cost.get("text", 0)))
    summary.add_row("belief tokens generated", "0", str(res.cost.get("belief", 0)))
    summary.add_row("sequential depth, upper bound (tokens on the critical path)", str(cot_tokens),
                    str(ops * cfg.L + res.stopped_cycle * cfg.probe_tokens + cfg.answer_tokens))
    summary.add_row("wall-clock", f"{cot_s:.0f}s", f"{res.seconds:.0f}s")
    console.print(summary)


def cmd_run(a):
    console = Console()
    cfg = _config(a, distributed=a.distributed)
    insts = T.make_synthetic(a.task, a.size, a.n, a.seed, start=a.start) if a.task in T.SYNTHETIC else T.load_benchmark(a.task, a.n, a.seed)
    with console.status(f"loading {a.model}"):
        sob = StreamOfBelief(a.model, cfg, device_map=a.device_map)
    out = open(a.out, "a") if a.out else None
    correct = 0
    for i, inst in enumerate(insts):
        res = sob.run(inst, words=0)
        correct += bool(res.correct)
        console.print(f"[{i + 1}/{len(insts)}] {inst['id']}: {res.answer} (ref {inst.get('answer')}) "
                      f"{'✓' if res.correct else '✗'}  cycles={res.stopped_cycle}  acc={correct / (i + 1):.3f}  "
                      f"probes={[c.probe_majority for c in res.cycles]}")
        if out:
            d = res.to_dict()
            for c in d["cycles"]:
                c["pooled"] = None
            out.write(json.dumps(dict(id=inst["id"], reference=inst.get("answer"), **d)) + "\n")
            out.flush()
    console.print(f"accuracy {correct}/{len(insts)} = {correct / max(1, len(insts)):.3f}")


def cmd_gif(a):
    if a.terminal:
        from .termgif import make_terminal_gif
        make_terminal_gif(a.trace, a.out, seconds_per_cycle=a.seconds)
    else:
        from .viz import make_gif
        make_gif(a.trace, a.out, fps=a.fps)
    print(f"wrote {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sob", description="Stream of Belief (arXiv:2611.21250)")
    sp = p.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("demo", help="watch four agents reach a collective answer from distributed evidence")
    _add_config_args(d, L=96)
    d.set_defaults(P=3, K_max=8, probe_style="brief", probe_tokens=96)
    d.add_argument("--task", default="connectivity", help="connectivity | shortest_path | a benchmark name")
    d.add_argument("--size", type=int, default=8, help="graph size / chain depth for synthetic tasks")
    d.add_argument("--index", type=int, default=0, help="which problem of the seeded stream (Figure 3 of the paper: seed 0, index 24)")
    d.add_argument("--question", default=None, help="your own question instead of a task")
    d.add_argument("--shared", action="store_true", help="give every agent the full problem (not distributed)")
    d.add_argument("--trace", default=None, help="write the belief trajectory to this JSON file")
    d.set_defaults(fn=cmd_demo)

    c = sp.add_parser("compare", help="Chain of Thought vs Stream of Belief on the same problem, side by side")
    _add_config_args(c)
    c.set_defaults(probe_style="brief", probe_tokens=96)
    c.add_argument("--task", default="gsm8k")
    c.add_argument("--size", type=int, default=12)
    c.add_argument("--index", type=int, default=0)
    c.add_argument("--question", default=None)
    c.add_argument("--cot-tokens", type=int, default=1024)
    c.set_defaults(fn=cmd_compare)

    r = sp.add_parser("run", help="evaluate on a benchmark")
    _add_config_args(r)
    r.add_argument("--task", default="gsm8k")
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--size", type=int, default=12)
    r.add_argument("--start", type=int, default=0, help="first problem index of the seeded stream")
    r.add_argument("--distributed", action="store_true")
    r.add_argument("--out", default=None, help="append per-problem records to this JSONL file")
    r.set_defaults(fn=cmd_run)

    g = sp.add_parser("gif", help="animate a trace written by `sob demo --trace`")
    g.add_argument("trace"); g.add_argument("out"); g.add_argument("--fps", type=float, default=12.0)
    g.add_argument("--terminal", action="store_true", help="render the terminal view instead of the belief-space animation")
    g.add_argument("--seconds", type=float, default=1.6, help="seconds per cycle for --terminal")
    g.set_defaults(fn=cmd_gif)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

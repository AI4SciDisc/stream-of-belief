"""Ten lines: four agents, one frozen model, a collective answer."""
from sob import StreamOfBelief

sob = StreamOfBelief("Qwen/Qwen3-8B", N=4, L=384, m=1)          # the configuration of the paper
res = sob.solve("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                "How much does the ball cost?")
print("collective answer:", res.answer)
for c in res.cycles:
    print(f"cycle {c.k}: probes={c.probe_answers}  r_k={c.residual:.3f}")

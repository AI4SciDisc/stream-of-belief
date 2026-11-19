"""Problems: free-form questions, synthetic distributed-evidence tasks, and public benchmarks.

A problem is a dict with at least ``family``, ``question`` and ``answer`` (``None`` if unknown).
``render(inst, rng, subset)`` produces the text an agent sees; ``extract`` / ``grade`` read and
score a decoded answer.
"""
from __future__ import annotations

import random
import re
import string
from collections import Counter, deque

LETTERS = "ABCDEFGHIJKLMNOP"


# ------------------------------------------------------------------------------------- free-form
def free(question: str, answer: str | None = None, instruction: str = "") -> dict:
    """A question typed by the user. ``instruction`` tells the model the answer format."""
    return dict(id="free-0", family="free", question=question.strip(), answer=answer, instruction=instruction,
                options=None)


# ------------------------------------------------------------------------------------- synthetic graphs
def _bfs(n, adj, s):
    dist = [-1] * n
    dist[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _random_graph(n, m, rng):
    edges = set()
    while len(edges) < m:
        u, v = rng.sample(range(n), 2)
        edges.add((min(u, v), max(u, v)))
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return sorted(edges), adj


def make_connectivity(n: int, rng: random.Random) -> dict:
    """Undirected graph with ~1.1n edges; 'is there a path between s and t?', balanced yes/no."""
    want_yes = rng.random() < 0.5
    for _ in range(1000):
        edges, adj = _random_graph(n, int(1.1 * n), rng)
        s, t = rng.sample(range(n), 2)
        d = _bfs(n, adj, s)[t]
        if (d >= 2) == want_yes and d != 1:
            return dict(family="connectivity", edges=edges, n=n, s=s, t=t,
                        question=f"Is there a path between node {s} and node {t}?",
                        options=["yes", "no"], answer="yes" if want_yes else "no")
    raise RuntimeError("connectivity sampling failed")


def make_shortest_path(n: int, rng: random.Random, min_d: int = 2) -> dict:
    for _ in range(1000):
        edges, adj = _random_graph(n, int(1.3 * n), rng)
        s, t = rng.sample(range(n), 2)
        d = _bfs(n, adj, s)[t]
        if d >= min_d:
            return dict(family="shortest_path", edges=edges, n=n, s=s, t=t, dist=d,
                        question=f"What is the length of the shortest path between node {s} and node {t}?",
                        options=None, answer=str(d))
    raise RuntimeError("shortest-path sampling failed")


CONCEPTS = ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "impus", "jompus", "gorpus",
            "shumpus", "lempus", "sterpus", "brimpus", "lorpus", "grimpus", "zhorpus", "frompus", "kerpus", "daxpus",
            "borpus", "quimpus", "yerpus", "mimpus", "felpus", "hilpus", "nerpus", "porpus", "rorpus", "selpus", "timpus"]
NAMES = ["Alex", "Sam", "Max", "Rex", "Wren", "Fae", "Polly", "Stella"]


def make_fact_chain(depth: int, rng: random.Random) -> dict:
    """Two disjoint concept chains of length ``depth``; the entity belongs to chain A's root (ProsQA-style).
    'Is <name> a <A_k> or a <B_k>?' with distractor facts; answer A_k."""
    n_noise = 2 * depth
    pool = rng.sample(CONCEPTS, 2 * (depth + 1) + max(6, n_noise))
    A, B, P = pool[:depth + 1], pool[depth + 1:2 * (depth + 1)], pool[2 * (depth + 1):]
    name = rng.choice(NAMES)
    facts = [f"{name} is a {A[0]}."]
    for i in range(depth):
        facts.append(f"Every {A[i]} is a {A[i + 1]}.")
        facts.append(f"Every {B[i]} is a {B[i + 1]}.")
    for _ in range(n_noise):
        u, v = rng.sample(P, 2)
        facts.append(f"Every {u} is a {v}.")
    for _ in range(max(1, depth // 2)):
        facts.append(f"Every {rng.choice(A[:-1])} is a {rng.choice(P)}.")
        facts.append(f"Every {rng.choice(B[:-1])} is a {rng.choice(P)}.")
    facts = list(dict.fromkeys(facts))
    opts = [A[-1], B[-1]]
    rng.shuffle(opts)
    return dict(family="fact_chain", facts=facts, name=name, question=f"Is {name} a {opts[0]} or a {opts[1]}?",
                options=opts, answer=A[-1])


PEOPLE = ["Mia", "Noah", "Ava", "Leo", "Zoe", "Eli", "Iris", "Owen", "Nora", "Finn", "Luna", "Kai", "Ivy", "Theo", "Maya", "Jude"]
CITIES = ["Rivertown", "Ashford", "Brightwater", "Caldera", "Dunmore", "Eastvale", "Fernhill", "Glenrock", "Harbor Bay",
          "Ironwood", "Juniper", "Kestrel", "Larkspur", "Marlow", "Northgate", "Oakridge"]
COMPANIES = ["Helix Labs", "Northwind Co.", "Bluefin Analytics", "Orchard Systems", "Quill Media", "Summit Foods",
             "Tidewater Energy", "Verdant Robotics", "Lumen Health", "Granite Logistics", "Cobalt Games", "Pinecrest Bank"]
REGIONS = ["the Northern Province", "the Western Province", "the Coastal Province", "the Highland Province",
           "the Lake District", "the Southern Province", "the Central Plains", "the Eastern Province"]
_CHAIN = [("works at", COMPANIES, "{a} works at {b}."), ("is headquartered in", CITIES, "{b} is headquartered in {a}."),
          ("is located in", REGIONS, "{a} is located in {b}.")]


def make_relay(depth: int, rng: random.Random) -> dict:
    """Natural-language fact chain: person -> company -> city -> region (+ distractor facts of the same kinds).
    'In which region does <person> work?'  Facts are split across agents in the distributed setting."""
    depth = max(2, min(depth, 3))
    person = rng.choice(PEOPLE)
    comps = rng.sample(COMPANIES, 3); cities = rng.sample(CITIES, 3); regions = rng.sample(REGIONS, 3)
    chain = [f"{person} works at {comps[0]}.", f"{comps[0]} is headquartered in {cities[0]}.",
             f"{cities[0]} is located in {regions[0]}."][: depth]
    answer = [comps[0], cities[0], regions[0]][depth - 1]
    others = rng.sample([p for p in PEOPLE if p != person], 2)
    noise = [f"{others[0]} works at {comps[1]}.", f"{comps[1]} is headquartered in {cities[1]}.",
             f"{cities[1]} is located in {regions[1]}.", f"{others[1]} works at {comps[2]}.",
             f"{comps[2]} is headquartered in {cities[2]}.", f"{cities[2]} is located in {regions[2]}."]
    facts = chain + noise
    rng.shuffle(facts)
    q = {1: f"At which company does {person} work?", 2: f"In which city is {person}'s employer headquartered?",
         3: f"In which region does {person} work?"}[depth]
    return dict(family="relay", facts=facts, question=q, answer=answer, options=None)


SYNTHETIC = {"connectivity": make_connectivity, "shortest_path": make_shortest_path, "fact_chain": make_fact_chain,
             "relay": make_relay}


_LEGACY = {"connectivity": "conn", "shortest_path": "sp", "fact_chain": "prosqa"}   # seed keys of the paper's runs


def make_synthetic(family: str, size: int, n: int = 1, seed: int = 0, start: int = 0) -> list[dict]:
    """Problems ``start .. start+n-1`` of the seeded stream; ids such as ``conn12-0-24`` match the paper's runs
    (Figure 3 of the paper is ``connectivity`` size 12, seed 0, index 24)."""
    key = _LEGACY.get(family, family)
    rng = random.Random(f"{key}-{size}-{seed}")
    out = []
    for i in range(start + n):
        inst = SYNTHETIC[family](size, rng)
        inst["id"] = f"{key}{size}-{seed}-{i}"
        if i >= start:
            out.append(inst)
    return out


def partition(inst: dict, agent: int, N: int) -> list:
    """Distributed evidence: every edge (or paragraph) goes to exactly one agent."""
    if inst["family"] == "hotpotqa":
        paras, sup = inst["paragraphs"], inst["supporting_titles"]
        supp = [p for p in paras if p["title"] in sup]
        dist = [p for p in paras if p["title"] not in sup]
        r = random.Random(f"{inst['id']}-partition")
        r.shuffle(dist)
        mine = [p for k, p in enumerate(supp) if k % N == agent]
        mine += [p for k, p in enumerate(dist) if (k + len(supp)) % N == agent]
        random.Random(f"{inst['id']}-order-{agent}").shuffle(mine)
        return mine
    if inst["family"] == "fact_chain":          # the entity's root fact is shared; the rest is split
        shared, rest = inst["facts"][:1], inst["facts"][1:]
    elif inst["family"] == "relay":
        shared, rest = [], list(inst["facts"])
    else:
        shared, rest = [], list(inst["edges"])
    r = random.Random(f"{inst['id']}-partition")
    order = list(range(len(rest)))
    r.shuffle(order)
    return shared + [rest[k] for j, k in enumerate(order) if j % N == agent]


# ------------------------------------------------------------------------------------- rendering
def render(inst: dict, rng: random.Random | None = None, subset: list | None = None) -> str:
    """Problem text for one agent. ``rng`` shuffles option/edge order (agent-specific view);
    ``subset`` restricts the evidence shown (distributed setting)."""
    fam = inst["family"]
    if fam == "free":
        return inst["question"] + (("\n" + inst["instruction"]) if inst.get("instruction") else "")
    if fam == "fact_chain":
        facts = list(inst["facts"]) if subset is None else list(subset)
        if rng is not None:
            rng.shuffle(facts)
        head = ("Consider the following facts about fictional concepts." if subset is None else
                "You know only some of the facts about fictional concepts (other agents know the rest).")
        return (head + "\n" + "\n".join(facts) + f"\nQuestion: {inst['question']} Answer with exactly one of: "
                f"{inst['options'][0]}, {inst['options'][1]}.")
    if fam == "relay":
        facts = list(inst["facts"]) if subset is None else list(subset)
        if rng is not None:
            rng.shuffle(facts)
        head = ("Consider the following facts." if subset is None else
                "You know only some of the facts (other agents know the rest).")
        return (head + "\n" + "\n".join(facts) + f"\nQuestion: {inst['question']} Answer with the name only; "
                "if the facts you have are not sufficient, answer 'unknown'.")
    if fam in ("connectivity", "shortest_path"):
        edges = list(inst["edges"]) if subset is None else list(subset)
        if rng is not None:
            rng.shuffle(edges)
        es = ", ".join(f"{u}-{v}" for u, v in edges)
        intro = ("Consider an undirected graph with" if subset is None else
                 "Consider an undirected graph; you know only some of its edges (other agents know the rest). It has")
        head = f"{intro} nodes 0 to {inst['n'] - 1} and the following edges: {es}.\nQuestion: {inst['question']}"
        return head + (" Answer yes or no." if fam == "connectivity" else " Answer with a single integer.")
    if fam == "gsm8k":
        return f"{inst['question']}\nGive the final numeric answer."
    if fam == "math500":
        return f"{inst['question']}\nGive the final answer in simplest form."
    if fam in MC:
        q = inst["question"]
        if rng is not None and inst.get("choices"):
            q = _shuffled_mc(inst, rng)
        return f"{q}\nAnswer with the letter of the correct option."
    if fam in YESNO:
        return f"{inst['question']}\nAnswer yes or no."
    if fam == "hotpotqa":
        paras = inst["paragraphs"] if subset is None else subset
        head = ("Use the following paragraphs to answer the question." if subset is None else
                "You can see only some of the paragraphs; other agents see the rest.")
        body = "\n\n".join(f"[{p['title']}] " + "".join(p["sents"]) for p in paras)
        return (f"{head}\n\n{body}\n\nQuestion: {inst['question']}\nAnswer with a short phrase of a few words "
                f"('yes' or 'no' for yes/no questions).")
    raise ValueError(fam)


def _shuffled_mc(inst: dict, rng: random.Random) -> str:
    """Multiple choice with the options in an agent-specific order; the gold letter is remapped per agent via
    ``inst['_perm']`` is NOT stored: instead we keep letters fixed and shuffle only the display order of the lines."""
    lines = [f"{LETTERS[k]}. {c}" for k, c in enumerate(inst["choices"])]
    rng.shuffle(lines)
    return inst["stem"] + "\n" + "\n".join(lines)


# ------------------------------------------------------------------------------------- answer extraction
MC = ("arc_challenge", "medqa", "mmlu_pro", "gpqa_diamond", "bbh_logical_deduction_five_objects",
      "bbh_tracking_shuffled_objects_five_objects", "bbh_date_understanding")
YESNO = ("bbh_web_of_lies", "connectivity")
NUMERIC = ("gsm8k", "shortest_path")


def _strip_answer_tags(text: str) -> str:
    if "<" not in text:
        return text
    text = re.sub(r"</?answer>", " ", text)
    return re.sub(r"<\s*([A-Pa-p]|yes|no|\d+)\s*>", r"\1", text)


def extract_boxed(text: str):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def extract_answer(text: str) -> str:
    """The answer after the last 'Answer:' marker (or the last \\boxed{...})."""
    text = _strip_answer_tags(text)
    b = extract_boxed(text)
    if b is not None:
        return b.strip()
    last = None
    for mm in re.finditer(r"answer\s*(?:is)?\s*[:：]?\s*", text, re.I):
        last = mm
    tail = text[last.end():] if last else text
    tail = tail.strip().split("\n")[0]
    tail = re.sub(r"^\**\s*", "", tail)
    return tail.strip().rstrip(".").strip("*$ ")


def _num(s: str):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    m = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
    if not m:
        return None
    v = m[-1]
    try:
        return float(eval(v)) if "/" in v else float(v)
    except Exception:
        return None


def _letter(ans: str, text: str) -> str:
    ans = ans.strip()
    m = re.match(r"\(?([A-P])\)?(?:[\.\):\s]|$)", ans)
    if m and (len(ans) <= 3 or re.match(r"\(?[A-P]\)?[\.\):]", ans) or len(ans.split()) <= 12):
        return m.group(1)
    m = re.findall(r"(?:answer|option)\s*(?:is|:)?\s*\(?([A-P])\)?\b", text, re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r"\(([A-P])\)", ans)
    return m[-1] if m else ""


def _norm_qa(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gold: str) -> float:
    p, g = _norm_qa(pred).split(), _norm_qa(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def extract(inst: dict, text: str) -> str:
    """Benchmark-aware answer extraction: a letter, yes/no, a number, or a short phrase."""
    fam = inst["family"]
    text = _strip_answer_tags(text)
    ans = extract_answer(text)
    if fam in MC:
        return _letter(ans, text)
    if fam in YESNO:
        m = re.search(r"\b(yes|no)\b", ans.lower())
        if m:
            return m.group(1)
        m = re.findall(r"\b(yes|no)\b", text.lower())
        return m[-1] if m else ""
    if fam in NUMERIC:
        v = _num(ans)
        if v is None:
            v = _num(text)
        return "" if v is None else (str(int(v)) if float(v).is_integer() else str(v))
    if fam == "hotpotqa":
        return ans.split("\n")[0].strip()
    if fam == "relay":
        a = ans.strip().strip("*.\"'").split("\n")[0].strip()
        return "" if a.lower().startswith("unknown") else a
    if fam == "fact_chain":
        low = ans.lower()
        named = [o for o in inst["options"] if re.search(rf"\b{re.escape(o)}\b", low)]
        if len(named) != 1:
            low = text.lower()
            named = [o for o in inst["options"] if re.search(rf"\b{re.escape(o)}\b", low.split("answer")[-1])]
        return named[0] if len(named) == 1 else ""
    return ans


def grade(inst: dict, text: str):
    """True/False, or ``None`` when the problem has no reference answer."""
    gold = inst.get("answer")
    if gold is None:
        return None
    fam = inst["family"]
    pred = extract(inst, text)
    if fam in MC or fam in YESNO:
        return pred.lower() == str(gold).lower()
    if fam in NUMERIC:
        a, g = _num(pred), _num(str(gold))
        return a is not None and g is not None and abs(a - g) < 1e-6
    if fam == "math500":
        try:
            from math_verify import parse, verify
            return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
        except Exception:
            norm = lambda s: re.sub(r"\s|\\left|\\right|\\!|\\,", "", s).replace("dfrac", "frac").replace("tfrac", "frac")
            return norm(pred) == norm(str(gold))
    if fam == "relay":
        return _norm_qa(str(gold)) in _norm_qa(pred) and len(_norm_qa(pred).split()) <= len(_norm_qa(str(gold)).split()) + 3
    if fam == "hotpotqa":
        g, p = _norm_qa(str(gold)), _norm_qa(pred)
        if not p:
            return False
        if g in ("yes", "no"):
            return p.split()[0] == g
        return p == g or (f" {g} " in f" {p} " and len(p.split()) <= len(g.split()) + 4) or qa_f1(pred, str(gold)) >= 0.8
    return _norm_qa(pred) == _norm_qa(str(gold))


# ------------------------------------------------------------------------------------- public benchmarks
def load_benchmark(family: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """Public HuggingFace versions of the eight benchmarks of the paper.

    The paper's runs used fixed local subsets of these datasets (released separately); the loaders here
    take the first ``n`` examples after a seeded shuffle.
    """
    from datasets import load_dataset  # lazy: only needed for benchmark runs

    if family == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        insts = [dict(family=family, question=r["question"], answer=r["answer"].split("####")[-1].strip().replace(",", ""))
                 for r in rows]
    elif family == "math500":
        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        insts = [dict(family=family, question=r["problem"], answer=r["answer"]) for r in rows]
    elif family == "arc_challenge":
        rows = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        insts = []
        for r in rows:
            labs, texts = r["choices"]["label"], r["choices"]["text"]
            gold = LETTERS[labs.index(r["answerKey"])] if r["answerKey"] in labs else r["answerKey"]
            insts.append(_mc(family, r["question"], texts, gold))
    elif family == "gpqa_diamond":
        rows = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")   # gated: accept the terms on HF first
        insts = []
        for k, r in enumerate(rows):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            rr = random.Random(f"gpqa-{k}")
            rr.shuffle(opts)
            insts.append(_mc(family, r["Question"], opts, LETTERS[opts.index(r["Correct Answer"])]))
    elif family == "mmlu_pro":
        rows = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        insts = [_mc(family, r["question"], r["options"], r["answer"]) for r in rows]
    elif family == "medqa":
        rows = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="test", trust_remote_code=True)
        insts = []
        for r in rows:
            opts = [o["value"] for o in r["options"]]
            keys = [o["key"] for o in r["options"]]
            insts.append(_mc(family, r["question"], opts, r["answer_idx"] if r["answer_idx"] in LETTERS else LETTERS[keys.index(r["answer_idx"])]))
    elif family.startswith("bbh_"):
        rows = load_dataset("lukaemon/bbh", family[len("bbh_"):], split="test")
        insts = []
        for r in rows:
            g = r["target"].strip()
            g = g.strip("()") if g.startswith("(") else g.lower()
            insts.append(dict(family=family, question=r["input"], answer=g))
    elif family == "hotpotqa":
        rows = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
        insts = []
        for r in rows:
            paras = [dict(title=t, sents=s) for t, s in zip(r["context"]["title"], r["context"]["sentences"])]
            insts.append(dict(family=family, question=r["question"], answer=r["answer"], paragraphs=paras,
                              supporting_titles=sorted(set(r["supporting_facts"]["title"]))))
    else:
        raise ValueError(f"unknown benchmark {family}")
    random.Random(seed).shuffle(insts)
    insts = insts[:n] if n else insts
    for i, x in enumerate(insts):
        x["id"] = f"{family}-{i}"
        x.setdefault("options", None)
    return insts


def _mc(family: str, stem: str, choices: list[str], gold: str) -> dict:
    q = stem.strip() + "\n" + "\n".join(f"{LETTERS[k]}. {c}" for k, c in enumerate(choices))
    return dict(family=family, stem=stem.strip(), choices=list(choices), question=q, answer=gold)

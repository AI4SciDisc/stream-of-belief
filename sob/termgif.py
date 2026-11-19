"""Render the terminal view of a stream (from ``sob demo --trace``) as a GIF, without external tools.

Each cycle of the trace is drawn with rich into styled segments, which are painted with a monospace
font through Pillow. The result looks like a terminal recording and is fully reproducible.
"""
from __future__ import annotations

import json
import os

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    cands = []
    try:
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        cands.append(os.path.join(d, "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"))
    except Exception:
        pass
    cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def _rgb(color, default):
    if color is None:
        return default
    t = color.get_truecolor()
    return (t.red, t.green, t.blue)


def render_frames(renderables, width: int = 118, font_size: int = 15, bg=(24, 26, 32), fg=(222, 224, 230)):
    """Draw each rich renderable as one PIL image of a ``width``-column terminal."""
    from PIL import Image, ImageDraw
    reg, bold = _font(font_size), _font(font_size, True)
    cw = reg.getbbox("M")[2]
    lh = int(font_size * 1.35)
    frames = []
    for r in renderables:
        console = Console(width=width, record=False, force_terminal=True, color_system="truecolor", file=open(os.devnull, "w"))
        lines = console.render_lines(r, console.options, pad=True)
        img = Image.new("RGB", (cw * width + 2 * 18, lh * len(lines) + 2 * 18), bg)
        draw = ImageDraw.Draw(img)
        y = 18
        for line in lines:
            x = 18
            for seg in line:
                if not seg.text:
                    continue
                st = seg.style
                f = bold if (st and st.bold) else reg
                col = _rgb(st.color, fg) if st else fg
                if st and st.dim:
                    col = tuple(int(c * 0.6) for c in col)
                if st and st.bgcolor is not None:
                    draw.rectangle([x, y, x + cw * len(seg.text), y + lh], fill=_rgb(st.bgcolor, bg))
                draw.text((x, y), seg.text, font=f, fill=col)
                x += cw * len(seg.text)
            y += lh
        frames.append(img)
    return frames


def make_terminal_gif(trace_path: str, out: str, seconds_per_cycle: float = 1.6, width: int = 118):
    from .cli import StreamView, _final_panel
    from .dynamics import BeliefConfig, Cycle, Result
    tr = json.load(open(trace_path))
    inst, cfg = tr["problem"], BeliefConfig(**tr["config"])
    rd = tr["result"]
    cycles = [Cycle(**{**c, "pooled": None}) for c in rd["cycles"]]
    res = Result(**{**rd, "cycles": cycles})
    header = Panel(Text(inst["question"], style="bold"), title=f"[bold]{inst['family']}[/bold]"
                   + ("  ·  distributed evidence: each agent sees only part of the problem" if cfg.distributed else ""))
    view = StreamView(inst, cfg.N, None)
    frames = []
    for c in cycles:                      # render immediately: the view is mutable and shared
        view.cycles.append(c)
        frames += render_frames([_group(header, view)], width=width)
    view.final = _final_panel(inst, res, cfg)
    frames += render_frames([_group(header, view)], width=width)
    h = max(f.height for f in frames)
    w = max(f.width for f in frames)
    from PIL import Image
    padded = []
    for f in frames:
        im = Image.new("RGB", (w, h), frames[0].getpixel((0, 0)))
        im.paste(f, (0, 0))
        padded.append(im)
    dur = [int(seconds_per_cycle * 1000)] * (len(padded) - 1) + [int(4 * seconds_per_cycle * 1000)]
    padded[0].save(out, save_all=True, append_images=padded[1:], duration=dur, loop=0, optimize=False)


def _group(*parts):
    from rich.console import Group
    return Group(*parts)

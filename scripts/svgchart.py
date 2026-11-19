"""Dependency-free SVG charts (lines, dots, bars, heatmaps, small multiples) for the results plots.

Style: thin marks (2px lines, >= 8px markers with a surface ring, bars <= 24px with rounded data ends),
hairline solid grid, a legend row under every multi-series panel, text in ink tokens never in series colours,
the validated categorical palette in fixed slot order (blue, orange, aqua, yellow, magenta, green, violet, red).
"""
from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
GRAY, GRID, AXIS, INK, INK2, MUTED, SURFACE = "#898781", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def text_w(s: str, size: float) -> float:
    return 0.53 * size * len(s)


def fmt_num(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:g}B"
    if a >= 1e6:
        return f"{v / 1e6:g}M"
    if a >= 1e3:
        return f"{v / 1e3:g}K"
    if a == 0:
        return "0"
    if a < 0.01:
        return f"{v:.3g}"
    return f"{v:g}"


def fmt_tokens(v: float) -> str:
    return "0" if v == 0 else (f"{v / 1e9:.3g}B" if v >= 1e9 else f"{v / 1e6:.3g}M")


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if (hi - lo) / step <= n + 1:
            break
    t0 = math.ceil(lo / step - 1e-9) * step
    ticks, t = [], t0
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    k0, k1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    decades = k1 - k0
    for k in range(k0, k1 + 1):
        for m in ((1, 2, 5) if decades <= 3 else (1,)):
            t = m * 10 ** k
            if lo * 0.999 <= t <= hi * 1.001:
                ticks.append(t)
    return ticks


def smooth(ys: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(ys)
    out, n, h = [], len(ys), window // 2
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        out.append(sum(ys[a:b]) / (b - a))
    return out


class Panel:
    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = "", xlog: bool = False, ylog: bool = False,
                 xlim=None, ylim=None, xfmt=None, yfmt=None, legend: bool = True, note: str = ""):
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.xlog, self.ylog, self.xlim, self.ylim = xlog, ylog, xlim, ylim
        self.xfmt, self.yfmt = xfmt or fmt_num, yfmt or fmt_num
        self.legend_on, self.note = legend, note
        self.items: list[dict] = []      # drawables
        self.legend: list[tuple[str, str, str]] = []   # (label, colour, kind)
        self.cats: list[str] | None = None   # categorical x (bars) or y (dots/hbars)
        self.kind = "xy"
        self._auto = 0

    def _color(self, color):
        if color is None:
            color = PALETTE[self._auto % len(PALETTE)]
            self._auto += 1
        return color

    def line(self, xs, ys, label=None, color=None, width=2.0, opacity=1.0, marker=False, dash=None, legend=True):
        pts = [(float(x), float(y)) for x, y in zip(xs, ys) if y is not None and not (isinstance(y, float) and math.isnan(y))]
        color = self._color(color)
        if len(pts) > 1500:                      # keep the files small: 1500 points is beyond screen resolution
            step = math.ceil(len(pts) / 1500)
            pts = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
        self.items.append({"t": "line", "pts": pts, "c": color, "w": width, "o": opacity, "m": marker, "d": dash})
        if label and legend:
            self.legend.append((label, color, "line"))
        return color

    def points(self, xs, ys, label=None, color=None, r=4.5, opacity=1.0, labels=None):
        color = self._color(color)
        self.items.append({"t": "pts", "pts": [(float(x), float(y)) for x, y in zip(xs, ys)], "c": color, "r": r,
                           "o": opacity, "labels": labels})
        if label:
            self.legend.append((label, color, "pt"))
        return color

    def bars(self, cats, vals, label=None, color=None, group=(0, 1), value_labels=False):
        self.kind, self.cats = "bars", list(cats)
        color = self._color(color)
        self.items.append({"t": "bars", "vals": [float(v) for v in vals], "c": color, "g": group, "vl": value_labels})
        if label:
            self.legend.append((label, color, "bar"))
        return color

    def dots(self, cats, vals, label=None, color=None, r=5, mark_min=True):
        """Horizontal dot plot: categories down the y axis, values along x; several series = several calls."""
        self.kind, self.cats = "dots", list(cats)
        color = self._color(color)
        self.items.append({"t": "dots", "vals": [None if v is None else float(v) for v in vals], "c": color, "r": r, "mm": mark_min})
        if label:
            self.legend.append((label, color, "pt"))
        return color

    def heatmap(self, matrix, rows, cols, fmt="{:.3f}", lower_is_darker=True, vmin=None, vmax=None):
        self.kind = "heat"
        self.items.append({"t": "heat", "m": matrix, "rows": list(rows), "cols": list(cols), "fmt": fmt,
                           "inv": lower_is_darker, "vmin": vmin, "vmax": vmax})

    def hline(self, y, color=GRAY, label=None, dash="4 3"):
        self.items.append({"t": "hline", "y": float(y), "c": color, "l": label, "d": dash})

    def vline(self, x, color=GRAY, label=None, dash="4 3"):
        self.items.append({"t": "vline", "x": float(x), "c": color, "l": label, "d": dash})

    def text(self, x, y, s, color=INK2, size=10.5, anchor="start", dx=4, dy=-4):
        self.items.append({"t": "text", "x": float(x), "y": float(y), "s": s, "c": color, "sz": size, "a": anchor, "dx": dx, "dy": dy})

    # ------------------------------------------------------------------ rendering
    def _ranges(self):
        xs, ys = [], []
        for it in self.items:
            if it["t"] in ("line", "pts"):
                xs += [p[0] for p in it["pts"]]
                ys += [p[1] for p in it["pts"]]
            elif it["t"] == "bars":
                ys += it["vals"] + [0.0]
            elif it["t"] == "dots":
                xs += [v for v in it["vals"] if v is not None]
            elif it["t"] == "hline":
                ys.append(it["y"])
            elif it["t"] == "vline":
                xs.append(it["x"])
        def rng(v, lim, log):
            if lim is not None:
                return lim
            v = [a for a in v if a is not None and math.isfinite(a) and (a > 0 or not log)]
            if not v:
                return (0.0, 1.0)
            lo, hi = min(v), max(v)
            if log:
                return (lo / 1.15, hi * 1.15)
            pad = (hi - lo) * 0.05 or abs(hi) * 0.05 or 1.0
            return (lo - pad, hi + pad)
        return rng(xs, self.xlim, self.xlog), rng(ys, self.ylim, self.ylog)

    def render(self, x0: float, y0: float, w: float, h: float) -> str:
        out = []
        title_h = 20 if self.title else 6
        ml, mr, mb = 58, 14, 40
        if self.kind == "dots":
            ml = 14 + max(text_w(c, 11) for c in self.cats) if self.cats else 58
        if self.kind == "heat":
            it = self.items[0]
            ml = 14 + max(text_w(r, 11) for r in it["rows"])
            mb = 36
        legend_rows = []
        if self.legend_on and self.legend and self.kind != "heat":
            row, cur = [], 0.0
            for lab, col, kind in self.legend:
                tw = 26 + text_w(lab, 11)
                if cur + tw > w - 10 and row:
                    legend_rows.append(row)
                    row, cur = [], 0.0
                row.append((lab, col, kind, cur))
                cur += tw + 6
            legend_rows.append(row)
        leg_h = 16 * len(legend_rows)
        top = y0 + title_h + leg_h + 8
        px, py, pw, ph = x0 + ml, top, w - ml - mr, h - (top - y0) - mb
        if self.title:
            out.append(f'<text x="{x0:.1f}" y="{y0 + 13:.1f}" font-size="13" font-weight="600" fill="{INK}">{escape(self.title)}</text>')
        for ri, row in enumerate(legend_rows):
            ly = y0 + title_h + 10 + 16 * ri
            for lab, col, kind, lx in row:
                X = x0 + lx
                if kind == "line":
                    out.append(f'<line x1="{X:.1f}" y1="{ly:.1f}" x2="{X + 16:.1f}" y2="{ly:.1f}" stroke="{col}" stroke-width="2.5" stroke-linecap="round"/>')
                elif kind == "pt":
                    out.append(f'<circle cx="{X + 8:.1f}" cy="{ly:.1f}" r="4" fill="{col}"/>')
                else:
                    out.append(f'<rect x="{X + 2:.1f}" y="{ly - 5:.1f}" width="12" height="10" rx="2" fill="{col}"/>')
                out.append(f'<text x="{X + 22:.1f}" y="{ly + 4:.1f}" font-size="11" fill="{INK2}">{escape(lab)}</text>')
        if self.kind == "heat":
            out += self._render_heat(px, py, pw, ph)
            return "\n".join(out)
        (xlo, xhi), (ylo, yhi) = self._ranges()
        if self.kind == "bars":
            ylo = min(0.0, ylo) if self.ylim is None else ylo
        def sx(v):
            if self.xlog:
                return px + pw * (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo))
            return px + pw * (v - xlo) / (xhi - xlo)
        def sy(v):
            if self.ylog:
                return py + ph - ph * (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))
            return py + ph - ph * (v - ylo) / (yhi - ylo)
        # y axis: ticks + grid (or categories for dots)
        if self.kind == "dots":
            n = len(self.cats)
            for i, c in enumerate(self.cats):
                cy = py + ph * (i + 0.5) / n
                out.append(f'<line x1="{px:.1f}" y1="{cy:.1f}" x2="{px + pw:.1f}" y2="{cy:.1f}" stroke="{GRID}" stroke-width="1"/>')
                out.append(f'<text x="{px - 8:.1f}" y="{cy + 4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{escape(c)}</text>')
        else:
            yt = log_ticks(ylo, yhi) if self.ylog else nice_ticks(ylo, yhi)
            for t in yt:
                Y = sy(t)
                if Y < py - 1 or Y > py + ph + 1:
                    continue
                out.append(f'<line x1="{px:.1f}" y1="{Y:.1f}" x2="{px + pw:.1f}" y2="{Y:.1f}" stroke="{GRID}" stroke-width="1"/>')
                out.append(f'<text x="{px - 6:.1f}" y="{Y + 3.5:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{escape(self.yfmt(t))}</text>')
        # x axis
        if self.kind == "bars":
            n = len(self.cats)
            for i, c in enumerate(self.cats):
                cx = px + pw * (i + 0.5) / n
                out.append(f'<text x="{cx:.1f}" y="{py + ph + 14:.1f}" font-size="10.5" fill="{INK2}" text-anchor="middle">{escape(c)}</text>')
        else:
            xt = log_ticks(xlo, xhi) if self.xlog else nice_ticks(xlo, xhi, 6)
            for t in xt:
                X = sx(t)
                if X < px - 1 or X > px + pw + 1:
                    continue
                if self.kind == "dots" or self.xlog:
                    out.append(f'<line x1="{X:.1f}" y1="{py:.1f}" x2="{X:.1f}" y2="{py + ph:.1f}" stroke="{GRID}" stroke-width="1"/>')
                out.append(f'<text x="{X:.1f}" y="{py + ph + 14:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{escape(self.xfmt(t))}</text>')
        out.append(f'<line x1="{px:.1f}" y1="{py + ph:.1f}" x2="{px + pw:.1f}" y2="{py + ph:.1f}" stroke="{AXIS}" stroke-width="1"/>')
        if self.xlabel:
            out.append(f'<text x="{px + pw / 2:.1f}" y="{py + ph + 30:.1f}" font-size="11" fill="{INK2}" text-anchor="middle">{escape(self.xlabel)}</text>')
        if self.ylabel and self.kind != "dots":
            out.append(f'<text transform="translate({x0 + 12:.1f},{py + ph / 2:.1f}) rotate(-90)" font-size="11" fill="{INK2}" text-anchor="middle">{escape(self.ylabel)}</text>')
        # clip + data
        cid = f"c{abs(hash((x0, y0, w, h))) % 10**8}"
        out.append(f'<clipPath id="{cid}"><rect x="{px - 6:.1f}" y="{py - 6:.1f}" width="{pw + 12:.1f}" height="{ph + 12:.1f}"/></clipPath>')
        out.append(f'<g clip-path="url(#{cid})">')
        for it in self.items:
            t = it["t"]
            if t == "line":
                pts = [(sx(x), sy(y)) for x, y in it["pts"] if (x > 0 or not self.xlog) and (y > 0 or not self.ylog)]
                if len(pts) >= 2:
                    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                    dash = f' stroke-dasharray="{it["d"]}"' if it["d"] else ""
                    out.append(f'<path d="{d}" fill="none" stroke="{it["c"]}" stroke-width="{it["w"]}" stroke-opacity="{it["o"]}" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
                if it["m"]:
                    for x, y in pts:
                        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{it["c"]}" stroke="{SURFACE}" stroke-width="1.5" fill-opacity="{it["o"]}"/>')
            elif t == "pts":
                for i, (x, y) in enumerate(it["pts"]):
                    X, Y = sx(x), sy(y)
                    out.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="{it["r"]}" fill="{it["c"]}" fill-opacity="{it["o"]}" stroke="{SURFACE}" stroke-width="2"/>')
                    if it["labels"]:
                        out.append(f'<text x="{X + 7:.1f}" y="{Y + 4:.1f}" font-size="10.5" fill="{INK2}">{escape(str(it["labels"][i]))}</text>')
            elif t == "bars":
                n, (gi, gn) = len(self.cats), it["g"]
                slot = pw / n
                bw = min(24.0, (slot * 0.7) / gn)
                for i, v in enumerate(it["vals"]):
                    cx = px + slot * (i + 0.5) + (gi - (gn - 1) / 2) * (bw + 2)
                    y1, y2 = sy(max(v, ylo if self.ylim else 0.0)), sy(min(v, ylo if self.ylim else 0.0) if v < 0 else max(ylo, 0.0) if self.ylim is None else ylo)
                    top_, bot = min(y1, y2), max(y1, y2)
                    r = min(4.0, bw / 2, max(0.0, bot - top_))
                    X = cx - bw / 2
                    d = (f"M{X:.1f},{bot:.1f} L{X:.1f},{top_ + r:.1f} Q{X:.1f},{top_:.1f} {X + r:.1f},{top_:.1f} "
                         f"L{X + bw - r:.1f},{top_:.1f} Q{X + bw:.1f},{top_:.1f} {X + bw:.1f},{top_ + r:.1f} L{X + bw:.1f},{bot:.1f} Z")
                    out.append(f'<path d="{d}" fill="{it["c"]}"/>')
                    if it["vl"]:
                        out.append(f'<text x="{cx:.1f}" y="{top_ - 4:.1f}" font-size="10" fill="{INK2}" text-anchor="middle">{escape(self.yfmt(v))}</text>')
            elif t == "dots":
                n = len(self.cats)
                vals = it["vals"]
                best = min((v for v in vals if v is not None), default=None)
                for i, v in enumerate(vals):
                    if v is None:
                        continue
                    cy = py + ph * (i + 0.5) / n
                    X = sx(v)
                    out.append(f'<circle cx="{X:.1f}" cy="{cy:.1f}" r="{it["r"]}" fill="{it["c"]}" stroke="{SURFACE}" stroke-width="2"/>')
                    if it["mm"] and v == best:
                        out.append(f'<text x="{X + 9:.1f}" y="{cy + 4:.1f}" font-size="10.5" fill="{INK2}">{escape(self.xfmt(v))}</text>')
            elif t == "hline":
                Y = sy(it["y"])
                out.append(f'<line x1="{px:.1f}" y1="{Y:.1f}" x2="{px + pw:.1f}" y2="{Y:.1f}" stroke="{it["c"]}" stroke-width="1.5" stroke-dasharray="{it["d"]}"/>')
                if it["l"]:
                    out.append(f'<text x="{px + pw - 4:.1f}" y="{Y - 4:.1f}" font-size="10.5" fill="{INK2}" text-anchor="end">{escape(it["l"])}</text>')
            elif t == "vline":
                X = sx(it["x"])
                out.append(f'<line x1="{X:.1f}" y1="{py:.1f}" x2="{X:.1f}" y2="{py + ph:.1f}" stroke="{it["c"]}" stroke-width="1.5" stroke-dasharray="{it["d"]}"/>')
                if it["l"]:
                    out.append(f'<text x="{X + 4:.1f}" y="{py + 11:.1f}" font-size="10.5" fill="{INK2}">{escape(it["l"])}</text>')
            elif t == "text":
                X, Y = sx(it["x"]) + it["dx"], sy(it["y"]) + it["dy"]
                out.append(f'<text x="{X:.1f}" y="{Y:.1f}" font-size="{it["sz"]}" fill="{it["c"]}" text-anchor="{it["a"]}">{escape(it["s"])}</text>')
        out.append("</g>")
        if self.note:
            out.append(f'<text x="{px + pw:.1f}" y="{py + ph + 30:.1f}" font-size="10" fill="{MUTED}" text-anchor="end">{escape(self.note)}</text>')
        return "\n".join(out)

    def _render_heat(self, px, py, pw, ph):
        it = self.items[0]
        m, rows, cols = it["m"], it["rows"], it["cols"]
        vals = [v for r in m for v in r if v is not None]
        vmin = it["vmin"] if it["vmin"] is not None else min(vals)
        vmax = it["vmax"] if it["vmax"] is not None else max(vals)
        out = []
        cw, ch = pw / len(cols), ph / len(rows)
        for j, c in enumerate(cols):
            out.append(f'<text x="{px + cw * (j + 0.5):.1f}" y="{py + ph + 14:.1f}" font-size="10.5" fill="{INK2}" text-anchor="middle">{escape(c)}</text>')
        for i, r in enumerate(rows):
            out.append(f'<text x="{px - 8:.1f}" y="{py + ch * (i + 0.5) + 4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{escape(r)}</text>')
            for j, v in enumerate(m[i]):
                X, Y = px + cw * j + 1, py + ch * i + 1
                if v is None:
                    out.append(f'<rect x="{X:.1f}" y="{Y:.1f}" width="{cw - 2:.1f}" height="{ch - 2:.1f}" rx="3" fill="{GRID}"/>')
                    continue
                f = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
                if it["inv"]:
                    f = 1 - f
                k = int(round(f * (len(SEQ) - 1)))
                col = SEQ[k]
                ink = "#ffffff" if k >= 6 else INK
                out.append(f'<rect x="{X:.1f}" y="{Y:.1f}" width="{cw - 2:.1f}" height="{ch - 2:.1f}" rx="3" fill="{col}"/>')
                out.append(f'<text x="{X + (cw - 2) / 2:.1f}" y="{Y + (ch - 2) / 2 + 4:.1f}" font-size="10.5" fill="{ink}" text-anchor="middle">{escape(it["fmt"].format(v))}</text>')
        if self.xlabel:
            out.append(f'<text x="{px + pw / 2:.1f}" y="{py + ph + 30:.1f}" font-size="11" fill="{INK2}" text-anchor="middle">{escape(self.xlabel)}</text>')
        return out


class Figure:
    def __init__(self, title: str, subtitle: str = "", ncols: int = 1, panel_w: int = 440, panel_h: int = 280):
        self.title, self.subtitle, self.ncols, self.pw, self.ph = title, subtitle, ncols, panel_w, panel_h
        self.panels: list[Panel] = []

    def add(self, panel: Panel) -> Panel:
        self.panels.append(panel)
        return panel

    def svg(self) -> str:
        n = len(self.panels)
        nrows = math.ceil(n / self.ncols)
        head = 46 if self.subtitle else 30
        W = 16 + self.ncols * (self.pw + 16)
        H = head + nrows * (self.ph + 12) + 8
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
               f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
               f'<text x="16" y="20" font-size="15" font-weight="600" fill="{INK}">{escape(self.title)}</text>']
        if self.subtitle:
            out.append(f'<text x="16" y="37" font-size="11.5" fill="{INK2}">{escape(self.subtitle)}</text>')
        for i, p in enumerate(self.panels):
            r, c = divmod(i, self.ncols)
            out.append(p.render(16 + c * (self.pw + 16), head + r * (self.ph + 12), self.pw, self.ph))
        out.append("</svg>")
        return "\n".join(out)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.svg())
        return path

#!/usr/bin/env python
"""Live training dashboard: serves charts of every run's log.jsonl in a browser.

  python scripts/dashboard.py --runs /mnt/localssd/spp-runs --port 8787
  (remote box: ssh -L 8787:localhost:8787 <host>, then open http://localhost:8787)

No dependencies beyond the standard library; the page polls /data every 10 s.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

MAX_ROWS = 800     # per-round rows are strided down to this many; eval/ICL rows are always kept
KEEP_KEYS = ("learner/loss", "reward/mean", "reward/frac_pos", "prog/len_mean", "prog/emitted_mean",
             "prog/frac_full_output", "archive/niches", "bank/size", "gen/kl_to_prior", "time/round",
             "tokens", "lr")


def find_runs(roots: list[str]) -> list[Path]:
    out = []
    for r in roots:
        p = Path(r)
        if (p / "log.jsonl").exists():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if (q / "log.jsonl").exists()))
    return out


def load_run(d: Path) -> dict:
    rows = []
    with open(d / "log.jsonl") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    cfg = {}
    if (d / "config.json").exists():
        cfg = json.loads((d / "config.json").read_text())
    special = [r for r in rows if any(k.startswith(("eval", "icl/")) for k in r)]
    stride = max(1, len(rows) // MAX_ROWS)
    kept = {id(r) for r in rows[::stride]} | {id(r) for r in special} | ({id(rows[-1])} if rows else set())
    slim = []
    for r in rows:
        if id(r) not in kept:
            continue
        s = {"round": r["round"]}
        for k in KEEP_KEYS:
            if k in r:
                s[k] = r[k]
        for k, v in r.items():
            if k.startswith(("eval", "icl/")):
                s[k] = v
        slim.append(s)
    last = rows[-1] if rows else {}
    mtime = (d / "log.jsonl").stat().st_mtime
    age = time.time() - mtime
    round_time = last.get("time/round", 0.0)
    return {
        "name": d.name, "dir": str(d), "rows": slim, "n_rounds": len(rows),
        "rounds_target": cfg.get("rounds"), "data": cfg.get("data"), "preset": cfg.get("preset"),
        "tokens": last.get("tokens", 0), "round_time": round_time, "age_s": age,
        "alive": age < max(60.0, 5 * round_time),
        "eta_h": ((cfg.get("rounds", 0) - len(rows)) * round_time / 3600) if round_time and cfg.get("rounds") else None,
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-Play Pretraining</title>
<style>
:root { color-scheme: light; --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --grid:#e6e5e1; --axis:#b8b7b1;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#6250d6; --s8:#e34948; --good:#0ca30c; --critical:#d03b3b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19; --panel:#222221; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e88; --grid:#33332f; --axis:#55544f;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#0ca30c; --s7:#9085e9; --s8:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --panel:#222221; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e88; --grid:#33332f; --axis:#55544f;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#0ca30c; --s7:#9085e9; --s8:#e66767; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--ink); font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; padding:16px; }
h1 { font-size:18px; margin:0 0 4px; } .sub { color:var(--ink2); font-size:13px; margin-bottom:12px; }
table.runs { border-collapse:collapse; margin-bottom:16px; font-size:13px; width:100%; max-width:1100px; }
table.runs th, table.runs td { text-align:left; padding:4px 10px 4px 0; border-bottom:1px solid var(--grid); font-variant-numeric: tabular-nums; }
table.runs th { color:var(--ink2); font-weight:600; }
.swatch { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:-1px; margin-right:6px; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:12px; }
.card { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:10px 10px 4px; position:relative; }
.card h3 { margin:0 0 2px; font-size:13px; font-weight:600; } .card .unit { color:var(--muted); font-weight:400; }
svg text { fill:var(--ink2); font-size:11px; } svg .tick { stroke:var(--grid); } svg .axis { stroke:var(--axis); }
.tip { position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--axis); border-radius:6px; padding:6px 8px; font-size:12px; display:none; z-index:2; white-space:nowrap; }
.tip b { font-variant-numeric: tabular-nums; }
.empty { color:var(--muted); font-size:12px; padding:30px 0; text-align:center; }
</style></head><body>
<h1>Self-Play Pretraining with Zero Data</h1>
<div class="sub" id="sub">loading…</div>
<table class="runs"><thead><tr><th>run</th><th>data</th><th>model</th><th>rounds</th><th>tokens</th><th>s/round</th><th>ETA</th><th>status</th></tr></thead><tbody id="runs"></tbody></table>
<div class="grid" id="grid"></div>
<script>
const COLORS = ['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8'];
const CHARTS = [
  ['learner/loss', 'Learner loss on the pool', 'nats / byte'],
  ['reward/mean', 'Reward, pool mean', 'abs. preconditioned alignment'],
  ['reward/frac_pos', 'Reward > 0', 'fraction of pool'],
  ['gen/kl_to_prior', 'Generator KL to uniform prior', 'nats / program'],
  ['prog/len_mean', 'Program length', 'tokens (mean)'],
  ['prog/emitted_mean', 'Bytes emitted per program', 'mean'],
  ['prog/frac_full_output', 'Programs filling the full output', 'fraction'],
  ['archive/niches', 'MAP-Elites niches occupied', 'of 36'],
  ['bank/size', 'Replay bank size', 'programs'],
  ['time/round', 'Round time', 'seconds'],
];
const fmt = (v) => Math.abs(v) >= 1e6 ? (v/1e6).toFixed(2)+'M' : Math.abs(v) >= 1e4 ? (v/1e3).toFixed(1)+'k' : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toExponential(2);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function series(runs, key) {
  return runs.map((r, i) => ({ name: r.name, color: css(COLORS[i % 8]), pts: r.rows.filter(x => key in x).map(x => [x.round, x[key]]) })).filter(s => s.pts.length);
}
function lineChart(el, title, unit, ser, xLabel) {
  const W = el.clientWidth - 20 || 320, H = 190, L = 46, R = 14, T = 8, B = 26;
  if (!ser.length) { el.innerHTML = `<h3>${title} <span class="unit">${unit}</span></h3><div class="empty">no data yet</div>`; return; }
  const xs = ser.flatMap(s => s.pts.map(p => p[0])), ys = ser.flatMap(s => s.pts.map(p => p[1]));
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y0 -= 0.5; y1 += 0.5; } const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
  const X = v => L + (v - x0) / (x1 - x0) * (W - L - R), Y = v => T + (y1 - v) / (y1 - y0) * (H - T - B);
  const yt = [0, 0.5, 1].map(f => y0 + f * (y1 - y0)), xt = [0, 0.5, 1].map(f => x0 + f * (x1 - x0));
  let svg = `<svg width="${W}" height="${H}" role="img" aria-label="${title}">`;
  yt.forEach(v => { svg += `<line class="tick" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v)}</text>`; });
  svg += `<line class="axis" x1="${L}" x2="${W - R}" y1="${Y(y0)}" y2="${Y(y0)}"/>`;
  xt.forEach(v => { svg += `<text x="${X(v)}" y="${H - 8}" text-anchor="middle">${fmt(v)}</text>`; });
  svg += `<text x="${W - R}" y="${H - 8}" text-anchor="end" style="fill:var(--muted)">${xLabel}</text>`;
  ser.forEach(s => {
    const d = s.pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join('');
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
    if (s.pts.length === 1) svg += `<circle cx="${X(s.pts[0][0])}" cy="${Y(s.pts[0][1])}" r="4" fill="${s.color}"/>`;
  });
  svg += `<line id="xh" class="axis" y1="${T}" y2="${H - B}" x1="0" x2="0" style="display:none" stroke-dasharray="3 3"/><g id="hm"></g></svg>`;
  el.innerHTML = `<h3>${title} <span class="unit">${unit}</span></h3>${svg}<div class="tip"></div>`;
  const node = el.querySelector('svg'), tip = el.querySelector('.tip'), xh = el.querySelector('#xh'), hm = el.querySelector('#hm');
  node.addEventListener('mousemove', ev => {
    const bx = node.getBoundingClientRect(), mx = ev.clientX - bx.left; if (mx < L || mx > W - R) return;
    const xv = x0 + (mx - L) / (W - L - R) * (x1 - x0);
    let html = '', marks = '';
    ser.forEach(s => { let best = s.pts[0]; for (const p of s.pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      html += `<div><span class="dot" style="background:${s.color}"></span>${s.name}: <b>${fmt(best[1])}</b> <span style="color:var(--muted)">@ ${best[0]}</span></div>`;
      marks += `<circle cx="${X(best[0])}" cy="${Y(best[1])}" r="4" fill="${s.color}" stroke="var(--panel)" stroke-width="2"/>`; });
    hm.innerHTML = marks; xh.setAttribute('x1', mx); xh.setAttribute('x2', mx); xh.style.display = '';
    tip.innerHTML = html; tip.style.display = 'block'; tip.style.left = (mx + 14 + 180 > W ? mx - 180 : mx + 14) + 'px'; tip.style.top = '28px';
  });
  node.addEventListener('mouseleave', () => { tip.style.display = 'none'; xh.style.display = 'none'; hm.innerHTML = ''; });
}
function iclSeries(run) {
  const latest = [...run.rows].reverse().find(r => Object.keys(r).some(k => k.startsWith('icl/'))); if (!latest) return [];
  const by = {}; Object.entries(latest).forEach(([k, v]) => { if (!k.startsWith('icl/')) return; const [, task, m] = k.split('/'); (by[task] ||= []).push([+m.slice(1), v]); });
  return Object.entries(by).map(([task, pts], i) => ({ name: task, color: css(COLORS[i % 8]), pts: pts.sort((a, b) => a[0] - b[0]), round: latest.round }));
}
let cards = {};
function card(id) { let c = cards[id]; if (!c) { c = document.createElement('div'); c.className = 'card'; cards[id] = c; document.getElementById('grid').appendChild(c); } return c; }
async function refresh() {
  const res = await fetch('data'); const runs = await res.json();
  const tb = document.getElementById('runs'); tb.innerHTML = runs.map((r, i) =>
    `<tr><td><span class="swatch" style="background:${css(COLORS[i % 8])}"></span>${r.name}</td><td>${r.data || ''}</td><td>${r.preset || ''}</td><td>${r.n_rounds}${r.rounds_target ? ' / ' + r.rounds_target : ''}</td><td>${fmt(r.tokens)}</td><td>${r.round_time.toFixed(1)}</td><td>${r.eta_h == null ? '' : r.eta_h.toFixed(1) + ' h'}</td><td><span class="dot" style="background:${r.alive ? 'var(--good)' : 'var(--critical)'}"></span>${r.alive ? 'running' : 'idle ' + Math.round(r.age_s / 60) + ' min'}</td></tr>`).join('');
  document.getElementById('sub').textContent = `${runs.length} run(s) · refreshed ${new Date().toLocaleTimeString()} · hover a chart for values · x axis = self-play round`;
  const evalKeys = [...new Set(runs.flatMap(r => r.rows.flatMap(x => Object.keys(x).filter(k => /^eval\d*\//.test(k)))))].sort();
  evalKeys.forEach(k => { const [pre, name] = k.split('/'); const ctx = pre === 'eval' ? 'training ctx' : pre.slice(4) + ' ctx';
    lineChart(card(k), 'Zero-shot eval: ' + name.replace('_bpb', '') + ' (' + ctx + ')', 'bits / byte', series(runs, k), 'round'); });
  CHARTS.forEach(([k, t, u]) => { const s = series(runs, k); if (s.length) lineChart(card(k), t, u, s, 'round'); });
  runs.forEach(r => { const s = iclSeries(r); if (s.length) lineChart(card('icl/' + r.name), `ICL greedy success · ${r.name} (round ${s[0].round})`, 'success rate vs in-context examples m', s, 'm'); });
}
refresh(); setInterval(refresh, 10000); window.addEventListener('resize', refresh);
</script></body></html>
"""


def make_handler(roots):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = PAGE.encode()
                ctype = "text/html; charset=utf-8"
            elif path == "/data":
                body = json.dumps([load_run(d) for d in find_runs(roots)]).encode()
                ctype = "application/json"
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs"], help="run dirs, or parent dirs containing runs")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(a.runs))
    print(f"dashboard: http://localhost:{a.port}  (runs: {', '.join(a.runs)})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Every results figure, as SVG, grouped by experiment under results/plots/<group>/, plus results/plots/README.md.

Sources: the small files in results/ (CSV summaries, eval curves) and, when present on this machine, the full run logs
(self-play: <runs>/<name>/log.jsonl and eval_curve.jsonl; downstream sweeps: <pp1m>/<run>/log.jsonl, <pp25m>/<run>/log.jsonl,
<pp25m>/ppt/<curriculum>/log.jsonl). Figures whose sources are missing are skipped with a note.

  python scripts/plot_results.py [--out results/plots] [--runs /mnt/localssd/spp-runs] [--pp1m /mnt/localssd/spp-pp1m-fresh]
                                 [--pp25m /mnt/localssd/spp-pp25m]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import svgchart as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = sc.PALETTE
GRAY = sc.GRAY
FIGS: list[tuple[str, str, str]] = []     # (group, file, caption)
GROUPS: dict[str, str] = {}


def jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in open(path)] if path.exists() else []


def by_round(rows):
    return {r["round"]: r for r in rows}


def series(rows, key, max_round=None):
    xs, ys = [], []
    for r in rows:
        if key in r and (max_round is None or r["round"] <= max_round):
            xs.append(r["round"])
            ys.append(r[key])
    return xs, ys


def csv_rows(path):
    path = Path(path)
    return list(csv.DictReader(open(path))) if path.exists() else []


def best_row(rows, budget, arm):
    cand = [r for r in rows if r["budget"] == budget and r["arm"] == arm]
    return min(cand, key=lambda r: float(r["final"])) if cand else None


def sweep_curve(root, name):
    rows = [r for r in jsonl(Path(root) / name / "log.jsonl") if "val_bpb" in r]
    return [r["tokens"] for r in rows], [r["val_bpb"] for r in rows]


def save(fig, group, name, caption, desc=None):
    path = Path(ARGS.out) / group / name
    fig.save(path)
    FIGS.append((group, name, caption))
    if desc:
        GROUPS[group] = desc
    print(f"  {path.relative_to(ROOT)}")


def skip(group, name, why):
    print(f"  skipped {group}/{name}: {why}")


# ---------------------------------------------------------------- shared figure builders
def eval_panel(fig, title, srcs, ylog=False, window=5, xlabel="self-play round", ylabel="bits per byte ↓", ylim=None):
    """srcs: list of (label, xs, ys, colour, dash). Raw at 45% + centred moving average (window evals) in full."""
    p = fig.add(sc.Panel(title, xlabel, ylabel, ylog=ylog, ylim=ylim))
    for lab, xs, ys, col, dash in srcs:
        if not xs:
            continue
        p.line(xs, ys, color=col, width=1.25, opacity=0.4, dash=dash, legend=False)
        p.line(xs, sc.smooth(ys, window), label=lab, color=col, width=2.2, dash=dash)
    return p


DYN_KEYS = [
    ("learner/pool_loss_nats", "Learner loss on the pool (nats/byte)", False),
    ("loss/fresh_mean", "Learner loss, fresh programs", False),
    ("loss/replay_mean", "Learner loss, replayed programs", False),
    ("prog/emitted_mean", "Bytes emitted per program", False),
    ("prog/frac_full_output", "Programs filling the whole output", False),
    ("prog/frac_empty_output", "Programs with empty output", False),
    ("prog/len_mean", "Program length (bytes)", False),
    ("prog/depth_mean", "Loop nesting depth", False),
    ("prog/steps_mean", "Interpreter steps per program", False),
    ("gen/kl_to_prior", "Generator KL to the uniform prior (nats/program)", False),
    ("gen/nats_per_token", "Generator nats per program token", False),
    ("gen/frac_max_len", "Programs at the maximum length", False),
    ("reward/mean", "Reward mean (raw directional derivative, log)", True),
    ("archive/niches", "Occupied archive niches", False),
    ("learner/grad_norm", "Learner gradient norm (log)", True),
    ("learner/param_norm", "Learner parameter norm", False),
]


def dynamics_fig(title, subtitle, runs, window=50, max_round=None, keys=DYN_KEYS):
    """runs: list of (label, rows, colour)."""
    fig = sc.Figure(title, subtitle + "; descriptive statistics, no preferred direction (hence no arrow)", ncols=4, panel_w=330, panel_h=230)
    for key, ptitle, ylog in keys:
        p = fig.add(sc.Panel(ptitle, "round", "", ylog=ylog))
        for lab, rows, col in runs:
            xs, ys = series(rows, key, max_round)
            if xs:
                p.line(xs, sc.smooth(ys, window), label=lab if len(runs) > 1 else None, color=col, width=1.8)
    return fig


ICL_TASKS = ["stack", "assoc_recall", "reverse", "max", "min", "sum"]
ICL_NAMES = {"stack": "Stack", "assoc_recall": "Associative recall", "reverse": "Reverse", "max": "Max", "min": "Min", "sum": "Sum"}
SHOTS = [0, 1, 2, 4, 8, 16, 32]


def icl_rounds_fig(title, subtitle, runs, max_round=None):
    """One panel per task; per run, accuracy at 32 shots (solid), 16 (dashed) and 8 (dotted)."""
    fig = sc.Figure(title, subtitle, ncols=3, panel_w=400, panel_h=240 + 40 * (len(runs) - 1))
    for t in ICL_TASKS:
        p = fig.add(sc.Panel(ICL_NAMES[t], "round", "accuracy ↑", ylim=(0, 1.02)))
        for lab, rows, col in runs:
            for m, dash in ((32, None), (16, "7 4"), (8, "2 3")):
                xs, ys = series([r for r in rows if f"icl/{t}/m{m}" in r], f"icl/{t}/m{m}", max_round)
                if xs:
                    p.line(xs, ys, label=f"{lab + ', ' if len(runs) > 1 else ''}{m} shots", color=col, width=2 if m == 32 else 1.5, dash=dash, marker=len(xs) < 40)
    return fig


def icl_shots_fig(title, subtitle, runs, rnd):
    fig = sc.Figure(title, subtitle, ncols=3, panel_w=400, panel_h=240)
    xf = lambda t: str(SHOTS[int(round(t))]) if abs(t - round(t)) < 1e-6 and 0 <= round(t) < len(SHOTS) else ""
    for t in ICL_TASKS:
        p = fig.add(sc.Panel(ICL_NAMES[t], "in-context examples", "accuracy ↑", ylim=(0, 1.02), xlim=(-0.3, 6.3), xfmt=xf))
        for lab, rows, col in runs:
            br = by_round(rows)
            r = br.get(rnd)
            if r and f"icl/{t}/m32" in r:
                p.line(range(len(SHOTS)), [r[f"icl/{t}/m{m}"] for m in SHOTS], label=lab if len(runs) > 1 else None, color=col, marker=True)
    return fig


def dots_fig(title, subtitle, rows, arms, labels, budgets=("1B", "2B"), xlim=None):
    fig = sc.Figure(title, subtitle, ncols=1, panel_w=560, panel_h=40 + 26 * len(arms))
    p = fig.add(sc.Panel("", "final validation bpb (after the LR decay) ↓", xlim=xlim, xfmt=lambda v: f"{v:.3f}"))
    for b, col in zip(budgets, (BLUE, ORANGE)):
        vals = [float(best_row(rows, b, a)["final"]) if best_row(rows, b, a) else None for a in arms]
        p.dots(labels, vals, label=f"{b} fresh tokens", color=col)
    return fig


def curves_panel(fig, title, root, rows, budget, arms, ylim, frac_from=0.0, xfmt=sc.fmt_tokens):
    """arms: list of (arm, label, colour, dash). Best LR x WD config per arm; part of the epoch from frac_from."""
    p = fig.add(sc.Panel(title, "fresh DCLM tokens", "validation bpb ↓", ylim=ylim, xfmt=xfmt))
    for arm, lab, col, dash in arms:
        r = best_row(rows, budget, arm)
        if not r:
            continue
        xs, ys = sweep_curve(root, r["name"])
        if not xs:
            continue
        tot = xs[-1]
        pts = [(x, y) for x, y in zip(xs, ys) if x >= frac_from * tot]
        p.line([a for a, _ in pts], [b for _, b in pts], label=f"{lab} (lr {float(r['lr']):g}, wd {float(r['wd']):g})", color=col, dash=dash)
    return p


def thresholds_fig(title, subtitle, rows, budget, arms, labels_cols):
    cols = [c for c in rows[0].keys() if re.fullmatch(r"to[0-9.]+_M", c)]
    thr = [float(c[2:-2]) for c in cols]
    fig = sc.Figure(title, subtitle, ncols=1, panel_w=560, panel_h=300)
    p = fig.add(sc.Panel("", "validation bpb threshold", "fresh tokens to reach it ↓", yfmt=sc.fmt_tokens))
    for arm, lab, col, dash in arms:
        r = best_row(rows, budget, arm)
        if not r:
            continue
        xs = [t for t, c in zip(thr, cols) if float(r[c]) > 0]
        ys = [float(r[c]) * 1e6 for c in cols if float(r[c]) > 0]
        p.line(xs, ys, label=lab, color=col, dash=dash, marker=True)
    return fig


def heat_fig(title, subtitle, rows, budgets, arms, labels):
    fig = sc.Figure(title, subtitle, ncols=len(budgets), panel_w=470, panel_h=50 + 30 * len(arms))
    for b in budgets:
        cfgs = sorted({(float(r["lr"]), float(r["wd"])) for r in rows if r["budget"] == b})
        m = []
        for a in arms:
            m.append([next((float(r["final"]) for r in rows if r["budget"] == b and r["arm"] == a and float(r["lr"]) == lr and float(r["wd"]) == wd), None) for lr, wd in cfgs])
        p = fig.add(sc.Panel(f"{b} fresh tokens", "learning rate, weight decay"))
        p.heatmap(m, labels, [f"{lr:g}, {wd:g}" for lr, wd in cfgs])
    return fig


# ---------------------------------------------------------------- groups
def group_1m_replication(runs):
    g = "1m_replication"
    desc = ("Self-play replication at 1m (`sp_1m_seed0`, 8192 rounds, 43B synthetic tokens, 5.2M per round) and the "
            "uniform-prior program baseline.")
    base = jsonl(runs / "sp_1m_seed0" / "log.jsonl")
    first = jsonl(RES / "1m_replication" / "sp_1m_seed0_eval_curve_first256.jsonl")
    rand = jsonl(RES / "1m_replication" / "sp_1m_seed0_eval_curve_random256.jsonl")
    held = jsonl(RES / "1m_reward_variant" / "sp_1m_seed0_eval_curve_heldout256.jsonl")
    unif = jsonl(RES / "1m_replication" / "uniform_1m_seed0_eval_curve_random256.jsonl")
    fig = sc.Figure("1m self-play: zero-shot transfer over training", "bits per byte of every 100-round checkpoint; thin = raw, thick = 5-checkpoint moving average", ncols=2, panel_w=470)
    for key, name in (("eval/dclm_bpb", "DCLM"), ("eval/metamath_bpb", "Metamath")):
        eval_panel(fig, name, [("self-play, first 256 records", *series(first, key), BLUE, None),
                               ("self-play, 256 random records", *series(rand, key), AQUA, None),
                               ("self-play, last 256 records (held out)", *series(held, key), VIOLET, None),
                               ("uniform-prior programs, 256 random records", *series(unif, key), GRAY, "6 4")])
    save(fig, g, "zero_shot_curves.svg", "Zero-shot DCLM and Metamath bpb over 8192 rounds under three eval samples, against the uniform-prior baseline.", desc)
    if base:
        fig = sc.Figure("1m self-play: the trainer's own evals", "every 100 rounds, first 256 records; arithmetic on a log axis", ncols=3, panel_w=400, panel_h=250)
        eval_panel(fig, "Arithmetic (log)", [("self-play", *series(base, "eval/arithmetic_bpb"), BLUE, None)], ylog=True)
        eval_panel(fig, "Metamath", [("self-play", *series(base, "eval/metamath_bpb"), BLUE, None)])
        eval_panel(fig, "Random bytes", [("self-play", *series(base, "eval/random_bytes_bpb"), BLUE, None)])
        save(fig, g, "trainer_evals.svg", "Arithmetic, Metamath and random-byte bpb from the trainer's evals.")
        save(icl_rounds_fig("1m self-play: in-context learning over training", "accuracy on synthetic ICL tasks every 500 rounds; 32 shots solid, 16 dashed, 8 dotted", [("self-play", base, BLUE)]),
             g, "icl_vs_rounds.svg", "ICL accuracy per task over training at 8, 16 and 32 in-context examples.")
        last = max(r["round"] for r in base if "icl/stack/m32" in r)
        save(icl_shots_fig("1m self-play: accuracy against the number of in-context examples", f"final checkpoint (round {last})", [("self-play", base, BLUE)], last),
             g, "icl_vs_shots.svg", "Final-checkpoint ICL accuracy as a function of the number of in-context examples.")
        save(dynamics_fig("1m self-play: training dynamics", "50-round moving averages over the full 8192-round run", [("self-play", base, BLUE)]),
             g, "dynamics.svg", "Learner, program, generator, reward and archive statistics over training.")
    else:
        skip(g, "trainer_evals/icl/dynamics", "no local log.jsonl for sp_1m_seed0")


def group_1m_plateau():
    g = "1m_prepretraining_plateau"
    desc = ("Paper Appendix A protocol at 1m: train on the 270 MB DCLM stream until the validation bpb plateaus, from scratch "
            "over initialisation settings x LR x WD, and warm-started from self-play checkpoints (`results/1m_prepretraining_plateau`).")
    rows = csv_rows(RES / "1m_prepretraining_plateau" / "results.csv")
    if not rows:
        return skip(g, "*", "no results.csv")
    settings = []
    for r in rows:
        lab = f"{r['arm']} {r['setting']}".replace(" -", " final")
        if lab not in settings:
            settings.append(lab)
    def best(lab):
        c = [r for r in rows if f"{r['arm']} {r['setting']}".replace(" -", " final") == lab]
        return min(c, key=lambda r: float(r["converged"]))
    order = sorted(settings, key=lambda s: float(best(s)["converged"]))
    fig = sc.Figure("1m plateau protocol: converged validation bpb by initialisation", "best of LR {1e-3, 3e-3, 1e-2} x WD {0, 0.1} per setting; lower is better", ncols=1, panel_w=560, panel_h=40 + 26 * len(order))
    p = fig.add(sc.Panel("", "converged validation bpb ↓", xfmt=lambda v: f"{v:.3f}"))
    p.dots(order, [float(best(s)["converged"]) for s in order], color=BLUE)
    save(fig, g, "converged_bpb_by_setting.svg", "Converged bpb of each scratch initialisation setting and of the self-play warm starts, best config per setting.", desc)
    cfgs = sorted({(float(r["lr"]), float(r["wd"])) for r in rows})
    m = [[next((float(r["converged"]) for r in rows if f"{r['arm']} {r['setting']}".replace(" -", " final") == s and float(r["lr"]) == lr and float(r["wd"]) == wd), None) for lr, wd in cfgs] for s in order]
    fig = sc.Figure("1m plateau protocol: converged bpb over the LR x WD grid", "converged validation bpb ↓; darker = lower (better)", ncols=1, panel_w=640, panel_h=50 + 30 * len(order))
    fig.add(sc.Panel("", "learning rate, weight decay")).heatmap(m, order, [f"{lr:g}, {wd:g}" for lr, wd in cfgs])
    save(fig, g, "lr_wd_grid.svg", "Converged bpb for every setting and every LR x WD configuration.")
    fig = sc.Figure("1m plateau protocol: tokens to the plateau", "best config per setting (same rows as above)", ncols=1, panel_w=560, panel_h=40 + 26 * len(order))
    p = fig.add(sc.Panel("", "DCLM tokens until the validation bpb plateaued ↓", xfmt=sc.fmt_tokens))
    p.dots(order, [float(best(s)["tokens_plateau_M"]) * 1e6 for s in order], color=BLUE)
    save(fig, g, "tokens_to_plateau.svg", "Tokens consumed before the plateau criterion fired, best config per setting.")


ARMS_1M = [("scratch_std0.02", "scratch, std 0.02", GRAY, None), ("scratch_std0.005", "scratch, std 0.005", GRAY, "6 4"),
           ("warm_final", "self-play final (round 8192)", VIOLET, None), ("warm_r2800", "self-play round 2800", AQUA, None),
           ("warm_r4100", "self-play round 4100", BLUE, None)]


def group_1m_fresh(pp1m):
    g = "1m_prepretraining_fresh"
    desc = ("Fresh-token protocol at 1m: one pass over the first 1B or 2B bytes of a 2B-byte DCLM stream (constant LR, 200-step "
            "cosine decay at the end), best of LR {3e-3, 1e-2} x WD {0, 0.1}, validated on a disjoint stream "
            "(`results/1m_prepretraining_fresh`).")
    rows = csv_rows(RES / "1m_prepretraining_fresh" / "results.csv")
    if not rows:
        return skip(g, "*", "no results.csv")
    arms = [a for a, *_ in ARMS_1M]
    labels = [l for _, l, *_ in ARMS_1M]
    save(dots_fig("1m fresh-token protocol: final bpb by warm start", "best LR x WD per arm and budget; lower is better", rows, arms, labels),
         g, "final_bpb_by_arm.svg", "Final validation bpb after 1B and 2B fresh DCLM tokens for scratch and self-play warm starts.", desc)
    save(heat_fig("1m fresh-token protocol: sensitivity to LR and weight decay", "final validation bpb ↓; darker = lower (better)", rows, ("1B", "2B"), arms, labels),
         g, "lr_wd_sensitivity.svg", "Final bpb for every arm under each LR x WD configuration, per budget.")
    for b in ("2B", "1B"):
        if not sweep_curve(pp1m, best_row(rows, b, "warm_final")["name"])[0]:
            skip(g, f"val_curves_{b}.svg", "no local sweep logs")
            continue
        fig = sc.Figure(f"1m fresh-token protocol: validation curves, {b} budget", "best LR x WD per arm; left: whole epoch, right: last 60% with a tight axis", ncols=2, panel_w=470, panel_h=300)
        curves_panel(fig, "Whole epoch", pp1m, rows, b, ARMS_1M, (1.6, 3.0))
        curves_panel(fig, "Last 60% of the epoch", pp1m, rows, b, ARMS_1M, (1.65, 1.95), frac_from=0.4)
        save(fig, g, f"val_curves_{b}.svg", f"Validation bpb against fresh tokens at the {b} budget for scratch and self-play warm starts.")
    save(thresholds_fig("1m fresh-token protocol: tokens to reach a bpb threshold", "2B-budget runs, best LR x WD per arm; a missing point = threshold not reached", rows, "2B", ARMS_1M, None),
         g, "tokens_to_threshold.svg", "Fresh tokens needed to first reach each validation bpb threshold (2B runs).")


ARMS_25M = [("scratch_std0.02", "scratch, std 0.02", GRAY, None), ("scratch_std0.005", "scratch, std 0.005", GRAY, "6 4"),
            ("sp_1B", "self-play round 200 (1.05B synthetic)", BLUE, None), ("sp_2B", "self-play round 400 (2.1B)", AQUA, None),
            ("sp_final", "self-play final, round 6560 (34.4B)", VIOLET, None),
            ("declref_2B", "DeclRef, 2B", ORANGE, None), ("nca_2B", "NCA, 2B", YELLOW, None), ("kdyck_2B", "k-Dyck, 2B", MAGENTA, None)]


def group_25m(runs, pp25m):
    g = "25m_study"
    desc = ("25m study: the self-play run `sp_25m_seed0` (6560 rounds, 34.4B synthetic tokens, 5.2M per round), the k-Dyck, NCA and "
            "DeclRef curricula of Rethinking Pre-pretraining (2B synthetic tokens each), the fresh-DCLM downstream comparison "
            "(one pass over 1B or 2B tokens, best of LR {1e-3, 3e-3}, WD 0.1) and the compute-matched scratch baselines "
            "(`results/25m`).")
    ev = jsonl(RES / "25m" / "sp_25m_seed0_eval_curve_first256.jsonl")
    fig = sc.Figure("25m self-play: zero-shot transfer over training", "trainer evals every 50 rounds, first 256 records; thin = raw, thick = 5-checkpoint moving average", ncols=2, panel_w=470)
    eval_panel(fig, "DCLM", [("self-play", *series(ev, "eval/dclm_bpb"), BLUE, None)])
    eval_panel(fig, "Metamath", [("self-play", *series(ev, "eval/metamath_bpb"), BLUE, None)])
    eval_panel(fig, "Arithmetic (log)", [("self-play", *series(ev, "eval/arithmetic_bpb"), BLUE, None)], ylog=True)
    eval_panel(fig, "Random bytes", [("self-play", *series(ev, "eval/random_bytes_bpb"), BLUE, None)])
    save(fig, g, "zero_shot_curves.svg", "Zero-shot bpb on DCLM, Metamath, arithmetic and random bytes over 6560 rounds.", desc)
    save(icl_rounds_fig("25m self-play: in-context learning over training", "every 500 rounds; 32 shots solid, 16 dashed, 8 dotted", [("self-play", ev, BLUE)]),
         g, "icl_vs_rounds.svg", "ICL accuracy per task over training at 8, 16 and 32 in-context examples.")
    save(icl_shots_fig("25m self-play: accuracy against the number of in-context examples", "final checkpoint (round 6559)", [("self-play", ev, BLUE)], 6559),
         g, "icl_vs_shots.svg", "Final-checkpoint ICL accuracy as a function of the number of in-context examples.")
    base = jsonl(runs / "sp_25m_seed0" / "log.jsonl")
    if base:
        save(dynamics_fig("25m self-play: training dynamics", "50-round moving averages over the 6560-round run", [("self-play", base, BLUE)]),
             g, "dynamics.svg", "Learner, program, generator, reward and archive statistics over training.")
    else:
        skip(g, "dynamics.svg", "no local log.jsonl for sp_25m_seed0")
    ppt = {k: [r for r in jsonl(Path(pp25m) / "ppt" / k / "log.jsonl") if "val_bpb" in r] for k in ("declref", "nca", "kdyck")}
    if any(ppt.values()):
        fig = sc.Figure("25m curricula: pre-pretraining on each synthetic stream", "validation bpb on a held-out stream of the same curriculum (levels are not comparable across curricula); 2B tokens, LR 1e-3", ncols=1, panel_w=560, panel_h=300)
        p = fig.add(sc.Panel("", "synthetic tokens", "validation bpb ↓ (log)", ylog=True, xfmt=sc.fmt_tokens))
        for k, lab, col in (("declref", "DeclRef", ORANGE), ("nca", "NCA", YELLOW), ("kdyck", "k-Dyck", MAGENTA)):
            if ppt[k]:
                p.line([r["tokens"] for r in ppt[k]], [r["val_bpb"] for r in ppt[k]], label=lab, color=col)
        save(fig, g, "curricula_pretraining.svg", "Training curves of the three curricula on their own held-out streams.")
    rows = csv_rows(RES / "25m" / "downstream_fresh_dclm_results.csv")
    if not rows:
        return skip(g, "downstream*", "no downstream results.csv")
    arms = [a for a, *_ in ARMS_25M]
    labels = [l for _, l, *_ in ARMS_25M]
    save(dots_fig("25m downstream: final bpb by warm start", "fresh DCLM, one epoch; best of LR {1e-3, 3e-3}, WD 0.1; lower is better", rows, arms, labels),
         g, "downstream_final_bpb.svg", "Final validation bpb after 1B and 2B fresh DCLM tokens for scratch, self-play checkpoints and the curricula.")
    save(heat_fig("25m downstream: sensitivity to the learning rate", "final validation bpb ↓; darker = lower (better)", rows, ("1B", "2B"), arms, labels),
         g, "downstream_lr_sensitivity.svg", "Final bpb per arm at LR 1e-3 and 3e-3.")
    sp_arms = [ARMS_25M[0], ARMS_25M[2], ARMS_25M[3], ARMS_25M[4]]
    cur_arms = [ARMS_25M[0], ARMS_25M[5], ARMS_25M[6], ARMS_25M[7]]
    for b in ("2B", "1B"):
        if not sweep_curve(pp25m, best_row(rows, b, "sp_1B")["name"])[0]:
            skip(g, f"downstream_val_curves_{b}.svg", "no local sweep logs")
            continue
        fig = sc.Figure(f"25m downstream: validation curves, {b} budget", "best LR per arm; top row: whole epoch, bottom row: last 60% with a tight axis", ncols=2, panel_w=470, panel_h=290)
        curves_panel(fig, "Self-play warm starts vs scratch", pp25m, rows, b, sp_arms, (1.3, 2.2))
        curves_panel(fig, "Curricula vs scratch", pp25m, rows, b, cur_arms, (1.3, 2.2))
        curves_panel(fig, "Self-play warm starts, last 60%", pp25m, rows, b, sp_arms, (1.35, 1.55), frac_from=0.4)
        curves_panel(fig, "Curricula, last 60%", pp25m, rows, b, cur_arms, (1.35, 1.55), frac_from=0.4)
        save(fig, g, f"downstream_val_curves_{b}.svg", f"Validation bpb against fresh tokens at the {b} budget: self-play checkpoints and curricula against scratch.")
    save(thresholds_fig("25m downstream: tokens to reach a bpb threshold", "2B-budget runs, best LR per arm; a missing point = threshold not reached", rows, "2B", ARMS_25M, None),
         g, "downstream_tokens_to_threshold.svg", "Fresh tokens needed to first reach each validation bpb threshold (2B runs).")
    # matched compute
    sb = {b: best_row(rows, b, "scratch_std0.02") for b in ("2B", "3.05B", "4.1B", "36.4B")}
    if all(sb.values()):
        tok = {"2B": 2e9, "3.05B": 3.05e9, "4.1B": 4.1e9, "36.4B": 36.4e9}
        fig = sc.Figure("25m compute-matched comparison", "x = all tokens the arm ever consumed (synthetic + 2B fresh DCLM); scratch runs consume fresh DCLM only", ncols=2, panel_w=470, panel_h=320)
        p = fig.add(sc.Panel("Final bpb against total tokens", "total tokens consumed (log)", "final validation bpb ↓", xlog=True, xfmt=sc.fmt_tokens, ylim=(1.27, 1.43)))
        p.line([tok[b] for b in sb], [float(sb[b]["final"]) for b in sb], label="scratch on fresh DCLM (2B, 3.05B, 4.1B, 36.4B)", color=GRAY, marker=True)
        spv = {a: best_row(rows, "2B", a) for a in ("sp_1B", "sp_2B", "sp_final")}
        p.points([3.05e9, 4.1e9, 36.4e9], [float(spv[a]["final"]) for a in ("sp_1B", "sp_2B", "sp_final")], label="self-play warm start + 2B fresh", color=BLUE, labels=["round 200", "round 400", "final"])
        rdv = {a: best_row(rows, "2B", a) for a in ("sp_rawdelta_1B", "sp_rawdelta_2B", "sp_rawdelta_final")}
        if all(rdv.values()):
            p.points([3.05e9 * 1.06, 4.1e9 * 1.06, 36.4e9 * 1.06], [float(rdv[a]["final"]) for a in rdv], label="self-play without the AdamW step operator + 2B fresh", color=ORANGE)
        for a, lab, col in (("declref_2B", "DeclRef", ORANGE), ("nca_2B", "NCA", YELLOW), ("kdyck_2B", "k-Dyck", MAGENTA)):
            r = best_row(rows, "2B", a)
            if r:
                p.points([4e9], [float(r["final"])], label=f"{lab} 2B + 2B fresh", color=col)
        hist = jsonl(RES / "25m" / "scratch_36.4B_fresh_dclm_history.jsonl")
        if hist:
            p2 = fig.add(sc.Panel("Scratch, 36.4B fresh tokens: the training curve", "fresh DCLM tokens (log)", "validation bpb ↓", xlog=True, xfmt=sc.fmt_tokens, ylim=(1.25, 1.8), xlim=(5e7, 4e10)))
            p2.line([r["tokens"] for r in hist if r["tokens"] > 0], [r["val_bpb"] for r in hist if r["tokens"] > 0], label="scratch, LR 1e-3, constant then 200-step decay", color=GRAY)
            p2.hline(float(spv["sp_1B"]["final"]), color=BLUE, label=f"self-play round 200 + 2B fresh: {float(spv['sp_1B']['final']):.3f}")
            p2.hline(float(spv["sp_final"]["final"]), color=VIOLET, label=f"self-play final + 2B fresh: {float(spv['sp_final']['final']):.3f}")
            p2.vline(3.05e9, label="3.05B")
            p2.vline(4.1e9, label="4.1B")
        save(fig, g, "matched_compute.svg", "Final bpb against total tokens consumed, scratch against warm starts, and the 36.4B scratch curve with the warm-start results as reference lines.")


def group_1m_reward_variant(runs, pp1m):
    g = "1m_reward_variant"
    desc = ("Ablation at 1m: the learning-progress reward without the AdamW step operator (`--tangent_precond identity`, run "
            "`sp_1m_rawdelta_seed0`, 4096 rounds) against the baseline at matching rounds (`results/1m_reward_variant`).")
    base_h = jsonl(RES / "1m_reward_variant" / "sp_1m_seed0_eval_curve_heldout256.jsonl")
    var_h = jsonl(RES / "1m_reward_variant" / "sp_1m_rawdelta_seed0_eval_curve_heldout256.jsonl")
    fig = sc.Figure("1m reward ablation: zero-shot transfer, held-out records", "last 256 records of each file, every 100 rounds up to 4100; thin = raw, thick = 5-checkpoint moving average", ncols=2, panel_w=470)
    for key, name in (("eval/dclm_bpb", "DCLM"), ("eval/metamath_bpb", "Metamath")):
        eval_panel(fig, name, [("baseline (AdamW step operator)", *series(base_h, key, 4100), BLUE, None),
                               ("no step operator (raw displacement)", *series(var_h, key, 4100), ORANGE, None)])
    save(fig, g, "zero_shot_heldout.svg", "Held-out DCLM and Metamath bpb, baseline against the no-step-operator variant.", desc)
    base = jsonl(runs / "sp_1m_seed0" / "log.jsonl")
    var = jsonl(runs / "sp_1m_rawdelta_seed0" / "log.jsonl")
    if base and var:
        rr = [("baseline", base, BLUE), ("no step operator", var, ORANGE)]
        fig = sc.Figure("1m reward ablation: arithmetic and in-context learning", "trainer evals every 100 rounds (arithmetic, log axis) and ICL accuracy at 32 shots every 500 rounds, up to round 4100", ncols=4, panel_w=330, panel_h=230)
        eval_panel(fig, "Arithmetic (log)", [(l, *series(r, "eval/arithmetic_bpb", 4100), c, None) for l, r, c in rr], ylog=True)
        eval_panel(fig, "Random bytes", [(l, *series(r, "eval/random_bytes_bpb", 4100), c, None) for l, r, c in rr])
        for t in ICL_TASKS:
            p = fig.add(sc.Panel(f"ICL: {ICL_NAMES[t]}, 32 shots", "round", "accuracy ↑", ylim=(0, 1.02)))
            for l, r, c in rr:
                xs, ys = series([x for x in r if f"icl/{t}/m32" in x], f"icl/{t}/m32", 4100)
                p.line(xs, ys, label=l, color=c, marker=True)
        save(fig, g, "arithmetic_and_icl.svg", "Arithmetic and random-byte bpb and ICL accuracy at 32 shots, both runs.")
        save(dynamics_fig("1m reward ablation: training dynamics", "50-round moving averages, rounds 0 to 4096", rr, max_round=4096),
             g, "dynamics.svg", "Training dynamics of the baseline and the variant over the same 4096 rounds.")
    else:
        skip(g, "arithmetic_and_icl/dynamics", "no local logs")
    rows = csv_rows(RES / "1m_prepretraining_fresh" / "results.csv")
    arms = [("scratch_std0.02", "scratch, std 0.02", GRAY, None), ("warm_r2800", "baseline round 2800", AQUA, None), ("warm_r4100", "baseline round 4100", BLUE, None),
            ("warm_final", "baseline final (8192)", VIOLET, None), ("warm_rawdelta_r2800", "no step operator, round 2800", YELLOW, None), ("warm_rawdelta_r4096", "no step operator, round 4096", ORANGE, None)]
    if rows:
        fig = sc.Figure("1m reward ablation: warm-start value on fresh DCLM", "best LR x WD per arm; left: final bpb; right: last 60% of the epoch at each budget", ncols=3, panel_w=400, panel_h=300)
        p = fig.add(sc.Panel("Final bpb", "final validation bpb ↓", xfmt=lambda v: f"{v:.3f}"))
        for b, col in (("1B", BLUE), ("2B", ORANGE)):
            p.dots([l for _, l, *_ in arms], [float(best_row(rows, b, a)["final"]) if best_row(rows, b, a) else None for a, *_ in arms], label=f"{b} fresh tokens", color=col)
        sub = [arms[0], arms[2], arms[5], arms[4]]
        if sweep_curve(pp1m, best_row(rows, "2B", "warm_r4100")["name"])[0]:
            curves_panel(fig, "2B budget, last 60%", pp1m, rows, "2B", sub, (1.65, 1.85), frac_from=0.4)
            curves_panel(fig, "1B budget, last 60%", pp1m, rows, "1B", sub, (1.68, 1.9), frac_from=0.4)
        save(fig, g, "downstream.svg", "Warm-start value: final bpb per arm and the tail of the validation curves at both budgets.")


def group_25m_reward_variant(runs, pp25m):
    g = "25m_reward_variant"
    desc = ("Ablation at 25m: the reward without the AdamW step operator for the full 6560 rounds (`sp_25m_rawdelta_seed0`) "
            "against `sp_25m_seed0` (`results/25m/reward_variant.md`).")
    base = jsonl(RES / "25m" / "sp_25m_seed0_eval_curve_first256.jsonl")
    var = jsonl(RES / "25m" / "sp_25m_rawdelta_seed0_eval_curve_first256.jsonl")
    rr = [("baseline (AdamW step operator)", base, BLUE), ("no step operator (raw displacement)", var, ORANGE)]
    fig = sc.Figure("25m reward ablation: zero-shot transfer over training", "trainer evals every 50 rounds, first 256 records; thin = raw, thick = 5-checkpoint moving average", ncols=2, panel_w=470)
    for key, name, ylog in (("eval/dclm_bpb", "DCLM", False), ("eval/metamath_bpb", "Metamath", False), ("eval/arithmetic_bpb", "Arithmetic (log)", True), ("eval/random_bytes_bpb", "Random bytes", False)):
        eval_panel(fig, name, [(l, *series(r, key), c, None) for l, r, c in rr], ylog=ylog)
    save(fig, g, "zero_shot_curves.svg", "Zero-shot bpb over training for both runs.", desc)
    save(icl_rounds_fig("25m reward ablation: in-context learning over training", "every 500 rounds; 32 shots solid, 16 dashed, 8 dotted", [("baseline", base, BLUE), ("no step operator", var, ORANGE)]),
         g, "icl_vs_rounds.svg", "ICL accuracy per task over training for both runs.")
    bl, vl = jsonl(runs / "sp_25m_seed0" / "log.jsonl"), jsonl(runs / "sp_25m_rawdelta_seed0" / "log.jsonl")
    if bl and vl:
        save(dynamics_fig("25m reward ablation: training dynamics", "50-round moving averages over the 6560-round runs", [("baseline", bl, BLUE), ("no step operator", vl, ORANGE)]),
             g, "dynamics.svg", "Training dynamics of both runs.")
    else:
        skip(g, "dynamics.svg", "no local logs")
    rows = csv_rows(RES / "25m" / "downstream_fresh_dclm_results.csv")
    if not rows:
        return
    arms = [("scratch_std0.02", "scratch, std 0.02", GRAY, None), ("sp_1B", "baseline round 200", BLUE, None), ("sp_rawdelta_1B", "no step operator, round 200", BLUE, "6 4"),
            ("sp_2B", "baseline round 400", AQUA, None), ("sp_rawdelta_2B", "no step operator, round 400", AQUA, "6 4"),
            ("sp_final", "baseline final", VIOLET, None), ("sp_rawdelta_final", "no step operator, final", VIOLET, "6 4")]
    fig = sc.Figure("25m reward ablation: warm-start value on fresh DCLM", "best LR per arm, WD 0.1; left: final bpb; right: last 60% of the epoch (dashed = no step operator)", ncols=3, panel_w=400, panel_h=300)
    p = fig.add(sc.Panel("Final bpb", "final validation bpb ↓", xfmt=lambda v: f"{v:.3f}"))
    for b, col in (("1B", BLUE), ("2B", ORANGE)):
        p.dots([l for _, l, *_ in arms], [float(best_row(rows, b, a)["final"]) if best_row(rows, b, a) else None for a, *_ in arms], label=f"{b} fresh tokens", color=col)
    sub = [arms[0], arms[1], arms[2], arms[5], arms[6]]
    if sweep_curve(pp25m, best_row(rows, "2B", "sp_1B")["name"])[0]:
        curves_panel(fig, "2B budget, last 60%", pp25m, rows, "2B", sub, (1.35, 1.5), frac_from=0.4)
        curves_panel(fig, "1B budget, last 60%", pp25m, rows, "1B", sub, (1.4, 1.56), frac_from=0.4)
    save(fig, g, "downstream.svg", "Warm-start value: final bpb per arm and the tail of the validation curves at both budgets.")
    fig = sc.Figure("25m: warm-start value against the self-play checkpoint", "final bpb after 1B / 2B fresh DCLM tokens, by the synthetic tokens the checkpoint had consumed (log); dashed lines = scratch", ncols=1, panel_w=560, panel_h=320)
    p = fig.add(sc.Panel("", "synthetic tokens consumed by the checkpoint (log)", "final validation bpb ↓", xlog=True, xfmt=sc.fmt_tokens))
    xs = [1.05e9, 2.1e9, 34.4e9]
    for b, col in (("1B", BLUE), ("2B", ORANGE)):
        p.line(xs, [float(best_row(rows, b, a)["final"]) for a in ("sp_1B", "sp_2B", "sp_final")], label=f"baseline, {b} fresh", color=col, marker=True)
        if best_row(rows, b, "sp_rawdelta_final"):
            p.line(xs, [float(best_row(rows, b, a)["final"]) for a in ("sp_rawdelta_1B", "sp_rawdelta_2B", "sp_rawdelta_final")], label=f"no step operator, {b} fresh", color=col, dash="6 4", marker=True)
        p.hline(float(best_row(rows, b, "scratch_std0.02")["final"]), color=col, label=f"scratch, {b} fresh")
    save(fig, g, "warm_start_vs_synthetic_tokens.svg", "Final bpb as a function of how far into self-play the warm-start checkpoint was taken, both runs.")


def group_1m_generator_variant(runs, pp1m):
    g = "1m_generator_variant"
    desc = ("Ablation at 1m: the generator updated on its own fresh samples only (`--gen_update fresh`, run `sp_1m_genfresh_seed0`, "
            "4096 rounds), against the baseline and the reward ablation at matching rounds (`results/1m_generator_variant`).")
    H = {"baseline": jsonl(RES / "1m_reward_variant" / "sp_1m_seed0_eval_curve_heldout256.jsonl"),
         "no step operator": jsonl(RES / "1m_reward_variant" / "sp_1m_rawdelta_seed0_eval_curve_heldout256.jsonl"),
         "generator on fresh samples only": jsonl(RES / "1m_generator_variant" / "sp_1m_genfresh_seed0_eval_curve_heldout256.jsonl")}
    if not H["generator on fresh samples only"]:
        return skip(g, "*", "no held-out eval curve for sp_1m_genfresh_seed0")
    cols = {"baseline": BLUE, "no step operator": ORANGE, "generator on fresh samples only": AQUA}
    fig = sc.Figure("1m generator ablation: zero-shot transfer, held-out records", "last 256 records of each file, every 100 rounds up to 4100; thin = raw, thick = 5-checkpoint moving average", ncols=2, panel_w=470)
    for key, name in (("eval/dclm_bpb", "DCLM"), ("eval/metamath_bpb", "Metamath")):
        eval_panel(fig, name, [(l, *series(r, key, 4100), cols[l], None) for l, r in H.items()])
    save(fig, g, "zero_shot_heldout.svg", "Held-out DCLM and Metamath bpb for the baseline and both 1m ablations.", desc)
    L = {"baseline": jsonl(runs / "sp_1m_seed0" / "log.jsonl"), "no step operator": jsonl(runs / "sp_1m_rawdelta_seed0" / "log.jsonl"),
         "generator on fresh samples only": jsonl(runs / "sp_1m_genfresh_seed0" / "log.jsonl")}
    if all(L.values()):
        rr = [(l, r, cols[l]) for l, r in L.items()]
        fig = sc.Figure("1m generator ablation: arithmetic and in-context learning", "trainer evals every 100 rounds (arithmetic, log axis) and ICL accuracy at 32 shots every 500 rounds, up to round 4100", ncols=4, panel_w=330, panel_h=230)
        eval_panel(fig, "Arithmetic (log)", [(l, *series(r, "eval/arithmetic_bpb", 4100), c, None) for l, r, c in rr], ylog=True)
        eval_panel(fig, "Random bytes", [(l, *series(r, "eval/random_bytes_bpb", 4100), c, None) for l, r, c in rr])
        for t in ICL_TASKS:
            p = fig.add(sc.Panel(f"ICL: {ICL_NAMES[t]}, 32 shots", "round", "accuracy ↑", ylim=(0, 1.02)))
            for l, r, c in rr:
                xs, ys = series([x for x in r if f"icl/{t}/m32" in x], f"icl/{t}/m32", 4100)
                p.line(xs, ys, label=l, color=c, marker=True)
        save(fig, g, "arithmetic_and_icl.svg", "Arithmetic and random-byte bpb and ICL accuracy at 32 shots, three runs.")
        save(dynamics_fig("1m generator ablation: training dynamics", "50-round moving averages, rounds 0 to 4096", rr, max_round=4096),
             g, "dynamics.svg", "Training dynamics of the three runs over the same 4096 rounds.")
    else:
        skip(g, "arithmetic_and_icl/dynamics", "no local logs")
    rows = csv_rows(RES / "1m_prepretraining_fresh" / "results.csv")
    arms = [("scratch_std0.02", "scratch, std 0.02", GRAY, None), ("warm_r2800", "baseline round 2800", AQUA, "6 4"), ("warm_r4100", "baseline round 4100", BLUE, None),
            ("warm_final", "baseline final (8192)", VIOLET, None), ("warm_rawdelta_r4096", "no step operator, round 4096", ORANGE, None),
            ("warm_genfresh_r2800", "generator on fresh only, round 2800", AQUA, "2 3"), ("warm_genfresh_r4096", "generator on fresh only, round 4096", AQUA, None)]
    if rows and best_row(rows, "2B", "warm_genfresh_r4096"):
        fig = sc.Figure("1m generator ablation: warm-start value on fresh DCLM", "best LR x WD per arm; left: final bpb; right: last 60% of the epoch at each budget", ncols=3, panel_w=400, panel_h=300)
        p = fig.add(sc.Panel("Final bpb", "final validation bpb ↓", xfmt=lambda v: f"{v:.3f}"))
        for b, col in (("1B", BLUE), ("2B", ORANGE)):
            p.dots([l for _, l, *_ in arms], [float(best_row(rows, b, a)["final"]) if best_row(rows, b, a) else None for a, *_ in arms], label=f"{b} fresh tokens", color=col)
        sub = [arms[0], arms[2], arms[4], arms[6]]
        if sweep_curve(pp1m, best_row(rows, "2B", "warm_r4100")["name"])[0]:
            curves_panel(fig, "2B budget, last 60%", pp1m, rows, "2B", sub, (1.65, 1.85), frac_from=0.4)
            curves_panel(fig, "1B budget, last 60%", pp1m, rows, "1B", sub, (1.68, 1.9), frac_from=0.4)
        save(fig, g, "downstream.svg", "Warm-start value: final bpb per arm and the tail of the validation curves at both budgets.")


def group_1m_mutation_variant(runs, pp1m):
    g = "1m_mutation_variant"
    desc = ("Ablation at 1m: no mutations (`--mut_frac 0`, run `sp_1m_nomut_seed0`, 4096 rounds; all 1024 programs per round are fresh "
            "generator samples), against the baseline and the two earlier 1m ablations at matching rounds (`results/1m_mutation_variant`).")
    NOMUT = "no mutations"
    H = {"baseline": jsonl(RES / "1m_reward_variant" / "sp_1m_seed0_eval_curve_heldout256.jsonl"),
         "no step operator": jsonl(RES / "1m_reward_variant" / "sp_1m_rawdelta_seed0_eval_curve_heldout256.jsonl"),
         "generator on fresh only": jsonl(RES / "1m_generator_variant" / "sp_1m_genfresh_seed0_eval_curve_heldout256.jsonl"),
         NOMUT: jsonl(RES / "1m_mutation_variant" / "sp_1m_nomut_seed0_eval_curve_heldout256.jsonl")}
    if not H[NOMUT]:
        return skip(g, "*", "no held-out eval curve for sp_1m_nomut_seed0")
    cols = {"baseline": BLUE, "no step operator": ORANGE, "generator on fresh only": AQUA, NOMUT: YELLOW}
    fig = sc.Figure("1m mutation ablation: zero-shot transfer, held-out records", "last 256 records of each file, every 100 rounds up to 4100; thin = raw, thick = 5-checkpoint moving average", ncols=2, panel_w=470)
    for key, name in (("eval/dclm_bpb", "DCLM"), ("eval/metamath_bpb", "Metamath")):
        eval_panel(fig, name, [(l, *series(r, key, 4100), cols[l], None) for l, r in H.items()])
    save(fig, g, "zero_shot_heldout.svg", "Held-out DCLM and Metamath bpb for the baseline and the three 1m ablations.", desc)
    L = {"baseline": jsonl(runs / "sp_1m_seed0" / "log.jsonl"), "no step operator": jsonl(runs / "sp_1m_rawdelta_seed0" / "log.jsonl"),
         "generator on fresh only": jsonl(runs / "sp_1m_genfresh_seed0" / "log.jsonl"), NOMUT: jsonl(runs / "sp_1m_nomut_seed0" / "log.jsonl")}
    if all(L.values()):
        rr = [(l, r, cols[l]) for l, r in L.items()]
        fig = sc.Figure("1m mutation ablation: arithmetic and in-context learning", "trainer evals every 100 rounds (arithmetic, log axis) and ICL accuracy at 32 shots every 500 rounds, up to round 4100", ncols=4, panel_w=330, panel_h=230)
        eval_panel(fig, "Arithmetic (log)", [(l, *series(r, "eval/arithmetic_bpb", 4100), c, None) for l, r, c in rr], ylog=True)
        eval_panel(fig, "Random bytes", [(l, *series(r, "eval/random_bytes_bpb", 4100), c, None) for l, r, c in rr])
        for t in ICL_TASKS:
            p = fig.add(sc.Panel(f"ICL: {ICL_NAMES[t]}, 32 shots", "round", "accuracy ↑", ylim=(0, 1.02)))
            for l, r, c in rr:
                xs, ys = series([x for x in r if f"icl/{t}/m32" in x], f"icl/{t}/m32", 4100)
                p.line(xs, ys, label=l, color=c, marker=True)
        save(fig, g, "arithmetic_and_icl.svg", "Arithmetic and random-byte bpb and ICL accuracy at 32 shots, four runs.")
        save(dynamics_fig("1m mutation ablation: training dynamics", "50-round moving averages, rounds 0 to 4096", rr, max_round=4096),
             g, "dynamics.svg", "Training dynamics of the four runs over the same 4096 rounds.")
        save(dynamics_fig("1m mutation ablation: the generator's policy-gradient term", "50-round moving averages, log axes", rr, max_round=4096,
                          keys=[("gen/rho_mean", "Mean importance ratio, PG rows (log)", True), ("gen/loss_pg", "Generator PG loss (signed)", False),
                                ("gen/grad_norm", "Generator gradient norm before clipping (log)", True), ("reward/fresh_mean", "Reward mean, fresh programs (log)", True)]),
             g, "generator_pg.svg", "Importance ratios, PG loss and generator gradient norm: the huge ratios in the baseline come from replayed mutation programs.")
    else:
        skip(g, "arithmetic_and_icl/dynamics", "no local logs")
    rows = csv_rows(RES / "1m_prepretraining_fresh" / "results.csv")
    arms = [("scratch_std0.02", "scratch, std 0.02", GRAY, None), ("warm_r2800", "baseline round 2800", BLUE, "6 4"), ("warm_r4100", "baseline round 4100", BLUE, None),
            ("warm_rawdelta_r4096", "no step operator, round 4096", ORANGE, None), ("warm_genfresh_r4096", "generator on fresh only, round 4096", AQUA, None),
            ("warm_nomut_r2800", "no mutations, round 2800", YELLOW, "6 4"), ("warm_nomut_r4096", "no mutations, round 4096", YELLOW, None)]
    if rows and best_row(rows, "2B", "warm_nomut_r4096"):
        fig = sc.Figure("1m mutation ablation: warm-start value on fresh DCLM", "best LR x WD per arm; left: final bpb; right: last 60% of the epoch at each budget", ncols=3, panel_w=400, panel_h=300)
        p = fig.add(sc.Panel("Final bpb", "final validation bpb ↓", xfmt=lambda v: f"{v:.3f}"))
        for b, col in (("1B", BLUE), ("2B", ORANGE)):
            p.dots([l for _, l, *_ in arms], [float(best_row(rows, b, a)["final"]) if best_row(rows, b, a) else None for a, *_ in arms], label=f"{b} fresh tokens", color=col)
        sub = [arms[0], arms[2], arms[3], arms[4], arms[6]]
        if sweep_curve(pp1m, best_row(rows, "2B", "warm_r4100")["name"])[0]:
            curves_panel(fig, "2B budget, last 60%", pp1m, rows, "2B", sub, (1.65, 1.85), frac_from=0.4)
            curves_panel(fig, "1B budget, last 60%", pp1m, rows, "1B", sub, (1.68, 1.9), frac_from=0.4)
        save(fig, g, "downstream.svg", "Warm-start value: final bpb per arm and the tail of the validation curves at both budgets.")


def group_overview():
    g = "overview"
    desc = "Cross-experiment summaries."
    r1 = csv_rows(RES / "1m_prepretraining_fresh" / "results.csv")
    r25 = csv_rows(RES / "25m" / "downstream_fresh_dclm_results.csv")
    if r1 and r25:
        def margin(rows, scratch, warm, b):
            s = min(float(best_row(rows, b, a)["final"]) for a in scratch if best_row(rows, b, a))
            w = min(float(best_row(rows, b, a)["final"]) for a in warm if best_row(rows, b, a))
            return s - w
        fig = sc.Figure("Self-play warm-start margin over scratch, by learner size", "scratch bpb minus the best self-play warm start (paper reward), fresh DCLM, best config per arm", ncols=1, panel_w=560, panel_h=300)
        p = fig.add(sc.Panel("", "", "bpb improvement over scratch ↑", yfmt=lambda v: f"{v:.3f}"))
        for i, (b, col) in enumerate((("1B", BLUE), ("2B", ORANGE))):
            p.bars(["1m learner", "25m learner"], [margin(r1, ("scratch_std0.02", "scratch_std0.005"), ("warm_final", "warm_r2800", "warm_r4100"), b),
                                                   margin(r25, ("scratch_std0.02", "scratch_std0.005"), ("sp_1B", "sp_2B", "sp_final"), b)], label=f"{b} fresh tokens", color=col, group=(i, 2), value_labels=True)
        save(fig, g, "selfplay_margin_by_scale.svg", "How much the best self-play warm start improves on scratch at 1m and 25m, at both fresh-token budgets.", desc)
    e1 = jsonl(RES / "1m_replication" / "sp_1m_seed0_eval_curve_first256.jsonl")
    e25 = jsonl(RES / "25m" / "sp_25m_seed0_eval_curve_first256.jsonl")
    if e1 and e25:
        fig = sc.Figure("Zero-shot transfer against synthetic tokens, 1m and 25m self-play", "first 256 records of each file; thin = raw, thick = 5-checkpoint moving average", ncols=2, panel_w=470)
        for key, name in (("eval/dclm_bpb", "DCLM"), ("eval/metamath_bpb", "Metamath")):
            p = fig.add(sc.Panel(name, "synthetic tokens (log)", "bits per byte ↓", xlog=True, xfmt=sc.fmt_tokens))
            for lab, rows, col in (("1m learner", e1, BLUE), ("25m learner", e25, ORANGE)):
                xs = [r["tokens"] for r in rows if key in r and r["tokens"] > 0]
                ys = [r[key] for r in rows if key in r and r["tokens"] > 0]
                p.line(xs, ys, color=col, width=1.25, opacity=0.4, legend=False)
                p.line(xs, sc.smooth(ys, 5), label=lab, color=col, width=2.2)
        save(fig, g, "zero_shot_1m_vs_25m.svg", "Zero-shot DCLM and Metamath bpb of the 1m and 25m self-play runs against synthetic tokens consumed.")


def write_index(out):
    out = Path(out)
    order = ["overview", "1m_replication", "1m_prepretraining_plateau", "1m_prepretraining_fresh", "25m_study", "1m_reward_variant", "25m_reward_variant", "1m_generator_variant", "1m_mutation_variant"]
    md = ["# Plots", "", "Generated by `scripts/plot_results.py` from the files in `results/` and the local run logs. SVG, one file per figure, grouped by experiment.", "", "Axis labels carry ↓ when lower is better (bits per byte, tokens to a threshold) and ↑ when higher is better (accuracy, improvement over scratch). The training-dynamics panels are descriptive and carry neither.", ""]
    html = ["<!doctype html><meta charset='utf-8'><title>self-play pretraining: plots</title>",
            "<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#f9f9f7}h2{margin-top:40px}figure{margin:16px 0}figcaption{color:#52514e;font-size:14px;margin-top:4px}img{max-width:100%;background:#fcfcfb}</style>",
            "<h1>Plots</h1><p>Axis labels carry ↓ when lower is better (bits per byte, tokens to a threshold) and ↑ when higher is better (accuracy, improvement over scratch). The training-dynamics panels are descriptive and carry neither.</p>"]
    for grp in order + [k for k in GROUPS if k not in order]:
        figs = [(f, c) for gg, f, c in FIGS if gg == grp]
        if not figs:
            continue
        md += [f"## {grp}", "", GROUPS.get(grp, ""), ""]
        html.append(f"<h2>{grp}</h2><p>{GROUPS.get(grp, '')}</p>")
        for f, c in figs:
            md += [f"### {f[:-4].replace('_', ' ')}", "", c, "", f"![{c}]({grp}/{f})", ""]
            html.append(f"<figure><img src='{grp}/{f}' alt='{c}'><figcaption><b>{grp}/{f}</b>: {c}</figcaption></figure>")
    (out / "README.md").write_text("\n".join(md))
    (out / "index.html").write_text("\n".join(html))
    print(f"  {out.relative_to(ROOT)}/README.md, index.html ({len(FIGS)} figures)")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "plots"))
    ap.add_argument("--runs", default="/mnt/localssd/spp-runs")
    ap.add_argument("--pp1m", default="/mnt/localssd/spp-pp1m-fresh")
    ap.add_argument("--pp25m", default="/mnt/localssd/spp-pp25m")
    ARGS = ap.parse_args()
    runs = Path(ARGS.runs)
    group_overview()
    group_1m_replication(runs)
    group_1m_plateau()
    group_1m_fresh(ARGS.pp1m)
    group_25m(runs, ARGS.pp25m)
    group_1m_reward_variant(runs, ARGS.pp1m)
    group_25m_reward_variant(runs, ARGS.pp25m)
    group_1m_generator_variant(runs, ARGS.pp1m)
    group_1m_mutation_variant(runs, ARGS.pp1m)
    write_index(ARGS.out)


if __name__ == "__main__":
    main()

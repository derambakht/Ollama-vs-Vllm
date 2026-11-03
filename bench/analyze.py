#!/usr/bin/env python3
"""
Turn results/*/summary.jsonl (+ gpu.csv, meta.json) into the LinkedIn-ready report:

  results/report/REPORT.md          all tables + environment block + chart links
  results/report/tables/*.md        one markdown table per traffic shape
  results/report/charts/*.png       16:9 PNGs, 1800x1012, ready for a LinkedIn post
  results/report/all-runs.csv       every run, flat, for your own charts

Numbers in tables are the MEDIAN across runs for a (label, mode, level) cell.
VRAM / temperature / clock come from gpu.csv rows inside each run's time window.

Usage:  python analyze.py [--results ../results] [--dark]
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- palette (validated, see docs)
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
             series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"])
DARK = dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
            series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"])
# Fixed slot per label: colour follows the entity, never its rank in the current chart.
LABEL_SLOT = {
    "vllm-bf16": 0, "ollama-default": 1, "ollama-tuned": 2, "vllm-awq-marlin": 3,
    "vllm-awq-plain": 4, "llamacpp": 5, "vllm-bf16-u050": 6, "vllm-peak": 7,
}
NICE = {
    "vllm-bf16": "vLLM (BF16)", "ollama-default": "Ollama default", "ollama-tuned": "Ollama tuned",
    "vllm-awq-marlin": "vLLM AWQ (Marlin)", "vllm-awq-plain": "vLLM AWQ (plain)", "llamacpp": "llama.cpp",
    "vllm-bf16-u050": "vLLM BF16, 50% VRAM", "vllm-peak": "vLLM peak",
}
MODE_X = {"batch": "Concurrent requests", "users": "Simultaneous users (closed loop, think time)",
          "rate": "Arrival rate (requests / s, open loop)"}
# Chart sets: the post needs 3 lines, not 7. Core = the headline comparison; extras = the Part-2 questions.
SETS = {
    "core":   ["ollama-default", "ollama-tuned", "vllm-bf16"],
    "extras": ["vllm-bf16", "vllm-awq-marlin", "vllm-bf16-u050"],
}


def color_for(label, theme):
    slot = LABEL_SLOT.get(label)
    if slot is None:
        used = set(LABEL_SLOT.values())
        slot = next(i for i in range(8) if i not in used)
        LABEL_SLOT[label] = slot
    return theme["series"][slot % 8]


# ----------------------------------------------------------------------------- load
def load(results: Path):
    rows = []
    for f in glob.glob(str(results / "*" / "summary.jsonl")):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no summary.jsonl under {results}")
    df = pd.DataFrame(rows)

    # join GPU monitor samples: epoch,mem_mib,util,temp,clock_mhz,power_w
    gpu_cols = ["vram_peak_gb", "temp_max_c", "clock_min_mhz", "power_avg_w", "util_avg_pct"]
    for c in gpu_cols:
        df[c] = None
    for label in df["label"].unique():
        gf = results / label / "gpu.csv"
        if not gf.exists():
            continue
        g = pd.read_csv(gf, names=["epoch", "mem", "util", "temp", "clock", "power"], comment="#")
        g = g.apply(pd.to_numeric, errors="coerce").dropna(subset=["epoch"])
        for i in df.index[df["label"] == label]:
            w = g[(g["epoch"] >= df.at[i, "t0_epoch"]) & (g["epoch"] <= df.at[i, "t1_epoch"])]
            if len(w):
                df.at[i, "vram_peak_gb"] = round(w["mem"].max() / 1024, 1)
                df.at[i, "temp_max_c"] = w["temp"].max()
                df.at[i, "clock_min_mhz"] = w["clock"].min()
                df.at[i, "power_avg_w"] = round(w["power"].mean())
                df.at[i, "util_avg_pct"] = round(w["util"].mean())
    for c in gpu_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    meta = {}
    for f in glob.glob(str(results / "*" / "meta.json")):
        meta[Path(f).parent.name] = json.loads(Path(f).read_text(encoding="utf-8"))
    return df, meta


# ----------------------------------------------------------------------------- tables
COLS = {
    "batch": ["ttft_med_ms", "ttft_p95_ms", "per_req_tok_s", "agg_tok_s", "lat_med_s", "lat_p95_s",
              "n_err", "vram_peak_gb", "temp_max_c", "clock_min_mhz"],
    "users": ["req_s", "agg_tok_s", "ttft_med_ms", "ttft_p95_ms", "lat_p95_s", "n_req", "n_err", "vram_peak_gb"],
    "rate": ["req_s", "agg_tok_s", "ttft_p95_ms", "lat_p95_s", "lat_p99_s", "n_req", "n_err"],
}
HEAD = {
    "ttft_med_ms": "TTFT med (ms)", "ttft_p95_ms": "TTFT p95 (ms)", "per_req_tok_s": "per-req tok/s",
    "agg_tok_s": "aggregate tok/s", "lat_med_s": "latency med (s)", "lat_p95_s": "latency p95 (s)",
    "lat_p99_s": "latency p99 (s)", "n_err": "errors", "n_req": "requests", "req_s": "served req/s",
    "vram_peak_gb": "VRAM peak (GB)", "temp_max_c": "temp max (C)", "clock_min_mhz": "SM clock min (MHz)",
}


def aggregate(df: pd.DataFrame):
    agg = {c: "median" for c in df.columns if df[c].dtype.kind in "fi"}
    for c in ("n_err", "n_req", "vram_peak_gb", "temp_max_c"):
        if c in agg:
            agg[c] = "max" if c != "clock_min_mhz" else "min"
    if "clock_min_mhz" in agg:
        agg["clock_min_mhz"] = "min"
    agg["run"] = "count"
    out = df.groupby(["mode", "label", "level"], as_index=False).agg(agg).rename(columns={"run": "runs"})
    order = {k: i for i, k in enumerate(LABEL_SLOT)}
    out["_o"] = out["label"].map(lambda l: order.get(l, 99))
    return out.sort_values(["mode", "_o", "level"]).drop(columns="_o")


def md_table(t: pd.DataFrame, mode: str) -> str:
    cols = [c for c in COLS[mode] if c in t.columns]
    sub = t[t["mode"] == mode][["label", "level"] + cols + ["runs"]].copy()
    if sub.empty:
        return ""
    sub["label"] = sub["label"].map(lambda l: NICE.get(l, l))
    hdr = ["config", {"batch": "conc", "users": "users", "rate": "offered req/s"}[mode]] + [HEAD[c] for c in cols] + ["runs"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    def decimals(c):
        if c == "level":
            return None                                   # 1, 16, 0.5
        if c.endswith("_ms") or c.endswith("_mhz") or c.endswith("_c") or c in ("n_err", "n_req", "runs"):
            return 0
        if c.endswith("tok_s") or c.endswith("_gb"):
            return 1
        return 2                                          # seconds, req/s
    for _, row in sub.iterrows():
        vals = []
        for c in ["label", "level"] + cols + ["runs"]:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                d = decimals(c)
                vals.append(f"{v:g}" if d is None else f"{v:.{d}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- charts
def style(theme):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 12,
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
        "axes.edgecolor": theme["axis"], "axes.labelcolor": theme["ink2"],
        "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": theme["grid"], "grid.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "legend.frameon": False, "text.color": theme["ink"],
    })


DPI = 150
FIG_W, FIG_H = 12, 6.75          # 1800 x 1012 px, LinkedIn 16:9
MARGINS = dict(left=0.08, right=0.97, top=0.82, bottom=0.14)


def new_fig(title, subtitle, xlabel, ylabel, theme, height=FIG_H, margins=None):
    m = margins or MARGINS
    fig, ax = plt.subplots(figsize=(FIG_W, height), dpi=DPI)
    fig.subplots_adjust(**m)
    fig.text(m["left"], 1 - 0.07 * FIG_H / height, title, fontsize=18, fontweight="semibold", color=theme["ink"], ha="left")
    fig.text(m["left"], 1 - 0.125 * FIG_H / height, subtitle, fontsize=12, color=theme["ink2"], ha="left")
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=False))
    return fig, ax


def px_per_unit(ax, axis="x"):
    """Data units -> pixels, so bar thickness can be capped at 24 px regardless of how many groups."""
    fig = ax.figure
    bbox = ax.get_position()
    if axis == "x":
        px = bbox.width * fig.get_figwidth() * fig.dpi
        lo, hi = ax.get_xlim()
    else:
        px = bbox.height * fig.get_figheight() * fig.dpi
        lo, hi = ax.get_ylim()
    return px / (hi - lo)


def fmt(v):
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"


def plain_log_axis(ax):
    from matplotlib.ticker import ScalarFormatter, NullLocator
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.set_minor_locator(NullLocator())


def footer(fig, text, theme):
    fig.text(0.08, 0.03, text, fontsize=9.5, color=theme["muted"], ha="left")


def pick(sub, only):
    present = set(sub["label"])
    if only:
        return [l for l in only if l in present]
    return [l for l in LABEL_SLOT if l in present] + sorted(present - set(LABEL_SLOT))


def chart_bars(t, mode, metric, title, subtitle, ylabel, out, theme, note, only=None):
    sub = t[t["mode"] == mode].dropna(subset=[metric])
    if only:
        sub = sub[sub["label"].isin(only)]
    if sub.empty:
        return
    labels = pick(sub, only)
    levels = sorted(sub["level"].unique())
    fig, ax = new_fig(title, subtitle, MODE_X[mode], ylabel, theme)
    n = len(labels)
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ppu = px_per_unit(ax, "x")
    bw, gap = 24 / ppu, 16 / ppu                     # 24 px bars; 16 px air so the value labels never collide
    for k, lab in enumerate(labels):
        ys = [sub[(sub["label"] == lab) & (sub["level"] == lv)][metric].max() for lv in levels]
        xs = [i + (k - (n - 1) / 2) * (bw + gap) for i in range(len(levels))]
        ax.bar(xs, [0 if pd.isna(y) else y for y in ys], width=bw, color=color_for(lab, theme), label=NICE.get(lab, lab),
               linewidth=0)
        for x, y in zip(xs, ys):
            if not pd.isna(y):
                ax.text(x, y, fmt(y), ha="center", va="bottom", fontsize=10, color=theme["ink2"])
    ax.set_xticks(range(len(levels)), [f"{lv:g}" for lv in levels])
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 1.18)                 # headroom for the legend row
    ax.legend(loc="upper left", ncols=min(n, 4), fontsize=11)
    footer(fig, note, theme)
    fig.savefig(out)
    plt.close(fig)


def chart_lines(t, mode, metric, title, subtitle, ylabel, out, theme, note, logy=False, only=None):
    sub = t[t["mode"] == mode].dropna(subset=[metric])
    if only:
        sub = sub[sub["label"].isin(only)]
    if sub.empty:
        return
    labels = pick(sub, only)
    fig, ax = new_fig(title, subtitle, MODE_X[mode], ylabel, theme)
    ends = []
    for lab in labels:
        s = sub[sub["label"] == lab].sort_values("level")
        c = color_for(lab, theme)
        ax.plot(s["level"], s[metric], color=c, linewidth=2, marker="o", markersize=8,
                markeredgecolor=theme["surface"], markeredgewidth=2, label=NICE.get(lab, lab), solid_capstyle="round")
        ends.append((lab, s["level"].iloc[-1], s[metric].iloc[-1]))
    ax.set_xticks(sorted(sub["level"].unique()), [f"{v:g}" for v in sorted(sub["level"].unique())])
    ax.set_xlim(right=max(sub["level"]) * 1.35)
    if logy:
        ax.set_yscale("log")
        plain_log_axis(ax)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 3)          # headroom so the legend never sits on a line
    else:
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 1.22)
    # Direct end labels with collision avoidance in pixel space (min 18 px apart); legend stays regardless.
    fig.canvas.draw()
    pts = sorted(((ax.transData.transform((x, y))[1], lab, x, y) for lab, x, y in ends), key=lambda p: p[0])
    placed = []
    for py, lab, x, y in pts:
        if placed and py - placed[-1] < 24:
            py = placed[-1] + 24
        placed.append(py)
        ax.annotate(f"  {NICE.get(lab, lab)}  {fmt(y)}", (x, y), xytext=(0, py - ax.transData.transform((x, y))[1]),
                    textcoords="offset pixels", fontsize=10.5, color=theme["ink2"], va="center")
    ax.legend(loc="upper left", ncols=min(len(labels), 4), fontsize=11)
    footer(fig, note, theme)
    fig.savefig(out)
    plt.close(fig)


def chart_single_user(t, out, theme, note, only=None):
    sub = t[(t["mode"] == "batch") & (t["level"] == 1)].dropna(subset=["per_req_tok_s"])
    if only:
        sub = sub[sub["label"].isin(only)]
    if sub.empty:
        return
    sub = sub.sort_values("per_req_tok_s")
    n = len(sub)
    h = 2.6 + 0.55 * n
    fig, ax = new_fig("Single user decode speed", "conc = 1, tokens / s for one request, median of runs",
                      "", "", theme, height=h, margins=dict(left=0.18, right=0.97, top=1 - 1.25 / h, bottom=1.0 / h))
    ax.grid(axis="x", color=theme["grid"]); ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(True)
    ys = list(range(n))
    ax.set_ylim(-0.6, n - 0.4)
    bh = 24 / px_per_unit(ax, "y")
    ax.barh(ys, sub["per_req_tok_s"], height=bh, color=[color_for(l, theme) for l in sub["label"]], linewidth=0)
    ax.set_yticks(ys, [NICE.get(l, l) for l in sub["label"]])
    ax.tick_params(axis="y", colors=theme["ink"], labelsize=12)
    for y, v, tt in zip(ys, sub["per_req_tok_s"], sub["ttft_med_ms"]):
        ax.text(v, y, f"  {fmt(v)} tok/s   (TTFT {tt:.0f} ms)", va="center", fontsize=11, color=theme["ink2"])
    ax.set_xlim(0, sub["per_req_tok_s"].max() * 1.45)
    ax.set_xlabel("tokens / s", labelpad=6)
    footer(fig, note, theme)
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def env_block(meta: dict) -> str:
    if not meta:
        return "_no meta.json captured; run server/capture-meta.sh_"
    lines = ["| config | engine | version | model | GPU | driver | image / build | notes |", "|---|---|---|---|---|---|---|---|"]
    for label, m in meta.items():
        lines.append(f"| {label} | {m.get('engine','')} | {m.get('engine_version','')} | {m.get('model','')} | "
                     f"{m.get('gpu','')} | {m.get('driver','')} | {m.get('image','')} | {m.get('notes','')} |")
    startup = [(l, m.get("vllm_startup_lines")) for l, m in meta.items() if m.get("vllm_startup_lines")]
    if startup:
        lines.append("\nvLLM start-up lines (KV budget, straight into the post):\n")
        for l, s in startup:
            lines.append(f"**{l}**\n```\n{s.strip()}\n```")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "results"))
    ap.add_argument("--dark", action="store_true", help="also render dark-mode PNGs")
    a = ap.parse_args()
    results = Path(a.results)
    rep = results / "report"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    (rep / "charts").mkdir(parents=True, exist_ok=True)

    df, meta = load(results)
    df.to_csv(rep / "all-runs.csv", index=False)
    t = aggregate(df)
    core_df = df[df["label"].isin(SETS["core"])] if df["label"].isin(SETS["core"]).any() else df
    batch = core_df[core_df["mode"] == "batch"] if (core_df["mode"] == "batch").any() else core_df
    runs = int(batch.groupby(["mode", "label", "level"]).size().median())
    model = ", ".join(sorted(df["model"].unique()))
    versions = ", ".join(sorted({f"{e} {v}" for e, v in zip(df["engine"], df["engine_version"])}))
    gpu = next((m.get("gpu") for m in meta.values() if m.get("gpu")), "GPU")
    users_s = int(df.loc[df["mode"] == "users", "duration_s"].dropna().max() or 0)
    note = (f"{gpu} | {model} | {versions} | max_tokens={int(batch['max_tokens'].iloc[0])} | "
            f"batch: median of {runs} runs; users: one {users_s} s window")

    tables = {m: md_table(t, m) for m in ("batch", "users", "rate")}
    for m, s in tables.items():
        if s:
            (rep / "tables" / f"{m}.md").write_text(s + "\n", encoding="utf-8")

    themes = [("light", LIGHT)] + ([("dark", DARK)] if a.dark else [])
    sets = {k: v for k, v in SETS.items() if t["label"].isin(v).sum() > 0} or {"all": None}
    for name, th in themes:
        style(th)
        c = rep / "charts"
        sfx = "" if name == "light" else "-dark"
        for sname, only in sets.items():
            p = f"{sname}-"
            chart_bars(t, "batch", "agg_tok_s", "Aggregate throughput under concurrency",
                       "Sum of output tokens / s across all concurrent requests, same prompts, same model",
                       "tokens / s", c / f"{p}batch-throughput{sfx}.png", th, note, only=only)
            chart_lines(t, "batch", "ttft_p95_ms", "Time to first token, p95",
                        "How long the slowest 5% waited for the first word, by concurrency", "ms",
                        c / f"{p}batch-ttft-p95{sfx}.png", th, note, logy=True, only=only)
            chart_lines(t, "batch", "per_req_tok_s", "Decode speed per request",
                        "Tokens / s that ONE user sees while N requests share the GPU", "tokens / s",
                        c / f"{p}batch-per-request{sfx}.png", th, note, only=only)
            chart_lines(t, "batch", "lat_p95_s", "Request latency, p95",
                        "Worst-case user wait for a full answer, by concurrency", "seconds",
                        c / f"{p}batch-latency-p95{sfx}.png", th, note, logy=True, only=only)
            chart_single_user(t, c / f"{p}single-user{sfx}.png", th, note, only=only)
            chart_lines(t, "users", "agg_tok_s", "Throughput with real users",
                        "Closed loop: each user sends, reads, thinks ~3 s, sends again", "tokens / s",
                        c / f"{p}users-throughput{sfx}.png", th, note, only=only)
            chart_lines(t, "users", "ttft_p95_ms", "Time to first token with real users, p95",
                        "Closed loop with think time; this is what a person with headphones on feels", "ms",
                        c / f"{p}users-ttft-p95{sfx}.png", th, note, logy=True, only=only)
            chart_lines(t, "rate", "lat_p95_s", "The queueing cliff",
                        "Open loop Poisson arrivals: p95 latency vs offered load", "seconds",
                        c / f"{p}rate-latency-p95{sfx}.png", th, note, logy=True, only=only)
            chart_lines(t, "rate", "req_s", "Served rate vs offered rate",
                        "Where the served line flattens is the real capacity of this configuration", "requests / s",
                        c / f"{p}rate-served{sfx}.png", th, note, only=only)
            chart_bars(t, "batch", "vram_peak_gb", "VRAM at peak load",
                       "Peak memory.used during each run (nvidia-smi, 1 s samples)", "GB",
                       c / f"{p}batch-vram{sfx}.png", th, note, only=only)
        if (t["label"] == "vllm-peak").any():
            pk = df[df["label"] == "vllm-peak"]
            peak_note = (f"{gpu} | {', '.join(sorted(pk['model'].unique()))} | "
                         f"{', '.join(sorted({f'{e} {v}' for e, v in zip(pk['engine'], pk['engine_version'])}))} | "
                         f"max_tokens={int(pk['max_tokens'].iloc[0])} | one run per level")
            chart_bars(t, "batch", "agg_tok_s", "Where vLLM tops out",
                       "Aggregate tokens / s at 32 / 64 / 96 concurrent requests, 512-token answers (max-num-seqs 64)",
                       "tokens / s", c / f"peak-throughput{sfx}.png", th, peak_note, only=["vllm-peak"])
            chart_bars(t, "batch", "ttft_p95_ms", "TTFT p95 at peak concurrency",
                       "max-num-seqs 64: the jump at 96 is most likely requests waiting for a free sequence slot", "ms",
                       c / f"peak-ttft-p95{sfx}.png", th, peak_note, only=["vllm-peak"])

    md = ["# Ollama vs vLLM: benchmark report", "", f"_Generated from `{results}`. {note}_", "",
          "## Environment", "", env_block(meta), ""]
    for m, title in (("batch", "Concurrency sweep (batch)"), ("users", "Virtual users (closed loop)"),
                     ("rate", "Open loop arrival rate")):
        if tables[m]:
            md += [f"## {title}", "", tables[m], ""]
    md += ["## Charts", ""] + [f"![{p.stem}](charts/{p.name})" for p in sorted((rep / "charts").glob("*.png"))
                                if "-dark" not in p.name]
    (rep / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {rep / 'REPORT.md'}")
    for m, s in tables.items():
        if s:
            print(f"\n### {m}\n{s}")


if __name__ == "__main__":
    main()

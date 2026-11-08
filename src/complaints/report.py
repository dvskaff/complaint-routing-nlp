"""The dashboard (site/index.html, with the in-browser demo) and results.json."""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from .report_base import (
    BLUE,
    CSS,
    GRID,
    LAYOUT_JS,
    MUTED,
    SERIES,
    _card,
    _div,
    _int,
    _pct,
    _style,
    _table,
)

SHORT = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Money transfer, virtual currency, or money service": "Money transfer & virtual currency",
    "Payday loan, title loan, personal loan, or advance loan": "Payday, title & personal loans",
    "Payday loan, title loan, or personal loan": "Payday, title & personal loans",
    "Checking or savings account": "Checking or savings",
    "Vehicle loan or lease": "Vehicle loan or lease",
    "Debt or credit management": "Debt or credit management",
}


def short(name: str) -> str:
    return SHORT.get(name, name.split(",")[0])


def _pp(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:+.1f} pts"


def _p1(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.1%}"


# --------------------------------------------------------------------------- charts
def chart_coverage(curves: dict, target: float, chosen: dict) -> str:
    fig = go.Figure()
    names = [n for n in curves if n != "Majority class"]
    for i, n in enumerate(names):
        c = curves[n]
        fig.add_trace(go.Scatter(x=c["auto_share"], y=c["accuracy"], name=n, mode="lines",
                                 line=dict(color=SERIES[i % len(SERIES)], width=2),
                                 hovertemplate="%{y:.1%} accurate when auto-routing the most confident %{x:.0%}"
                                               "<extra>" + n + "</extra>"))  # fmt: skip
    fig.add_hline(y=target, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text=f"target {target:.0%}", annotation_position="bottom left")  # fmt: skip
    r = chosen
    if r["auto_share"] > 0:
        fig.add_trace(go.Scatter(x=[r["auto_share"]], y=[r["auto_accuracy"]], mode="markers", showlegend=False,
                                 marker=dict(size=10, color=SERIES[len(names) - 1], line=dict(color="#fff", width=2)),
                                 hovertemplate="Chosen threshold: auto-route %{x:.0%} at %{y:.1%}<extra></extra>"))  # fmt: skip
    _style(fig, "Accuracy vs. share of complaints auto-routed (test months)", 400,
           "accuracy of auto-routed complaints", "share of complaints auto-routed (most confident first)",
           hovermode="closest")  # fmt: skip
    fig.update_xaxes(tickformat=".0%", range=[0, 1.02])
    fig.update_yaxes(tickformat=".0%")
    return _div(fig)


def chart_f1(per_class: pd.DataFrame) -> str:
    d = per_class.assign(name=per_class["product"].map(short)).sort_values("f1")
    fig = go.Figure(go.Bar(x=d["f1"], y=d["name"], orientation="h", marker=dict(color=BLUE, cornerradius=4),
                           text=[f"{v:.2f}" for v in d["f1"]], textposition="outside",
                           customdata=np.stack([d["precision"], d["recall"], d["support"]], axis=1),
                           hovertemplate="%{y}<br>F1 %{x:.2f} · precision %{customdata[0]:.2f} · recall "
                                         "%{customdata[1]:.2f}<br>%{customdata[2]:,} test complaints<extra></extra>"))  # fmt: skip
    _style(fig, "F1 by product (fine-tuned model, test months)", 70 + 34 * len(d), x_title="F1",
           margin=dict(l=210, r=40, t=60, b=40), showlegend=False, hovermode="closest")  # fmt: skip
    fig.update_xaxes(range=[0, 1.1], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _div(fig)


def chart_confusion(cm: pd.DataFrame) -> str:
    share = cm.div(cm.sum(1).replace(0, np.nan), axis=0).fillna(0)
    names = [short(n) for n in cm.index]
    fig = go.Figure(go.Heatmap(z=share.to_numpy(), x=names, y=names, colorscale=[[0, "#f4f8fd"], [1, "#104281"]],
                               zmin=0, zmax=1, text=[[f"{v:.0%}" if v >= 0.02 else "" for v in r] for r in share.to_numpy()],
                               texttemplate="%{text}", textfont=dict(size=10),
                               customdata=cm.to_numpy(), colorbar=dict(title="share", tickformat=".0%"),
                               hovertemplate="Actual %{y} → routed to %{x}<br>%{z:.1%} (%{customdata:,} complaints)"
                                             "<extra></extra>"))  # fmt: skip
    _style(fig, "Where each product's complaints were routed (rows sum to 100%)", 120 + 40 * len(names),
           "actual product", "routed to", margin=dict(l=210, r=20, t=60, b=150), hovermode="closest")  # fmt: skip
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-35)
    return _div(fig)


def chart_topics(summary: dict) -> str:
    fig = go.Figure()
    months = summary["months"]
    for i, t in enumerate(summary["emerging"][:4]):
        prod = html.escape(short(t["top_product"]))
        fig.add_trace(go.Scatter(x=months, y=t["monthly_share"], name=f"{t['label']} ({prod})", mode="lines+markers",
                                 line=dict(color=SERIES[i % len(SERIES)], width=2), marker=dict(size=8),
                                 connectgaps=False,
                                 hovertemplate="%{x}: %{y:.1%} of " + prod + " complaints<extra>" +
                                               html.escape(t["label"]) + "</extra>"))  # fmt: skip
    if summary["recent_months"]:
        fig.add_vrect(x0=summary["recent_months"][0], x1=summary["recent_months"][-1], fillcolor="#eef3fa",
                      opacity=0.6, line_width=0, layer="below", annotation_text="recent",
                      annotation_position="top left")  # fmt: skip
    _style(
        fig,
        "Emerging topics: share of their product's complaints, by month",
        430,
        "share of the product's complaints",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=64, r=20, t=50, b=110),
    )
    fig.update_yaxes(tickformat=".1%", rangemode="tozero")
    return _div(fig)


def chart_volume(by_month: dict, w: dict) -> str:
    months = list(by_month)
    role = {m: "Training" for m in w["train"]}
    role[w["val"]] = "Validation"
    for m in w["test"]:
        role[m] = "Test"
    colors = {"Training": BLUE, "Validation": SERIES[2], "Test": SERIES[1]}
    fig = go.Figure()
    for r, c in colors.items():
        ms = [m for m in months if role.get(m) == r]
        fig.add_trace(go.Bar(x=ms, y=[by_month[m] for m in ms], name=r, marker=dict(color=c, cornerradius=4),
                             hovertemplate="%{x}: %{y:,} published narratives<extra>" + r + "</extra>"))  # fmt: skip
    _style(fig, "Published complaint narratives per month in the archive (window used here)", 340,
           "narratives", hovermode="closest", bargap=0.25)  # fmt: skip
    return _div(fig)


def chart_mix(mix: dict) -> str:
    fig = go.Figure()
    months = mix["months"]
    for i, (p, shares) in enumerate(mix["shares"].items()):
        fig.add_trace(go.Scatter(x=months, y=shares, name=short(p), mode="lines+markers",
                                 line=dict(color=SERIES[i % len(SERIES)], width=2), marker=dict(size=7),
                                 hovertemplate="%{x}: %{y:.0%} of narratives<extra>" + html.escape(short(p)) +
                                               "</extra>"))  # fmt: skip
    _style(fig, "Product mix of the archive by month (largest products, natural sample)", 380,
           "share of narratives", hovermode="x unified")  # fmt: skip
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return _div(fig)


def mix_sentence(mix: dict) -> str:
    """Name the biggest shift in the product mix between the start and the end of the window, if there is one."""
    if not mix:
        return ""
    best = None
    for p, shares in mix["shares"].items():
        a, b = float(np.mean(shares[:3])), float(np.mean(shares[-3:]))
        if best is None or abs(b - a) > abs(best[2] - best[1]):
            best = (p, a, b)
    p, a, b = best
    if abs(b - a) < 0.15:
        return ""
    return (
        f"The mix of published narratives shifts sharply over the window: {short(p).lower()} goes from {a:.0%} of "
        f"narratives in the first three months to {b:.0%} in the last three. Training is balanced by product, and "
        f"calibration uses the validation month's mix, so the model is tuned to the recent mix."
    )


# --------------------------------------------------------------------------- demo
DEMO_HTML = """
<section id="demo"><h2>Try it: route a complaint in your browser</h2>
<p>The fine-tuned model runs on your device with ONNX Runtime WebAssembly. Nothing you type leaves your browser.
The first run downloads the model (%MODEL_MB% MB) and the runtime (about 26 MB); after that it is cached.</p>
<div class="demo">
  <textarea id="q" rows="5" placeholder="Describe a problem with a bank, lender, card, collector or credit report…"></textarea>
  <div class="ex" id="examples"></div>
  <button id="go" class="go">Route this complaint</button> <span id="status" class="muted"></span>
  <div id="result"></div>
</div></section>
"""

DEMO_JS = """
const SHORT = %SHORT%;
const short = (n) => SHORT[n] || n.split(",")[0];
const $ = (id) => document.getElementById(id);
let demo = null, classifier = null;
const demoReady = fetch("demo.json").then(r => r.json()).then(d => {
  demo = d;
  d.examples.forEach((e, i) => {
    const b = document.createElement("button");
    b.textContent = "Example " + (i + 1); b.className = "chipbtn";
    b.onclick = () => { $("q").value = e.text; route(); };
    $("examples").appendChild(b);
  });
});
async function getClassifier() {
  if (classifier) return classifier;
  if (!%RUNTIME%) throw new Error("The browser runtime is not bundled in this build.");
  $("status").textContent = "Loading the model (first time only)…";
  const { pipeline, env } = await import("./vendor/transformers.min.js");
  env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = "./models/";
  env.backends.onnx.wasm.wasmPaths = new URL("./vendor/", location.href).href;
  env.backends.onnx.wasm.numThreads = 1;
  classifier = await pipeline("text-classification", demo.model, { dtype: "q8", device: "wasm" });
  return classifier;
}
async function route() {
  await demoReady;
  const text = $("q").value.trim();
  if (text.length < 15) { $("status").textContent = "Write a sentence or two first."; return; }
  $("go").disabled = true;
  try {
    const clf = await getClassifier();
    $("status").textContent = "Routing…";
    const t0 = performance.now();
    const out = (await clf(text, { top_k: null })).sort((a, b) => b.score - a.score);
    const ms = performance.now() - t0;
    const top = out[0], auto = top.score >= demo.threshold;
    const badge = auto
      ? `<span class="chip good">Auto-route</span> to <b>${short(top.label)}</b>`
      : `<span class="chip warn">Human review</span> suggested teams: <b>${short(out[0].label)}</b> or <b>${short(out[1].label)}</b>`;
    const bars = out.slice(0, 3).map(r => `<div class="bar"><span>${short(r.label)}</span>
      <div class="track"><div class="fill" style="width:${(r.score * 100).toFixed(1)}%"></div></div>
      <b>${(r.score * 100).toFixed(1)}%</b></div>`).join("");
    $("result").innerHTML = `<p>${badge}</p>${bars}<p class="muted">Auto-route needs at least
      ${(demo.threshold * 100).toFixed(1)}% confidence (the level that was ${(demo.target_accuracy * 100).toFixed(0)}%
      accurate on the validation month) · ${ms.toFixed(0)} ms on this device</p>`;
    $("status").textContent = "";
  } catch (e) {
    $("status").textContent = "Could not run the model here: " + e.message;
  } finally { $("go").disabled = false; }
}
$("go").onclick = route;
"""

DEMO_CSS = """
.demo textarea { width:100%; font:inherit; font-size:14.5px; padding:10px 12px; border:1px solid #c3c2b7; border-radius:10px;
                 background:#fff; resize:vertical; }
.ex { margin:8px 0 4px; } .chipbtn { border:1px solid #c3c2b7; background:#fff; border-radius:999px; padding:3px 11px;
       margin:0 6px 6px 0; font-size:13px; cursor:pointer; color:#104281; }
.go { background:#104281; color:#fff; border:0; border-radius:9px; padding:8px 16px; font-weight:600; font-size:14px;
      cursor:pointer; margin:6px 10px 10px 0; } .go:disabled { opacity:.6; }
#result { margin:6px 0 14px; } .bar { display:grid; grid-template-columns:minmax(120px,240px) 1fr 56px; gap:10px;
      align-items:center; margin:6px 0; font-size:14px; }
.track { background:#eef1f5; border-radius:6px; height:14px; overflow:hidden; }
.fill { background:#2a78d6; height:100%; border-radius:6px; }
.quote { display:block; text-align:left; font-size:13px; color:#52514e; white-space:normal; min-width:260px; }
td.l b, td.l { min-width:150px; }
"""


# --------------------------------------------------------------------------- page
def build_site(cfg, run: dict, scores: dict, curves: dict, per_class: pd.DataFrame, cm: pd.DataFrame,
               errors: list[dict], dups: dict, topics: dict, examples: list[str]) -> None:  # fmt: skip
    from .pipeline import ENS_NAME, FT_NAME

    site = cfg.site_dir
    site.mkdir(parents=True, exist_ok=True)
    ft, tf = scores[FT_NAME], scores["TF-IDF + logistic regression"]
    maj = scores["Majority class"]
    r = ft["routing"]
    w = run["window"]
    em = topics["emerging"][0] if topics["emerging"] else None
    synthetic = "<div class='synthetic'>SYNTHETIC TEST DATA: not real complaints</div>" if cfg.fixture else ""
    cards = "".join([
        _card("Auto-routed", _pct(r["auto_share"]), f"of test complaints, {_p1(r['auto_accuracy'])} of them routed "
              f"correctly (target {cfg.target_accuracy:.0%}); the rest go to a person"),
        _card("Macro-F1", f"{ft['macro_f1']:.2f}", f"fine-tuned transformer vs. {tf['macro_f1']:.2f} for TF-IDF and "
              f"{scores[ENS_NAME]['macro_f1']:.2f} for both combined, across {len(run['labels'])} products"),
        _card("Accuracy", _p1(ft["accuracy"]), f"all test complaints · always-majority: {_p1(maj['accuracy'])}"),
        _card("Top-2 accuracy", _p1(ft["top2_accuracy"]), "right team among the two suggestions"),
        _card("Browser model", f"{run['onnx']['size_mb']:.0f} MB" if run["onnx"]["size_mb"] >= 10
              else f"{run['onnx']['size_mb']:.1f} MB", f"8-bit ONNX, same routing decision as the full model "
              f"on {run['onnx']['decision_agreement']:.1%} of complaints"),
        _card("Top emerging issue", html.escape(" · ".join(em["words"][:2])) if em else "–",
              f"{_pp(em['change_pts'])} share in the last {len(topics['recent_months'])} months" if em else ""),
    ])  # fmt: skip
    order = [
        "Majority class",
        "TF-IDF + logistic regression",
        "Sentence embeddings + logistic regression",
        FT_NAME,
        ENS_NAME,
    ]
    comp_rows = []
    for n in order:
        s = scores[n]
        rr = s["routing"]
        comp_rows.append([
            f"<b>{n}</b>" if n == FT_NAME else n, _p1(s["accuracy"]), f"{s['macro_f1']:.3f}", f"{s['weighted_f1']:.3f}",
            _p1(s["top2_accuracy"]), f"{s['ece_uncalibrated']:.3f} → {s['ece']:.3f}" if n != "Majority class" else "–",
            _pct(rr["auto_share"]), _p1(rr["auto_accuracy"]),
            "–" if not s["seconds_train"] else f"{s['seconds_train'] / 60:.1f} min",
            "–" if pd.isna(s["ms_per_complaint"]) else f"{s['ms_per_complaint']:.1f}",
        ])  # fmt: skip
    comp = _table(["Model", "Accuracy", "Macro-F1", "Weighted F1", "Top-2", "Calibration error",
                   f"Auto-routed ({cfg.target_accuracy:.0%} target)", "Accuracy auto-routed", "Train time",
                   "ms / complaint"], comp_rows)  # fmt: skip
    conf_rows = [[short(c["actual"]), short(c["predicted"]), _int(c["n"]), _pct(c["share_of_actual"])]
                 for c in evaluate_top(cm)]  # fmt: skip
    err_rows = [[f"<span class='quote'>{html.escape(e['text'])}</span>", short(e["actual"]), short(e["predicted"]),
                 _pct(e["confidence"])] for e in errors]  # fmt: skip
    emerging_rows = [[html.escape(t["label"]), _p1(t["base_share"]), _p1(t["recent_share"]), _pp(t["change_pts"]),
                      f"{t['ratio']:.1f}×", short(t["top_product"]),
                      f"<span class='quote'>{html.escape(t['example'])}</span>"] for t in topics["emerging"]]  # fmt: skip
    fading_rows = [[html.escape(t["label"]), _p1(t["base_share"]), _p1(t["recent_share"]), _pp(t["change_pts"]),
                    short(t["top_product"])] for t in topics["fading"]]  # fmt: skip
    largest_rows = [[html.escape(t["label"]), _int(t["docs"]), short(t["top_product"]), _pct(t["top_product_share"])]
                    for t in topics["largest"]]  # fmt: skip
    hist = run["finetune"]["history"]
    hist_txt = "; ".join(f"epoch {h['epoch']}: validation macro-F1 {h['val_macro_f1']:.3f}" for h in hist)
    unseen = dups.get("unseen", {})
    mix_rows = [[short(p), _int(run["train_by_product"].get(p, 0)), _int(n), _pct(n / max(run["test"], 1))]
                for p, n in sorted(run["test_by_product"].items(), key=lambda kv: -kv[1])]  # fmt: skip
    sparse = run.get("sparse_months") or {}
    sparse_txt = (f"The newest archive months hold almost nothing ({', '.join(f'{m}: {n:,}' for m, n in sparse.items())} "
                  f"narratives), so the window ends in {w['test'][-1]}." if sparse else "")  # fmt: skip
    mix_txt = mix_sentence(run.get("mix_by_month") or {})
    not_routed = ", ".join(f"{short(k)} ({v:,})" for k, v in run["not_routed"].items()) or "none"
    demo = DEMO_HTML.replace("%MODEL_MB%", f"{run['onnx']['size_mb']:.0f}")
    js = DEMO_JS.replace("%SHORT%", json.dumps(SHORT)).replace(
        "%RUNTIME%", "true" if run["runtime_bundled"] else "false"
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Consumer Complaint Routing</title>
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
<style>{CSS}{DEMO_CSS}</style></head><body>
{synthetic}
<header><div class="wrap"><h1>Consumer Complaint Routing with Transformers</h1>
<p>Can a model read a consumer's complaint and send it to the right product team, handling the easy cases on its own
and passing the uncertain ones to a person? And which issues are growing? Built on {
        run["test"] + run["val"] + run["train"]:,}
{"synthetic test" if cfg.fixture else "real"} complaint narratives from the CFPB's archive (trained on {
        w["train"][0]
    } to {w["train"][-1]}, tested on
{w["test"][0]} to {w["test"][-1]}).</p>
<a class="btn" href="#demo">Try it in your browser</a><a class="btn" href="{cfg.repo_url}">Source code</a>
<a class="btn" href="{cfg.repo_url}/blob/main/MODEL_CARD.md">Model card</a></div></header>
<main>
<div class="cards">{cards}</div>
{demo}
<section><h2>Models, from simple to transformer</h2>
<p>All models learn from the same {run["train"]:,} training complaints (class-balanced: at most
{cfg.train_cap_per_product:,} per product) and are scored on {
        run["test"]:,} complaints from the two newest months, in
their real product mix. Probabilities are calibrated on the validation month ({
        w["val"]
    }): a prior correction for the
real mix plus temperature scaling. Macro-F1 weighs every product equally, so it rewards getting the small products
right, not only the largest. The ensemble averages the calibrated probabilities of the fine-tuned model and TF-IDF; the
browser demo runs the fine-tuned model alone.</p>
{comp}
<p class="muted">Calibration error is the expected calibration error before → after calibration (0 = stated confidence
matches accuracy). Fine-tuning: {run["base_model"]}, {run["finetune"]["train_size"]:,} complaints, {
        hist_txt
    }. Times are
on a 4-core GitHub Actions runner (CPU only).</p></section>

<section><h2>Auto-route vs. human review</h2>
<p>The routing policy auto-routes a complaint only when the model's calibrated confidence is at least
{r["threshold"]:.1%}: on the validation month, complaints above that level were {
        cfg.target_accuracy:.0%} accurate. On the
test months that sends <b>{_pct(r["auto_share"])}</b> of complaints straight to a team, {
        _p1(r["auto_accuracy"])
    } of them
correctly; the other {
        _pct(r["review_share"])
    } go to a person, with the right team among the model's top two suggestions
{_p1(r["review_top2"])} of the time.</p>
{chart_coverage(curves, cfg.target_accuracy, r)}</section>

<section><h2>Where it gets confused</h2>
<p>Most errors are between products that genuinely overlap: a debt in collection is also on the credit report, and a
scam payment involves both a bank account and a money-transfer app. Consumers pick the product themselves when they
file, so some "errors" are arguably labeling choices.</p>
{chart_f1(per_class)}
{chart_confusion(cm)}
<h3>Most common mix-ups</h3>{
        _table(["Actual product", "Routed to", "Complaints", "Share of that product"], conf_rows, 2)
    }
<h3>Most confident mistakes</h3>
<p class="small">The test complaints the model got wrong with the highest confidence (redactions removed, cut to 260
characters). Several read as if the consumer could have picked either product.</p>
{_table(["Narrative", "Filed as", "Routed to", "Confidence"], err_rows, 1)}
<p class="small">Template letters: {
        _pct(dups["share_in_train"])
    } of test narratives also appear word for word in the
training set (credit-repair templates are common). On the test complaints that do not, macro-F1 is
{unseen.get("macro_f1", float("nan")):.3f} and accuracy {_p1(unseen.get("accuracy"))}.</p></section>

<section><h2>Emerging issues</h2>
<p>BERTopic grouped {topics["docs"]:,} complaints from {topics["months"][0]} to {
        topics["months"][-1]
    } (a natural sample of
every month) into {topics["topics"]} topics by meaning, using the same sentence embeddings; {
        _pct(topics["outlier_share"])
    }
did not fit any topic. A topic is <b>emerging</b> when its share of complaints in the last {
        len(topics["recent_months"])
    }
months is well above its share before. Shares are measured within each topic's main product (for example, the share
of checking-account complaints about a scam), so the archive's changing product mix doesn't create false trends.</p>
{chart_topics(topics) if topics["emerging"] else ""}
{
        _table(
            [
                "Topic (top words)",
                "Share of its product before",
                "Recently",
                "Change",
                "Ratio",
                "Main product",
                "Typical complaint",
            ],
            emerging_rows,
            1,
        )
        if emerging_rows
        else "<p class='muted'>No topic grew enough to flag.</p>"
    }
<h3>Fading</h3>{
        _table(
            ["Topic (top words)", "Share of its product before", "Recently", "Change", "Main product"],
            fading_rows,
            1,
        )
        if fading_rows
        else "<p class='muted'>None.</p>"
    }
<details><summary>Largest topics overall</summary>
{
        _table(["Topic (top words)", "Complaints", "Main product", "Share of topic"], largest_rows, 1)
    }</details></section>

<section><h2>Data</h2>
<p>In August 2026 the CFPB stopped publishing complaint narratives and moved every narrative it had published into a
downloadable archive. This window holds {
        sum(run["narratives_by_month"].values()):,} of them. The pipeline reads the
archive files directly ({", ".join(run["files"][-3:])} and earlier) and keeps a fixed, hash-based sample.
{sparse_txt} {mix_txt}</p>
{chart_volume(run["narratives_by_month"], w)}
{chart_mix(run["mix_by_month"]) if run.get("mix_by_month") else ""}
<div class="grid2"><div>{_table(["Product", "Training", "Test", "Test share"], mix_rows, 1)}</div>
<div><p class="small">Training is capped per product so small products are learned too; testing keeps the real mix.
Products with fewer than {cfg.min_train_per_product} training complaints are not routed and go to human review
(test complaints affected: {not_routed}).</p></div></div></section>

<section><h2>How it works</h2><ul class="method">
<li><b>Data:</b> CFPB Consumer Complaint Database narrative archive (public, consumer-consented narratives with personal
details redacted by the CFPB). Time-based split: training {w["train"][0]} to {w["train"][-1]}, validation {
        w["val"]
    },
test {w["test"][0]} to {w["test"][-1]}, so the model never sees the future.</li>
<li><b>Models:</b> TF-IDF (words and word pairs) + logistic regression; frozen {
        run["embedder"]
    } sentence embeddings +
logistic regression; and the same MiniLM encoder fine-tuned end to end with a classification head (PyTorch, AdamW,
linear warmup and decay, length-bucketed batches, best epoch by validation macro-F1).</li>
<li><b>Calibration and policy:</b> prior correction (its strength chosen on validation so small products are still
predicted) + temperature scaling, fitted on validation; the auto-route threshold is the lowest confidence whose
validation accuracy still meets the target.</li>
<li><b>Topics:</b> BERTopic (UMAP + HDBSCAN + class-based TF-IDF) on the sentence embeddings of a natural monthly sample.</li>
<li><b>Serving:</b> the fine-tuned model is exported to ONNX with calibration built in, quantized to 8-bit, checked against
PyTorch on {run["onnx"]["n"]:,} validation complaints (same routing decision on {
        run["onnx"]["decision_agreement"]:.1%},
same top team on {run["onnx"]["top1_agreement"]:.1%}; the differences are near-ties), and run in the browser
with transformers.js and ONNX Runtime WebAssembly. No server.</li>
<li><b>Automation:</b> GitHub Actions runs the pipeline on every change and checks the archive monthly; samples,
embeddings, the fine-tuned model and topics are cached between runs, and a quality gate blocks publishing if accuracy,
macro-F1 or browser parity falls short.</li>
<li><b>Limits:</b> labels are the products consumers chose; narratives are cut to the first {
        cfg.max_tokens
    } tokens for
the transformer; the archive ends in August 2026, so this is a fixed benchmark rather than a live feed.</li>
</ul></section>
</main>
<footer>Data: Consumer Financial Protection Bureau, Consumer Complaint Database narratives archive · Built {
        run["built_at"]
    } ·
<a href="{cfg.repo_url}">Source code</a></footer>
<script>{LAYOUT_JS}</script>
<script type="module">{js}</script>
</body></html>"""
    (site / "index.html").write_text(page, encoding="utf-8")
    (site / ".nojekyll").touch()
    summary = {"run": run, "scores": scores, "per_class": per_class.to_dict("records"),
               "confusion": cm.to_dict(), "duplicates": dups,
               "topics": {k: v for k, v in topics.items() if k != "largest"}, "synthetic": cfg.fixture,
               "built_at": run["built_at"]}  # fmt: skip
    (site / "results.json").write_text(json.dumps(_clean(summary), indent=2, allow_nan=False))


def evaluate_top(cm: pd.DataFrame) -> list[dict]:
    from .evaluate import top_confusions

    return top_confusions(cm)


def _clean(obj):
    """NaN -> null, so results.json is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return None if np.isnan(obj) else float(obj)
    return obj

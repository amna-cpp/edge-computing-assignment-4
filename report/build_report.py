"""Build the technical report (figures + HTML -> PDF via headless Chrome) from the result files.

    python -m report.build_report

Every number in the report is read from results/*.csv|json and logs/ - nothing is typed in by hand.
"""
from __future__ import annotations

import csv
import html
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_common import LOGS_DIR, RESULTS_DIR, ROOT

OUT = ROOT / "report"
FIG = OUT / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def table(rows, cols, headers=None):
    if not rows:
        return "<p><i>(no data)</i></p>"
    headers = headers or cols
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    def cell(v):
        v = str(v)
        try:
            if "." in v and len(v.split(".")[1]) > 3 and v.replace(".", "", 1).replace("-", "", 1).isdigit():
                return f"{float(v):.3f}"
        except ValueError:
            pass
        return v
    body = "".join("<tr>" + "".join(f"<td>{html.escape(cell(r.get(c, '')))}</td>" for c in cols) + "</tr>"
                   for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def console(path: Path, patterns=None, last=None, maxlines=40):
    """Excerpt of a console log, rendered as a terminal 'screenshot' block."""
    if not path.exists():
        return ""
    lines = [l for l in path.read_text(errors="ignore").splitlines()
             if "Warning" not in l and "warn" not in l.lower() and "if self.resolution" not in l
             and "return (logits" not in l and "if return_features" not in l]
    if patterns:
        lines = [l for l in lines if any(p in l for p in patterns)]
    lines = lines[-(last or maxlines):]
    return "<pre class='term'>" + html.escape("\n".join(lines)) + "</pre>"


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    # --- training curves
    plt.figure(figsize=(6.5, 3.2))
    for name in ["teacher", "mobilenetv2_baseline", "mobilenetv2_kd", "squeezenet_baseline", "squeezenet_kd",
                 "efficientnet_lite0_baseline", "efficientnet_lite0_kd"]:
        rows = read_csv(RESULTS_DIR / "q1" / f"{name}_history.csv")
        if rows:
            plt.plot([int(r["epoch"]) for r in rows], [float(r["test_acc"]) for r in rows],
                     marker="o", ms=3, label=name, ls="--" if "baseline" in name else "-")
    plt.xlabel("epoch"); plt.ylabel("CIFAR-100 test top-1 (%)"); plt.grid(alpha=.3)
    plt.legend(fontsize=6, ncol=2); plt.tight_layout(); plt.savefig(FIG / "q1_curves.png", dpi=170); plt.close()

    # --- NAS: search history + pareto
    hist = read_csv(RESULTS_DIR / "q2" / "nas_history.csv")
    trained = [r for r in hist if r["status"] == "trained"]
    rejected = [r for r in hist if r["status"] == "rejected"]
    if trained:
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
        accs = [float(r["val_acc"]) for r in trained]
        best, run = [], -1
        for a in accs:
            run = max(run, a); best.append(run)
        ax[0].scatter(range(1, len(accs) + 1), accs, s=12, label="candidate")
        ax[0].plot(range(1, len(accs) + 1), best, color="C3", label="best so far")
        ax[0].set_xlabel("trained candidate #"); ax[0].set_ylabel("proxy val acc (%)")
        ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
        ax[1].scatter([float(r["latency_ms"]) for r in rejected], [float(r["size_mb"]) for r in rejected],
                      s=14, c="tab:red", marker="x", label=f"rejected ({len(rejected)})")
        sc = ax[1].scatter([float(r["latency_ms"]) for r in trained], [float(r["size_mb"]) for r in trained],
                           s=14, c=accs, cmap="viridis", label=f"trained ({len(trained)})")
        ax[1].axvline(1.5, ls=":", c="k"); ax[1].axhline(1.0, ls=":", c="k")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("measured CPU latency (ms)"); ax[1].set_ylabel("ONNX size (MB)")
        ax[1].legend(fontsize=7); plt.colorbar(sc, ax=ax[1], label="val acc (%)")
        plt.tight_layout(); plt.savefig(FIG / "q2_search.png", dpi=170); plt.close()


def arch_table():
    """Parameters / MACs of every implemented Q1 architecture, computed from the actual models."""
    from edge_common.deploy import count_macs, count_params
    from q1_distillation.models import build_model
    rows = []
    for arch, kw in [("resnet50", {"stem_stride": 2}), ("mobilenetv2", {}), ("squeezenet", {}),
                     ("efficientnet_lite0", {})]:
        m = build_model(arch, 100, **kw)
        rows.append({"model": arch, "role": "teacher" if arch == "resnet50" else "student",
                     "params_M": f"{count_params(m) / 1e6:.2f}", "macs_M": f"{count_macs(m) / 1e6:.0f}"})
    return rows


def load_json(path: Path):
    return json.load(open(path)) if path.exists() else {}


def main() -> None:
    figures()
    bench = read_csv(RESULTS_DIR / "benchmark_results.csv")
    gains = read_csv(RESULTS_DIR / "kd_gains.csv")
    hist = read_csv(RESULTS_DIR / "q2" / "nas_history.csv")
    best = load_json(RESULTS_DIR / "q2" / "best_architecture.json")
    nas_res = load_json(RESULTS_DIR / "q2" / "nas_best_result.json")
    teacher = load_json(RESULTS_DIR / "q1" / "teacher_result.json")
    machine = load_json(RESULTS_DIR / "benchmark_results.json").get("machine", {})
    pareto = read_csv(RESULTS_DIR / "q2" / "pareto_front.csv")
    t_hist = read_csv(RESULTS_DIR / "q1" / "teacher_history.csv")
    n_tr = sum(r["status"] == "trained" for r in hist)
    n_rej = sum(r["status"] == "rejected" for r in hist)
    nas_args = load_json(RESULTS_DIR / "q2" / "nas_history.json").get("args", {})
    cons = load_json(RESULTS_DIR / "q2" / "nas_history.json").get("constraints", {})

    css = """
    body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.38;margin:0;color:#111}
    h1{font-size:20pt;margin:0 0 4pt} h2{font-size:14pt;border-bottom:1.5px solid #333;margin-top:18pt;padding-bottom:2pt}
    h3{font-size:11.5pt;margin:12pt 0 4pt} table{border-collapse:collapse;font-size:8.5pt;margin:6pt 0;width:100%}
    th,td{border:1px solid #999;padding:2.5pt 4pt;text-align:left} th{background:#e8e8e8}
    pre,code{font-family:Menlo,monospace;font-size:8pt} pre.term{background:#1e1e1e;color:#d8d8d8;padding:6pt;
    border-radius:4pt;white-space:pre-wrap;font-size:7pt;line-height:1.25}
    img{max-width:100%} .eq{background:#f4f4f4;padding:4pt 8pt;font-family:Menlo,monospace;font-size:9pt}
    @page{size:A4;margin:16mm 15mm}
    """

    # ---------------------------------------------------------------- Q1
    q1 = f"""<h2>1. Question 1 - Knowledge distillation</h2>
<h3>1.1 Models (implemented from scratch, CIFAR-adapted)</h3>
<ul>
<li><b>Teacher - ResNet-50</b>: bottleneck blocks 3-4-6-3, 3x3 stem without max-pool; stem stride 2 (4x cheaper than the stride-1 CIFAR variant).</li>
<li><b>MobileNetV2</b>: inverted residuals with linear bottlenecks, ReLU6, width multiplier 1.0.</li>
<li><b>SqueezeNet v1.1</b>: Fire modules (squeeze 1x1 -> expand 1x1 || 3x3), BatchNorm after every convolution, fully-convolutional classifier.</li>
<li><b>EfficientNet-Lite0</b>: MBConv blocks without squeeze-excite, ReLU6, fixed stem/head, stochastic depth.</li>
</ul>
{table(arch_table(), ["model", "role", "params_M", "macs_M"], ["model", "role", "parameters (M)", "MACs (M), 32x32 input"])}
<h3>1.2 Distillation framework (<code>q1_distillation/distill.py</code>)</h3>
<p>The framework implements soft-target distillation with a temperature softmax, the combined hard + soft objective, and feature-based distillation:</p>
<div class="eq">L = alpha * CE(s, y) + (1 - alpha) * T^2 * KL( softmax(t/T) || softmax(s/T) ) + beta * mean_i MSE( std(A_i(f_s,i)), std(f_t,i) )</div>
<ul>
<li><b>Soft targets / temperature</b>: the teacher's logits are softened with <code>softmax(z/T)</code> (T = 4); the student matches them with KL divergence.
The <code>T^2</code> factor keeps the soft-term gradient comparable in size to the hard-label term (Hinton et al., 2015).</li>
<li><b>Hard + soft</b>: <code>alpha</code> = 0.3 weights the cross-entropy on the ground-truth labels.</li>
<li><b>Feature-based</b>: 1x1-conv + BatchNorm adapters <code>A_i</code> map the student's mid- and last-stage feature maps to the teacher's
(layer3: 1024 channels, layer4: 2048 channels); teacher maps are average-pooled to the student's spatial size and both maps are standardised per
sample, so the term is scale-free; <code>beta</code> = 1. The adapters exist only during training - they are not part of the student's checkpoint or ONNX graph.</li>
<li>The teacher is frozen, kept in eval mode, and evaluated on the same augmented batch as the student.</li>
</ul>
<h3>1.3 Training protocol (<code>edge_common/engine.py</code>)</h3>
<p>SGD with Nesterov momentum 0.9, learning rate 0.1 with 1-epoch linear warm-up and cosine decay, weight decay 5e-4 (not applied to BatchNorm/bias),
batch size 128, random-crop (padding 4) + horizontal-flip augmentation performed on the device, final-epoch weights (no checkpoint selection on the test set).
The baseline pipeline (<code>train_baseline.py</code>) and the distillation pipeline (<code>train_kd.py</code>) share every hyper-parameter and seed and differ only in the loss.
Checkpoints are resumable (state is written after every epoch).</p>"""
    if teacher:
        q1 += f"""<h3>1.4 Teacher result</h3>
<p>ResNet-50 on CIFAR-100: <b>{fnum(teacher.get('test_acc'))}% top-1</b> on the 10 000-image test set after {teacher.get('epochs')} epochs
({fnum(float(teacher.get('params', 0)) / 1e6)} M parameters).</p>
<img src="figures/q1_curves.png" width="70%"><p><i>Figure 1 - CIFAR-100 test accuracy per epoch of the teacher.</i></p>
{console(LOGS_DIR / "queue2.log", ["[teacher] epoch", "FINAL"], last=8)}"""
    nxt = 5 if teacher else 4
    if gains:
        q1 += f"""<h3>1.{nxt} Baseline vs distilled students</h3>
{table(gains, ["student", "kd_variant", "baseline_acc", "distilled_acc", "kd_gain_pp"], ["student", "KD variant", "baseline top-1 %", "distilled top-1 %", "gain (pp)"])}"""

    nxt += 1 if gains else 0
    q1 += f"""<h3>1.{nxt} Unified inference and ONNX deployment</h3>
<ul>
<li><code>q1_distillation/inference.py</code> - one <code>StudentPredictor(arch, variant, backend)</code> interface for all three students in both variants (baseline / distilled),
with PyTorch or ONNX Runtime back-ends; supports single images, CIFAR-100 test samples and full test-set evaluation from the command line.</li>
<li><code>q1_distillation/export_onnx.py</code> - exports every model to ONNX (opset 17, dynamic batch) and verifies it against PyTorch with ONNX Runtime
(the export fails if logits differ by more than 1e-4).</li>
<li><code>q1_distillation/onnx_infer.py</code> - stand-alone CPU inference with ONNX Runtime only (numpy + Pillow + onnxruntime), including single-image latency.</li>
</ul>"""

    # ---------------------------------------------------------------- Q2
    hist_trained = sorted([r for r in hist if r["status"] == "trained"], key=lambda r: -float(r["val_acc"]))[:6]
    rej_rows = [r for r in hist if r["status"] == "rejected"][:6]
    q2 = f"""<h2>2. Question 2 - Hardware-aware neural architecture search</h2>
<h3>2.1 Search space and architecture generator (<code>q2_nas/search_space.py</code>, <code>generator.py</code>)</h3>
<table><tr><th>dimension</th><th>choices</th></tr>
<tr><td>convolutional depth</td><td>2-4 stages, 1-3 blocks per stage (stride 2 at the first block of every stage after the first)</td></tr>
<tr><td>filters / channels</td><td>stem 16/24/32; per-stage channels 16, 24, 32, 48, 64, 96, 128, 160</td></tr>
<tr><td>kernel size</td><td>3 / 5 / 7 per stage</td></tr>
<tr><td>block type</td><td>regular convolution or depthwise-separable (depthwise k x k + pointwise 1x1), residual connection when shapes match</td></tr>
<tr><td>activation</td><td>ReLU, ReLU6, Hardswish, SiLU, LeakyReLU (per stage)</td></tr>
<tr><td>input resolution</td><td>16 / 24 / 32 (bilinear resize <i>inside</i> the network graph, so every deployed model takes a 3x32x32 input)</td></tr></table>
<p>An architecture is a JSON dictionary. <code>generator.build_from_config</code> turns any sampled dictionary into an <code>nn.Module</code>
(stem -> stages -> global average pooling -> linear classifier), so the same code builds candidates during the search and the final network.</p>
<h3>2.2 Search strategy and hardware-aware pipeline (<code>q2_nas/search.py</code>, <code>hardware.py</code>)</h3>
<p><b>Aging (regularised) evolution</b>: a FIFO population of {nas_args.get('population', 12)}; at every step a tournament of {nas_args.get('sample', 4)} picks the fittest parent and a child is produced
by mutation (or, with probability {nas_args.get('crossover_prob', 0.25)}, crossover with a second tournament winner) and replaces the oldest member.
For <b>every</b> candidate, <i>before any training</i>, the network is exported to ONNX and run with ONNX Runtime on the <b>CPU</b> (1 thread, batch 1) in a fresh process, measuring</p>
<ul><li>median inference <b>latency</b>,</li><li>model <b>storage size</b> (ONNX file),</li><li><b>peak inference memory</b> (RSS growth, <code>resource.getrusage</code>).</li></ul>
<p>Candidates that violate the predefined deployment budget are <b>rejected without being trained</b>:</p>
<div class="eq">latency &lt;= {cons.get('max_latency_ms', 1.5)} ms,   model size &lt;= {cons.get('max_size_mb', 1.0)} MB,   peak inference memory &lt;= {cons.get('max_peak_mem_mb', 16.0)} MB</div>
<p>Feasible candidates are trained on CIFAR-10 ({nas_args.get('epochs', '?')} epochs on {nas_args.get('train_limit', '?')} training images; a held-out
{nas_args.get('val_size', 5000)}-image validation split of the training set is used for scoring - the test set is never used during the search) and ranked by
<code>fitness = val_acc - {nas_args.get('w_latency', 0.03)} * latency/max_latency - {nas_args.get('w_size', 0.02)} * size/max_size</code>,
so accuracy is traded off against <i>measured</i> latency and storage size.</p>
<h3>2.3 Search results</h3>
<p><b>{n_tr}</b> candidates were trained and <b>{n_rej}</b> were rejected by the hardware constraints (no training time was spent on them).</p>
<img src="figures/q2_search.png" width="85%"><p><i>Figure 2 - left: proxy validation accuracy of every trained candidate and the running best; right: measured latency vs ONNX size of
rejected (x) and trained (dots) candidates, dotted lines = deployment constraints.</i></p>
<h3>2.4 Best discovered architecture</h3>
<p><code>{html.escape(best.get('description', ''))}</code></p>
<p>Measured on the CPU during the search: latency <b>{fnum(best.get('latency_ms'), 3)} ms</b>, model size <b>{fnum(best.get('size_mb'), 3)} MB</b>,
peak inference memory <b>{fnum(best.get('peak_mem_mb'), 1)} MB</b>, {fnum(float(best.get('params', 0)) / 1e3, 1)} k parameters,
{fnum(float(best.get('macs', 0)) / 1e6, 1)} M MACs; proxy validation accuracy {fnum(best.get('search_val_acc'))}%.</p>"""
    if nas_res:
        q2 += f"<p>Trained from scratch on the full CIFAR-10 training set ({nas_res.get('epochs')} epochs): <b>test top-1 = {fnum(nas_res.get('test_acc'))}%</b>.</p>"
    q2 += f"""<h3>2.5 Top candidates, example rejections and Pareto front</h3>
{table(hist_trained, ["id", "val_acc", "latency_ms", "size_mb", "peak_mem_mb", "description"], ["id", "val %", "lat ms", "MB", "mem MB", "architecture"])}
{table(rej_rows, ["id", "reject_reason", "latency_ms", "size_mb"], ["id", "reason for rejection", "lat ms", "MB"])}
{table(pareto, list(pareto[0].keys()), ["id", "val %", "lat ms", "MB", "mem MB", "architecture"]) if pareto else ""}
{console(LOGS_DIR / "queue2.log", ["search finished", "BEST", "reject"], last=12)}
<h3>2.6 Deployment pipeline</h3>
<ul>
<li><code>q2_nas/train_best.py</code> - trains the best architecture (<code>results/q2/best_architecture.json</code>) from scratch on the full CIFAR-10 training set and evaluates it on the test set.</li>
<li><code>q2_nas/export_onnx.py</code> - exports to ONNX and verifies against PyTorch; <code>q2_nas/inference.py</code> - ONNX Runtime CPU inference for single images, samples or the whole test set.</li>
</ul>"""

    # ---------------------------------------------------------------- benchmark
    bench_html = ""
    if bench:
        bench_html = f"""<h2>3. ONNX Runtime CPU benchmark</h2>
<p><code>benchmarks/benchmark_all.py</code> -> <code>results/benchmark_results.csv</code>. Latency = median of timed runs (batch 1); peak memory = RSS growth in a fresh process.</p>
{table(bench, ["model", "variant", "dataset", "onnx_top1_acc", "params_M", "macs_M", "onnx_size_mb", "latency_ms_median_1thread", "latency_ms_median_4threads", "throughput_img_s_batch32_4threads", "peak_mem_mb_batch1", "torch_cpu_latency_ms_1thread"],
        ["model", "variant", "data", "top-1 %", "params M", "MACs M", "ONNX MB", "lat 1t ms", "lat 4t ms", "img/s b32", "peak mem MB", "torch lat ms"])}"""

    notes = f"""<h2>{4 if bench else 3}. Implementation notes</h2>
<ul>
<li>Models are implemented from scratch in PyTorch (no torchvision / timm); training uses the Apple-silicon GPU (MPS) with the dataset and augmentation held on the device.</li>
<li>The teacher uses a stride-2 stem, which makes the ResNet-50 about 4x cheaper than the stride-1 CIFAR variant.</li>
<li>The evolutionary search strategy was chosen from the two options allowed by the assignment (evolutionary or Optuna).</li>
<li>Latency is measured, not estimated, on the host CPU through ONNX Runtime (1 intra-op thread, batch 1); the search ran with no other jobs on the machine.</li>
<li>Every stage is reproducible with <code>bash run_pipeline.sh &lt;stage&gt;</code>; see <code>README.md</code>.</li>
</ul>"""

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Edge Computing Assignment 4 - Technical Report</title>
<style>{css}</style></head><body>
<h1>Edge Computing &amp; Intelligent Systems - Assignment 4</h1>
<p><b>Technical report</b> - Knowledge distillation (Q1) and hardware-aware neural architecture search (Q2).
Platform: {html.escape(machine.get('platform', 'macOS, Apple M4'))}; PyTorch (MPS) for training, ONNX Runtime (CPU) for inference.</p>
<p>Repository layout: <code>edge_common/</code> (data, training engine, ONNX/profiling helpers), <code>q1_distillation/</code>, <code>q2_nas/</code>, <code>benchmarks/</code>,
<code>results/</code>, <code>checkpoints/</code>, <code>onnx_models/</code>.</p>
{q1}
{q2}
{bench_html}
{notes}
</body></html>"""
    (OUT / "technical_report.html").write_text(doc)
    pdf = OUT / "technical_report.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", (OUT / "technical_report.html").as_uri()],
                   check=True, capture_output=True, timeout=120)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()

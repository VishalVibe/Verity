"""
Generate a standalone, self-contained HTML report from a checkpoint file
(the same one produced by eval_pipeline_resumable.py). Open the output
.html file directly in a browser -- no server needed, no dependencies
beyond what's already in the file.

This is the week 6 UI deliverable: a claim-by-claim annotated view of
results, not a live demo with API calls (a static report is more
appropriate for a resume artifact -- it's reproducible, doesn't expose
API keys, and doesn't break when a free-tier quota resets).

Usage:
    python -m reports.generate_report_html --provider gemini
    # writes report_gemini.html in the project root
"""

import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR_CANDIDATES = [
    ROOT_DIR / "eval" / "checkpoints",
    ROOT_DIR / "checkpoints",
]
OUTPUT_DIR = ROOT_DIR

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Claim Verifier Report — {provider}</title>
<style>
  :root {{
    --bg: #F7F5F1;
    --surface: #FFFFFF;
    --text: #1A1A1A;
    --text-secondary: #5F5E5A;
    --border: rgba(0,0,0,0.12);
    --supported: #3B6D11;
    --supported-bg: #EAF3DE;
    --contradicted: #A32D2D;
    --contradicted-bg: #FCEBEB;
    --unsupported: #854F0B;
    --unsupported-bg: #FAEEDA;
    --mono: 'IBM Plex Mono', 'SF Mono', Consolas, monospace;
    --sans: -apple-system, 'Segoe UI', Inter, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    margin: 0;
    padding: 2.5rem 1.5rem;
    line-height: 1.6;
  }}
  .container {{ max-width: 760px; margin: 0 auto; }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 14px; margin: 0 0 1.75rem; }}
  .stats {{ display: flex; gap: 10px; margin-bottom: 1.75rem; flex-wrap: wrap; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; flex: 1; min-width: 140px; }}
  .stat-label {{ font-size: 12px; color: var(--text-secondary); margin: 0 0 4px; }}
  .stat-value {{ font-size: 22px; font-weight: 600; margin: 0; }}
  .claim-row {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--row-color);
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
    cursor: pointer;
  }}
  .claim-row:hover {{ border-color: var(--row-color); }}
  .claim-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }}
  .claim-text {{ font-family: var(--mono); font-size: 13.5px; margin: 0; flex: 1; }}
  .badge {{
    font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 5px;
    white-space: nowrap; text-transform: uppercase; letter-spacing: 0.02em;
  }}
  .badge.correct {{ background: var(--supported-bg); color: var(--supported); }}
  .badge.wrong {{ background: var(--contradicted-bg); color: var(--contradicted); }}
  .detail {{ display: none; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 13px; }}
  .claim-row.open .detail {{ display: block; }}
  .detail-label {{ color: var(--text-secondary); margin: 0 0 2px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; }}
  .detail-value {{ margin: 0 0 10px; font-family: var(--mono); }}
  .verdict-pill {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Claim verifier report</h1>
  <p class="subtitle">Provider: {provider} · {total} claims evaluated · {accuracy_pct}% accuracy</p>

  <div class="stats">
    <div class="stat"><p class="stat-label">Total claims</p><p class="stat-value">{total}</p></div>
    <div class="stat"><p class="stat-label">Correct</p><p class="stat-value" style="color:var(--supported)">{correct}</p></div>
    <div class="stat"><p class="stat-label">Incorrect</p><p class="stat-value" style="color:var(--contradicted)">{incorrect}</p></div>
    <div class="stat"><p class="stat-label">Accuracy</p><p class="stat-value">{accuracy_pct}%</p></div>
  </div>

  <div id="claims-list">{claim_rows}</div>
</div>

<script>
document.querySelectorAll('.claim-row').forEach(row => {{
  row.addEventListener('click', () => row.classList.toggle('open'));
}});
</script>
</body>
</html>
"""

VERDICT_COLORS = {
    "supported": ("var(--supported)", "var(--supported-bg)"),
    "contradicted": ("var(--contradicted)", "var(--contradicted-bg)"),
    "unsupported": ("var(--unsupported)", "var(--unsupported-bg)"),
}

CLAIM_ROW_TEMPLATE = """
  <div class="claim-row" style="--row-color: {row_color};">
    <div class="claim-top">
      <p class="claim-text">{claim_text}</p>
      <span class="badge {correctness_class}">{correctness_label}</span>
    </div>
    <div class="detail">
      <p class="detail-label">True label</p>
      <p class="detail-value"><span class="verdict-pill" style="color:{true_color};background:{true_bg}">{true_label}</span></p>
      <p class="detail-label">Predicted label</p>
      <p class="detail-value"><span class="verdict-pill" style="color:{pred_color};background:{pred_bg}">{pred_label}</span></p>
      <p class="detail-label">Evidence used</p>
      <p class="detail-value">{evidence}</p>
      <p class="detail-label">Reasoning</p>
      <p class="detail-value" style="font-family: var(--sans);">{reasoning}</p>
    </div>
  </div>"""


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()

    checkpoint_name = f"results_{args.provider}.json"
    checkpoint_path = next(
        (path / checkpoint_name for path in CHECKPOINT_DIR_CANDIDATES if (path / checkpoint_name).exists()),
        CHECKPOINT_DIR_CANDIDATES[0] / checkpoint_name,
    )
    if not checkpoint_path.exists():
        searched = ", ".join(str(path / checkpoint_name) for path in CHECKPOINT_DIR_CANDIDATES)
        print(f"No checkpoint found. Searched: {searched}. Run eval_pipeline_resumable.py first.")
        return

    with open(checkpoint_path) as f:
        results = json.load(f)

    if not results:
        print("Checkpoint is empty -- nothing to report.")
        return

    rows_html = []
    correct_count = 0
    for r in results.values():
        is_correct = r["predicted_label"] == r["true_label"]
        correct_count += is_correct

        true_color, true_bg = VERDICT_COLORS[r["true_label"]]
        pred_color, pred_bg = VERDICT_COLORS[r["predicted_label"]]
        row_color = "var(--supported)" if is_correct else "var(--contradicted)"

        rows_html.append(
            CLAIM_ROW_TEMPLATE.format(
                row_color=row_color,
                claim_text=_escape(r["claim"]),
                correctness_class="correct" if is_correct else "wrong",
                correctness_label="correct" if is_correct else "wrong",
                true_color=true_color,
                true_bg=true_bg,
                true_label=r["true_label"],
                pred_color=pred_color,
                pred_bg=pred_bg,
                pred_label=r["predicted_label"],
                evidence=_escape(r["evidence_used"]),
                reasoning=_escape(r["reasoning"]),
            )
        )

    total = len(results)
    accuracy_pct = round(100 * correct_count / total, 1) if total else 0.0

    html = HTML_TEMPLATE.format(
        provider=args.provider,
        total=total,
        correct=correct_count,
        incorrect=total - correct_count,
        accuracy_pct=accuracy_pct,
        claim_rows="".join(rows_html),
    )

    output_path = OUTPUT_DIR / f"report_{args.provider}.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report written to {output_path}")
    print(f"Open it directly in a browser -- no server needed.")


if __name__ == "__main__":
    main()

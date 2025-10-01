"""Build an HTML dashboard summarising cohort analytics output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import base64
from typing import Dict, Iterable, List, Tuple

import pandas as pd

try:
    from .peer_benchmark_analysis import (
        KEY_METRICS,
        PRIMARY_PERFORMANCE_METRIC,
        RANK_PREFIX,
        SHARPE_METRIC,
        TOP_PRIMARY_COUNT,
        TOP_RELATIVE_COUNT,
        TOP_RANK_COUNT,
        TOP_SHARPE_COUNT,
    )
except ImportError:
    from peer_benchmark_analysis import (
        KEY_METRICS,
        PRIMARY_PERFORMANCE_METRIC,
        RANK_PREFIX,
        SHARPE_METRIC,
        TOP_PRIMARY_COUNT,
        TOP_RELATIVE_COUNT,
        TOP_RANK_COUNT,
        TOP_SHARPE_COUNT,
    )

TABLE_CLASS = "table table-sm"
CSS = """
body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
header { background: #264653; color: #fff; padding: 1.5rem; }
main { padding: 1.5rem; }
details.sheet { margin-bottom: 1.5rem; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); background: #fff; overflow: hidden; }
details.sheet summary { cursor: pointer; list-style: none; padding: 1rem 1.25rem; font-size: 1.25rem; font-weight: 600; color: #1d3557; border-bottom: 1px solid #e0e0e0; }
details.sheet summary::-webkit-details-marker { display: none; }
details.sheet[open] summary { background: #eef3f8; }
.sheet-body { padding: 1.25rem; }
section.metric { margin-top: 1.5rem; }
section.metric h3 { margin-bottom: 0.75rem; color: #1d3557; }
.metric-content { display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: flex-start; }
.metric-column { display: flex; flex-direction: column; gap: 1rem; flex: 1 1 320px; }
.metric-column.table-column { flex: 2 1 520px; }
.metric-column.chart-column { max-width: 420px; }
.metric-card { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.75rem; }
.metric-card h4 { margin: 0 0 0.5rem; font-size: 1rem; color: #1d3557; }
.metric-card img { max-width: 100%; height: auto; display: block; cursor: zoom-in; }
table.table { border-collapse: collapse; width: 100%; background: #fff; }
table.table thead { background: #457b9d; color: #fff; }
table.table td, table.table th { padding: 0.5rem; border: 1px solid #dee2e6; font-size: 0.9rem; }
.band-top { background: #e8f5e9; }
.band-bottom { background: #ffebee; }
.caption { font-size: 0.85rem; color: #555; margin: 0.25rem 0 0.75rem; }
.lightbox { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.75); display: none; align-items: center; justify-content: center; z-index: 1000; }
.lightbox.show { display: flex; }
.lightbox img { max-width: 90%; max-height: 90%; border: 4px solid #fff; border-radius: 6px; box-shadow: 0 0 20px rgba(0,0,0,0.4); }
.lightbox a { position: absolute; bottom: 2rem; right: 2rem; background: #fff; color: #264653; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-weight: 600; }
"""

LIGHTBOX_SCRIPT = """  <script>
    document.addEventListener('DOMContentLoaded', function () {{
      const overlay = document.createElement('div');
      overlay.className = 'lightbox';
      const bigImg = document.createElement('img');
      const downloadAnchor = document.createElement('a');
      downloadAnchor.textContent = 'Download chart';
      overlay.appendChild(bigImg);
      overlay.appendChild(downloadAnchor);
      overlay.addEventListener('click', function (event) {{
        if (event.target === overlay) {{
          overlay.classList.remove('show');
        }}
      }});
      document.body.appendChild(overlay);

      document.querySelectorAll('img.zoomable').forEach(function (img) {{
        img.addEventListener('click', function () {{
          bigImg.src = img.src;
          const name = img.getAttribute('data-download') || 'chart.png';
          downloadAnchor.href = img.src;
          downloadAnchor.download = name;
          overlay.classList.add('show');
        }});
      }});
    }});
  </script>"""



SHEET_LABELS: Dict[str, str] = {
    "equity": "Equity",
    "fixed_income": "Fixed Income",
    "hybrid_alternative": "Hybrid & Alternative",
    "uk": "UK",
}

PERCENT_KEYWORDS = ("Total Ret", "Over/Under")
LOW_IS_GOOD_KEYWORDS = ("% Rank", "Rank Cat")


def relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start)


def load_top_bottom(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    top = df[df["band"] == "top"].copy()
    bottom = df[df["band"] == "bottom"].copy()
    return top, bottom


def format_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [col for col in columns if col in df.columns]
    table = df.loc[:, cols].copy()
    numeric_cols = table.select_dtypes(include=["number"]).columns
    table[numeric_cols] = table[numeric_cols].round(2)
    return table


def value_behaviour(columns: Iterable[str]) -> Dict[str, str]:
    behaviour: Dict[str, str] = {}
    for col in columns:
        if any(key in col for key in LOW_IS_GOOD_KEYWORDS):
            behaviour[col] = "low"
        else:
            behaviour[col] = "high"
    return behaviour


def blend_color(weight: float, low: Tuple[int, int, int], high: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(int(low[i] + (high[i] - low[i]) * weight) for i in range(3))


def cell_background(value: float, min_val: float, max_val: float, invert: bool) -> str:
    if pd.isna(value):
        return ""
    if max_val - min_val < 1e-9:
        norm = 0.5
    else:
        norm = (value - min_val) / (max_val - min_val)
    norm = max(0.0, min(1.0, norm))
    if invert:
        norm = 1.0 - norm
    if norm <= 0.5:
        ratio = norm / 0.5
        color = blend_color(ratio, (252, 232, 230), (255, 255, 255))
    else:
        ratio = (norm - 0.5) / 0.5
        color = blend_color(ratio, (255, 255, 255), (212, 241, 227))
    return f"background-color: rgb({color[0]}, {color[1]}, {color[2]});"


def table_card(df: pd.DataFrame, title: str, behaviour: Dict[str, str]) -> str:
    if df.empty:
        return f"<div class='metric-card table-card'><h4>{title}</h4><p>No data available.</p></div>"
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    min_max = {col: (df[col].min(), df[col].max()) for col in numeric_cols}
    header_cells = "".join(f"<th>{col}</th>" for col in df.columns)
    body_rows: List[str] = []
    for _, row in df.iterrows():
        cells: List[str] = []
        for col in df.columns:
            val = row[col]
            style_attr = ""
            display = ""
            if pd.isna(val):
                display = ""
            elif col in numeric_cols:
                display = f"{val:,.2f}"
                style = cell_background(val, *min_max[col], invert=(behaviour.get(col) == "low"))
                if style:
                    style_attr = f" style=\"{style}\""
            else:
                display = str(val)
            cells.append(f"<td{style_attr}>{display}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    table_html = (
        f"<table class='{TABLE_CLASS}'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )
    return f"<div class='metric-card table-card'><h4>{title}</h4>{table_html}</div>"


def chart_card(img: Path, title: str, output_dir: Path) -> str:
    if not img.exists():
        return ""
    data = base64.b64encode(img.read_bytes()).decode('ascii')
    data_uri = f"data:image/png;base64,{data}"
    return (
        "<div class='metric-card chart-card'>"
        f"<h4>{title}</h4>"
        f"<img src='{data_uri}' alt='{title}' class='zoomable' data-download='{img.name}'>"
        f"<a href='{data_uri}' download='{img.name}' class='caption'>Download chart</a>"
        "</div>"
    )


def combine_columns(tables: List[str], charts: List[str]) -> str:
    columns: List[str] = []
    if tables:
        columns.append(f"<div class='metric-column table-column'>{''.join(tables)}</div>")
    if charts:
        columns.append(f"<div class='metric-column chart-column'>{''.join(charts)}</div>")
    if not columns:
        return ""
    return f"<div class='metric-content'>{''.join(columns)}</div>"


def build_primary_section(sheet_dir: Path, output_dir: Path) -> str:
    primary_files = list(sheet_dir.glob('*_primary_top_bottom.csv'))
    if not primary_files:
        return ""
    sections: List[str] = []
    for csv_path in primary_files:
        top, bottom = load_top_bottom(csv_path)
        metric = PRIMARY_PERFORMANCE_METRIC
        columns = ["Name", "Ticker", "section", metric, "Total Ret % Rank Cat YTD (Daily)"]
        top_table = format_table(top, columns)
        bottom_table = format_table(bottom, columns)
        behaviour = value_behaviour([col for col in top_table.columns if pd.api.types.is_numeric_dtype(top_table[col])])
        base = csv_path.stem.replace('_top_bottom', '')
        top_img = sheet_dir / f"{base}_top_{TOP_PRIMARY_COUNT}.png"
        bottom_img = sheet_dir / f"{base}_bottom_{TOP_PRIMARY_COUNT}.png"
        table_blocks = [
            table_card(top_table, f"Top {len(top_table)} primary funds", behaviour),
            table_card(bottom_table, f"Bottom {len(bottom_table)} primary funds", behaviour),
        ]
        chart_blocks = [
            chart_card(top_img, f"Top {TOP_PRIMARY_COUNT} chart", output_dir),
            chart_card(bottom_img, f"Bottom {TOP_PRIMARY_COUNT} chart", output_dir),
        ]
        content = combine_columns(table_blocks, [block for block in chart_blocks if block])
        sections.append(
            "<section class='metric'>"
            f"<h3>Primary Funds - {metric}</h3>"
            f"{content}"
            "<p class='caption'>Top/bottom determined using annualised Total Ret YTD (Daily) among cohort primaries.</p>"
            "</section>"
        )
    return "".join(sections)


def build_over_under_section(sheet_dir: Path, output_dir: Path) -> str:
    files = list(sheet_dir.glob('*_over_under_top_bottom.csv'))
    if not files:
        return ""
    sections: List[str] = []
    for csv_path in sorted(files):
        top, bottom = load_top_bottom(csv_path)
        delta_cols = [col for col in top.columns if col.endswith('Over/Under')]
        if not delta_cols:
            continue
        delta = delta_cols[0]
        metric_label = delta.replace(' Over/Under', '')
        columns = ["primary_name", "primary_ticker", "benchmark_name", delta]
        top_table = format_table(top, columns).rename(columns={
            "primary_name": "Fund",
            "primary_ticker": "Ticker",
            "benchmark_name": "Benchmark",
        })
        bottom_table = format_table(bottom, columns).rename(columns={
            "primary_name": "Fund",
            "primary_ticker": "Ticker",
            "benchmark_name": "Benchmark",
        })
        behaviour = value_behaviour([col for col in top_table.columns if pd.api.types.is_numeric_dtype(top_table[col])])
        base = csv_path.stem.replace('_top_bottom', '')
        top_img = sheet_dir / f"{base}_top_{TOP_RELATIVE_COUNT}.png"
        bottom_img = sheet_dir / f"{base}_bottom_{TOP_RELATIVE_COUNT}.png"
        table_blocks = [
            table_card(top_table, f"Top {len(top_table)} outperformers", behaviour),
            table_card(bottom_table, f"Bottom {len(bottom_table)} vs benchmark", behaviour),
        ]
        chart_blocks = [
            chart_card(top_img, f"Top {TOP_RELATIVE_COUNT} chart", output_dir),
            chart_card(bottom_img, f"Bottom {TOP_RELATIVE_COUNT} chart", output_dir),
        ]
        content = combine_columns(table_blocks, [block for block in chart_blocks if block])
        sections.append(
            "<section class='metric'>"
            f"<h3>Relative Performance - {metric_label}</h3>"
            f"{content}"
            "<p class='caption'>Positive values indicate the fund exceeded its benchmark over the stated period.</p>"
            "</section>"
        )
    return "".join(sections)


def build_rank_section(sheet_dir: Path, output_dir: Path) -> str:
    files = list(sheet_dir.glob('*_rank_top_bottom.csv'))
    if not files:
        return ""
    sections: List[str] = []
    for csv_path in sorted(files):
        top, bottom = load_top_bottom(csv_path)
        rank_cols = [col for col in top.columns if col.startswith(RANK_PREFIX)]
        if not rank_cols:
            continue
        rank_col = rank_cols[0]
        columns = ["Name", "Ticker", rank_col]
        top_table = format_table(top, columns)
        bottom_table = format_table(bottom, columns)
        behaviour = value_behaviour([rank_col])
        base = csv_path.stem.replace('_top_bottom', '')
        top_img = sheet_dir / f"{base}_top_{TOP_RANK_COUNT}.png"
        bottom_img = sheet_dir / f"{base}_bottom_{TOP_RANK_COUNT}.png"
        table_blocks = [
            table_card(top_table, f"Top-ranked ({len(top_table)})", behaviour),
            table_card(bottom_table, f"Bottom-ranked ({len(bottom_table)})", behaviour),
        ]
        chart_blocks = [
            chart_card(top_img, f"Top {TOP_RANK_COUNT} chart", output_dir),
            chart_card(bottom_img, f"Bottom {TOP_RANK_COUNT} chart", output_dir),
        ]
        content = combine_columns(table_blocks, [block for block in chart_blocks if block])
        sections.append(
            "<section class='metric'>"
            f"<h3>Category Rank - {rank_col}</h3>"
            f"{content}"
            "<p class='caption'>Lower rank values indicate better peer positioning (1 = best).</p>"
            "</section>"
        )
    return "".join(sections)


def build_sharpe_section(sheet_dir: Path, output_dir: Path) -> str:
    files = list(sheet_dir.glob('*sharpe_ratio_1_yr_mo_end_top_bottom.csv'))
    if not files:
        return ""
    sections: List[str] = []
    for csv_path in files:
        top, bottom = load_top_bottom(csv_path)
        sharpe_col = SHARPE_METRIC
        columns = ["Name", "Ticker", sharpe_col]
        top_table = format_table(top, columns)
        bottom_table = format_table(bottom, columns)
        behaviour = value_behaviour([sharpe_col])
        base = csv_path.stem.replace('_top_bottom', '')
        top_img = sheet_dir / f"{base}_top_{TOP_SHARPE_COUNT}.png"
        bottom_img = sheet_dir / f"{base}_bottom_{TOP_SHARPE_COUNT}.png"
        table_blocks = [
            table_card(top_table, f"Top {len(top_table)} Sharpe ratios", behaviour),
            table_card(bottom_table, f"Bottom {len(bottom_table)} Sharpe ratios", behaviour),
        ]
        chart_blocks = [
            chart_card(top_img, f"Top {TOP_SHARPE_COUNT} chart", output_dir),
            chart_card(bottom_img, f"Bottom {TOP_SHARPE_COUNT} chart", output_dir),
        ]
        content = combine_columns(table_blocks, [block for block in chart_blocks if block])
        sections.append(
            "<section class='metric'>"
            f"<h3>Risk-Adjusted Returns - {sharpe_col}</h3>"
            f"{content}"
            "<p class='caption'>Sharpe ratios use the 1-year, month-end calculation supplied in the workbook.</p>"
            "</section>"
        )
    return "".join(sections)


def build_sheet_section(sheet_dir: Path, output_dir: Path) -> str:
    sheet_key = sheet_dir.name
    sheet_label = SHEET_LABELS.get(sheet_key, sheet_key.replace('_', ' ').title())
    metrics = [
        build_primary_section(sheet_dir, output_dir),
        build_over_under_section(sheet_dir, output_dir),
        build_rank_section(sheet_dir, output_dir),
        build_sharpe_section(sheet_dir, output_dir),
    ]
    metrics_content = "".join(section for section in metrics if section)
    if not metrics_content:
        return ""
    open_attr = " open" if sheet_key == "equity" else ""
    return (
        f"<details class='sheet'{open_attr}>"
        f"<summary>{sheet_label}</summary>"
        f"<div class='sheet-body'>{metrics_content}</div>"
        "</details>"
    )


def build_dashboard(input_dir: Path, output_path: Path) -> None:
    sheet_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    sections = [build_sheet_section(sheet_dir, output_path.parent) for sheet_dir in sheet_dirs]
    body = "".join(section for section in sections if section)
    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Fund vs Benchmark Dashboard</title>
  <style>{CSS}</style>
</head>
<body>
  <header>
    <h1>Fund vs Benchmark Performance Dashboard</h1>
    <p>Automatically generated from cohort analysis outputs.</p>
  </header>
  <main>
    {body}
  </main>
  {LIGHTBOX_SCRIPT}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("analysis/output_all"),
        help="Directory containing per-sheet CSV/PNG outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/output_all/dashboard.html"),
        help="Path for the generated HTML dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    build_dashboard(input_dir, output_path)
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()

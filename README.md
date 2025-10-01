# Fund vs Benchmark Dashboard

This repository contains Python utilities and a Streamlit app that parse the Daily Fund ETF Peer Relative Benchmark Performance workbook, build cohort comparisons, and render an interactive dashboard.

## Structure

- `analysis/` – reusable parsing/visualization helpers and CLI scripts (outputs land in `analysis/output_all/`).
- `streamlit_app/app.py` – Streamlit frontend for uploading workbooks and refreshing the dashboard in the browser.
- `requirements.txt` – minimal dependency list for local installs and cloud deployment.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Refresh via Streamlit

```bash
streamlit run streamlit_app/app.py
```

Upload a workbook (or rely on the default path) and click **Run analysis and refresh dashboard**. The latest dashboard renders inline and can be downloaded.

## Batch refresh (CLI)

```bash
python analysis/peer_benchmark_analysis.py --workbook "path/to/workbook.xlsx" --export-dir analysis/output_all --quiet
python analysis/build_dashboard.py --input-dir analysis/output_all --output analysis/output_all/dashboard.html
```

## Outputs

Generated CSVs/PNGs/HTML live in `analysis/output_all/` and are gitignored by default.

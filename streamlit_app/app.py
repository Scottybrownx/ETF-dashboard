"""Streamlit app for fund vs benchmark dashboard refresh."""

from __future__ import annotations

import shutil
from pathlib import Path
import sys

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from analysis.peer_benchmark_analysis import run_analysis
from analysis.build_dashboard import build_dashboard

OUTPUT_DIR = BASE_DIR / "analysis" / "output_all"
DEFAULT_WORKBOOK = Path(r"C:\Users\nscott\Downloads\Daily Fund ETF Peer Relative Benchmark Performance_Crossborder & UK 29Sept25.xlsx")
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"


def save_uploaded_file(uploaded_file: UploadedFile, target_path: Path) -> None:
    with target_path.open("wb") as fh:
        shutil.copyfileobj(uploaded_file, fh)


def run_full_analysis(workbook_path: Path) -> None:
    run_analysis(
        workbook=workbook_path,
        sheet_names=None,
        export_dir=OUTPUT_DIR,
        highlight=None,
        quiet=True,
    )
    build_dashboard(OUTPUT_DIR, DASHBOARD_PATH)


def render_dashboard() -> None:
    if not DASHBOARD_PATH.exists():
        st.info("Run an analysis to generate the dashboard.")
        return
    with DASHBOARD_PATH.open("r", encoding="utf-8") as fh:
        html = fh.read()
    st.components.v1.html(html, height=900, scrolling=True)
    st.download_button(
        "Download dashboard HTML",
        data=html,
        file_name="dashboard.html",
        mime="text/html",
    )


def main() -> None:
    st.set_page_config(page_title="Fund vs Benchmark Dashboard", layout="wide")
    st.title("Fund vs Benchmark Dashboard")
    st.markdown(
        "Upload a workbook (or use the default path), then run the analysis to refresh charts and tables."
    )

    with st.expander("Configuration", expanded=False):
        st.write("Default workbook path:")
        st.code(str(DEFAULT_WORKBOOK))

    uploaded_file = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx"])
    workbook_path = DEFAULT_WORKBOOK

    if uploaded_file:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        uploaded_path = OUTPUT_DIR / uploaded_file.name
        save_uploaded_file(uploaded_file, uploaded_path)
        workbook_path = uploaded_path
        st.success(f"Uploaded workbook saved to {uploaded_path}")
    elif workbook_path.exists():
        st.info(f"Using default workbook: {workbook_path}")
    else:
        st.warning("No workbook supplied and default workbook not found.")

    if st.button("Run analysis and refresh dashboard", type="primary"):
        if not workbook_path.exists():
            st.error("Workbook path not found. Upload a file or update the default path.")
        else:
            with st.spinner("Analyzing workbook and updating dashboard..."):
                run_full_analysis(workbook_path)
            st.success("Dashboard refreshed!")

    st.markdown("---")
    render_dashboard()


if __name__ == "__main__":
    main()

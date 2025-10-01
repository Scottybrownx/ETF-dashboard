"""Parse Daily Fund ETF Peer Relative Benchmark Performance workbooks and
produce cohort-level analytics with highlight tables and charts."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype

plt.switch_backend("Agg")

NUMERIC_HINTS: Tuple[str, ...] = (
    "Fund Size",
    "Morningstar_Rating_Overall",
    "Total Ret 1 Day (Daily)",
    "Total Ret 1 Wk (Daily)",
    "Total Ret MTD (Daily)",
    "Total Ret QTD (Daily)",
    "Total Ret % Rank Cat QTD (Daily)",
    "Total Ret YTD (Daily)",
    "Total Ret % Rank Cat YTD (Daily)",
    "Total_Ret_1YR_daily",
    "Total Ret % Rank Cat 1 Yr (Daily)",
    "Total Ret Annlzd 3 Yr (Daily)",
    "Total Ret % Rank Cat 3 Yr (Daily)",
    "Total Ret Annlzd 5 Yr (Daily)",
    "Total Ret % Rank Cat 5 Yr (Daily)",
    "Total Ret Annlzd 10 Yr (Daily)",
    "Total Ret % Rank Cat 10 Yr (Daily)",
    "Std Dev 1 Yr (Mo-End)",
    "Std Dev 3 Yr (Mo-End)",
    "Alpha 1 Yr (Mo-End)",
    "Alpha 3 Yr (Mo-End)",
    "Beta 1 Yr (Mo-End)",
    "Beta 3 Yr (Mo-End)",
    "Information Ratio 1 Yr (Mo-End)",
    "Information Ratio 3 Yr (Mo-End)",
    "Tracking Error 1 Yr (Mo-End)",
    "Tracking Error 3 Yr (Mo-End)",
    "Sharpe Ratio 1 Yr (Mo-End)",
    "Sharpe Ratio 3 Yr (Mo-End)",
    "Upside Capture Ratio 1 Yr (Mo-End)",
    "Upside Capture Ratio 3 Yr (Mo-End)",
    "Downside Capture Ratio 1 Yr (Mo-End)",
    "Downside Capture Ratio 3 Yr (Mo-End)",
    "Average Eff Duration Survey",
    "Average YTM Survey",
    "Fixd-Inc_YTW_Avg_Calc_Net_FI%",
    "Average Credit Quality",
    "Portfolio Date",
)

KEY_METRICS: Tuple[str, ...] = (
    "Total Ret 1 Day (Daily)",
    "Total Ret QTD (Daily)",
    "Total Ret YTD (Daily)",
    "Total_Ret_1YR_daily",
)

META_COLUMNS: Tuple[str, ...] = (
    "sheet",
    "source_row",
    "Ticker",
    "Name",
    "row_type",
    "section",
)

PRIMARY_PERFORMANCE_METRIC = "Total Ret YTD (Daily)"
RANK_PREFIX = "Total Ret % Rank Cat"
SHARPE_METRIC = "Sharpe Ratio 1 Yr (Mo-End)"

TOP_PRIMARY_COUNT = 20
TOP_RELATIVE_COUNT = 10
TOP_RANK_COUNT = 10
TOP_SHARPE_COUNT = 10

BAR_COLOR_POSITIVE = "#2A9D8F"
BAR_COLOR_NEGATIVE = "#E76F51"
BAR_COLOR_NEUTRAL = "#577590"


@dataclass
class BenchmarkPair:
    benchmark: Dict[str, Any]
    over_under: Dict[str, Any]


@dataclass
class Cohort:
    sheet: str
    cohort_id: str
    section: Optional[str]
    primary: Dict[str, Any]
    benchmarks: List[BenchmarkPair] = field(default_factory=list)
    competitors: List[Dict[str, Any]] = field(default_factory=list)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return cleaned.strip("_") or "value"


def compose_label(name: Optional[str], ticker: Optional[str]) -> str:
    name = (name or "").strip()
    ticker = (ticker or "").strip()
    if ticker and name:
        return f"{name} ({ticker})"
    return name or ticker or "Unknown"


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def create_bar_chart(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    path: Path,
    ascending: bool,
    color: str,
) -> None:
    if data.empty:
        return
    ordered = data.sort_values(value_col, ascending=ascending)
    if not ascending:
        ordered = ordered.iloc[::-1]
    labels = ordered["label"]
    values = ordered[value_col]
    fig_height = max(4, 0.35 * len(ordered))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(labels, values, color=color)
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_unique(labels: Sequence[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for label in labels:
        clean = str(label or "").strip()
        if not clean:
            clean = "Unnamed"
        clean = clean.replace("\n", " / ")
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        result.append(clean)
    return result


def coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        return series
    cleaned = series.astype(str).str.replace("%", "", regex=False)
    cleaned = cleaned.replace({"nan": pd.NA, "": pd.NA, "-": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_idx: Optional[int] = None
    first_col = raw.iloc[:, 0]
    for idx, value in first_col.items():
        if isinstance(value, str) and value.strip().lower() == "ticker":
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not locate header row in sheet '{sheet_name}'")

    header_row = raw.iloc[header_idx]
    headers = make_unique(header_row)
    data = raw.iloc[header_idx + 1 :].copy()
    data.columns = headers
    data.insert(0, "source_row", raw.iloc[header_idx + 1 :].index)
    data.insert(0, "sheet", sheet_name)

    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            data[col] = data[col].replace({"": pd.NA, "-": pd.NA})

    drop_cols = [
        col
        for col in data.columns
        if col not in {"sheet", "source_row"} and data[col].isna().all()
    ]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    for col in NUMERIC_HINTS:
        if col in data.columns:
            data[col] = coerce_numeric(data[col])

    data["Ticker"] = data.get("Ticker", pd.Series(index=data.index, dtype="object")).fillna("")
    data["Name"] = data.get("Name", pd.Series(index=data.index, dtype="object")).fillna("")
    data["Ticker"] = data["Ticker"].astype(str).str.strip()
    data["Name"] = data["Name"].astype(str).str.strip()

    other_cols = [
        col
        for col in data.columns
        if col not in {"sheet", "source_row", "Ticker", "Name", "row_type", "section"}
    ]
    non_na_counts = data[other_cols].notna().sum(axis=1)
    row_type = pd.Series("data", index=data.index)
    row_type[(data["Name"].str.lower() == "over/under")] = "over_under"
    row_type[(data["Ticker"] == "") & (data["Name"] == "") & (non_na_counts == 0)] = "blank"
    row_type[(data["Name"] == "") & (data["Ticker"] != "") & (non_na_counts == 0)] = "section"
    data["row_type"] = row_type

    return data


def iter_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    ordered = df.sort_values(["source_row"]).reset_index(drop=True)
    records: List[Dict[str, Any]] = ordered.to_dict("records")
    return records


def parse_cohorts(df: pd.DataFrame) -> List[Cohort]:
    records = iter_records(df)
    cohorts: List[Cohort] = []
    sheet_counts: Dict[str, int] = {}
    current_section: Optional[str] = None
    i = 0

    def next_is_primary(start: int) -> bool:
        return (
            start + 2 < len(records)
            and records[start]["row_type"] == "data"
            and records[start + 1]["row_type"] == "data"
            and records[start + 2]["row_type"] == "over_under"
        )

    while i < len(records):
        rec = records[i]
        row_type = rec.get("row_type")
        if row_type == "section":
            current_section = rec.get("Ticker") or rec.get("Name") or current_section
            i += 1
            continue
        if row_type in {"blank", "over_under"}:
            i += 1
            continue
        if not next_is_primary(i):
            i += 1
            continue

        sheet = rec["sheet"]
        count = sheet_counts.get(sheet, 0) + 1
        sheet_counts[sheet] = count
        cohort_id = f"{sheet}-{count:03d}"
        primary = rec
        j = i + 1
        benchmarks: List[BenchmarkPair] = []
        while j + 1 < len(records) and records[j]["row_type"] == "data" and records[j + 1]["row_type"] == "over_under":
            benchmarks.append(BenchmarkPair(records[j], records[j + 1]))
            j += 2

        competitors: List[Dict[str, Any]] = []
        k = j
        while k < len(records):
            nxt = records[k]
            nxt_type = nxt.get("row_type")
            if nxt_type == "blank":
                k += 1
                break
            if nxt_type == "section":
                break
            if nxt_type == "over_under":
                k += 1
                continue
            if next_is_primary(k):
                break
            competitors.append(nxt)
            k += 1

        cohorts.append(
            Cohort(
                sheet=sheet,
                cohort_id=cohort_id,
                section=current_section,
                primary=primary,
                benchmarks=benchmarks,
                competitors=competitors,
            )
        )
        i = k

    return cohorts


def gather_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols: List[str] = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        if is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def build_entity_table(cohorts: Iterable[Cohort], numeric_cols: Sequence[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    subset_cols = [
        "Fund Size",
        "Morningstar_Rating_Overall",
        *numeric_cols,
    ]
    subset_cols = list(dict.fromkeys(subset_cols))

    def add_entry(cohort: Cohort, role: str, entry: Dict[str, Any]) -> None:
        row: Dict[str, Any] = {
            "sheet": cohort.sheet,
            "cohort_id": cohort.cohort_id,
            "section": cohort.section,
            "role": role,
            "Ticker": entry.get("Ticker"),
            "Name": entry.get("Name"),
        }
        for col in subset_cols:
            if col in entry:
                row[col] = entry[col]
        records.append(row)

    for cohort in cohorts:
        add_entry(cohort, "primary", cohort.primary)
        for pair in cohort.benchmarks:
            add_entry(cohort, "benchmark", pair.benchmark)
        for comp in cohort.competitors:
            add_entry(cohort, "competitor", comp)

    return pd.DataFrame(records)


def build_over_under_table(
    cohorts: Iterable[Cohort], numeric_cols: Sequence[str]
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for cohort in cohorts:
        primary = cohort.primary
        for pair in cohort.benchmarks:
            base: Dict[str, Any] = {
                "sheet": cohort.sheet,
                "cohort_id": cohort.cohort_id,
                "section": cohort.section,
                "primary_ticker": primary.get("Ticker"),
                "primary_name": primary.get("Name"),
                "benchmark_ticker": pair.benchmark.get("Ticker"),
                "benchmark_name": pair.benchmark.get("Name"),
            }
            for col in numeric_cols:
                if col in pair.over_under:
                    base[f"{col} Over/Under"] = pair.over_under[col]
            records.append(base)
    return pd.DataFrame(records)


def print_summary(
    cohorts: Sequence[Cohort],
    numeric_cols: Sequence[str],
    highlight: Optional[str] = None,
) -> None:
    if not cohorts:
        print("No cohorts detected.")
        return
    print(f"Detected {len(cohorts)} cohorts across sheet '{cohorts[0].sheet}'.")
    metrics_to_show = [col for col in KEY_METRICS if col in numeric_cols]
    for cohort in cohorts:
        primary_name = cohort.primary.get("Name") or "Unknown"
        print(f"\n[{cohort.cohort_id}] {primary_name}")
        if cohort.section:
            print(f"  Section: {cohort.section}")
        print(f"  Benchmarks: {len(cohort.benchmarks)} | Competitors: {len(cohort.competitors)}")
        for pair in cohort.benchmarks:
            bench_name = pair.benchmark.get("Name") or "Benchmark"
            print(f"    Benchmark: {bench_name}")
            for metric in metrics_to_show:
                value = pair.over_under.get(metric)
                if pd.notna(value):
                    print(f"      {metric} Over/Under: {value:+.4f}")
        if highlight and highlight.lower() in primary_name.lower():
            print("    * Highlighted entry matched user query *")


def generate_primary_performance_outputs(
    entities: pd.DataFrame,
    sheet: str,
    sheet_dir: Path,
    metric: str = PRIMARY_PERFORMANCE_METRIC,
    top_n: int = TOP_PRIMARY_COUNT,
) -> None:
    sheet_entities = entities[entities["sheet"] == sheet]
    primary = sheet_entities[sheet_entities["role"] == "primary"].copy()
    if metric not in primary.columns:
        return
    primary[metric] = ensure_numeric(primary[metric])
    primary = primary.dropna(subset=[metric])
    if primary.empty:
        return
    primary["label"] = primary.apply(lambda r: compose_label(r.get("Name"), r.get("Ticker")), axis=1)
    top = primary.nlargest(top_n, metric).assign(band="top")
    bottom = primary.nsmallest(top_n, metric).assign(band="bottom")
    summary = pd.concat([top, bottom], ignore_index=True)
    summary_path = sheet_dir / f"{slugify(sheet)}_{slugify(metric)}_primary_top_bottom.csv"
    summary.to_csv(summary_path, index=False)
    create_bar_chart(
        top,
        metric,
        f"{sheet}: Top {top_n} primary funds by {metric}",
        sheet_dir / f"{slugify(sheet)}_{slugify(metric)}_primary_top_{top_n}.png",
        ascending=False,
        color=BAR_COLOR_POSITIVE,
    )
    create_bar_chart(
        bottom,
        metric,
        f"{sheet}: Bottom {top_n} primary funds by {metric}",
        sheet_dir / f"{slugify(sheet)}_{slugify(metric)}_primary_bottom_{top_n}.png",
        ascending=True,
        color=BAR_COLOR_NEGATIVE,
    )


def generate_over_under_outputs(
    over: pd.DataFrame,
    sheet: str,
    sheet_dir: Path,
    metrics: Sequence[str] = KEY_METRICS,
    top_n: int = TOP_RELATIVE_COUNT,
) -> None:
    if over.empty:
        return
    sheet_over = over[over["sheet"] == sheet].copy()
    available_metrics = [m for m in metrics if f"{m} Over/Under" in sheet_over.columns]
    if not available_metrics:
        return
    for metric in available_metrics:
        delta_col = f"{metric} Over/Under"
        data = sheet_over[[
            "cohort_id",
            "section",
            "primary_name",
            "primary_ticker",
            "benchmark_name",
            "benchmark_ticker",
            delta_col,
        ]].copy()
        data[delta_col] = ensure_numeric(data[delta_col])
        data = data.dropna(subset=[delta_col])
        if data.empty:
            continue
        data["label"] = data.apply(
            lambda r: f"{compose_label(r['primary_name'], r['primary_ticker'])} vs {compose_label(r['benchmark_name'], r['benchmark_ticker'])}",
            axis=1,
        )
        top = data.nlargest(top_n, delta_col).assign(band="top")
        bottom = data.nsmallest(top_n, delta_col).assign(band="bottom")
        summary = pd.concat([top, bottom], ignore_index=True)
        summary_path = sheet_dir / f"{slugify(sheet)}_{slugify(metric)}_over_under_top_bottom.csv"
        summary.to_csv(summary_path, index=False)
        create_bar_chart(
            top,
            delta_col,
            f"{sheet}: Top {top_n} relative outperformers ({metric})",
            sheet_dir / f"{slugify(sheet)}_{slugify(metric)}_over_under_top_{top_n}.png",
            ascending=False,
            color=BAR_COLOR_POSITIVE,
        )
        create_bar_chart(
            bottom,
            delta_col,
            f"{sheet}: Bottom {top_n} relative performers ({metric})",
            sheet_dir / f"{slugify(sheet)}_{slugify(metric)}_over_under_bottom_{top_n}.png",
            ascending=True,
            color=BAR_COLOR_NEGATIVE,
        )


def generate_rank_outputs(
    entities: pd.DataFrame,
    sheet: str,
    sheet_dir: Path,
    prefix: str = RANK_PREFIX,
    top_n: int = TOP_RANK_COUNT,
) -> None:
    sheet_entities = entities[entities["sheet"] == sheet]
    primary = sheet_entities[sheet_entities["role"] == "primary"].copy()
    rank_cols = [col for col in primary.columns if col.startswith(prefix)]
    if not rank_cols:
        return
    for col in rank_cols:
        primary[col] = ensure_numeric(primary[col])
        data = primary[[
            "cohort_id",
            "section",
            "Name",
            "Ticker",
            col,
        ]].dropna(subset=[col])
        if data.empty:
            continue
        data["label"] = data.apply(lambda r: compose_label(r["Name"], r["Ticker"]), axis=1)
        top = data.nsmallest(top_n, col).assign(band="top")
        bottom = data.nlargest(top_n, col).assign(band="bottom")
        summary = pd.concat([top, bottom], ignore_index=True)
        summary_path = sheet_dir / f"{slugify(sheet)}_{slugify(col)}_rank_top_bottom.csv"
        summary.to_csv(summary_path, index=False)
        create_bar_chart(
            top,
            col,
            f"{sheet}: Top {top_n} category ranks ({col})",
            sheet_dir / f"{slugify(sheet)}_{slugify(col)}_rank_top_{top_n}.png",
            ascending=True,
            color=BAR_COLOR_POSITIVE,
        )
        create_bar_chart(
            bottom,
            col,
            f"{sheet}: Bottom {top_n} category ranks ({col})",
            sheet_dir / f"{slugify(sheet)}_{slugify(col)}_rank_bottom_{top_n}.png",
            ascending=False,
            color=BAR_COLOR_NEGATIVE,
        )


def generate_sharpe_outputs(
    entities: pd.DataFrame,
    sheet: str,
    sheet_dir: Path,
    sharpe_col: str = SHARPE_METRIC,
    top_n: int = TOP_SHARPE_COUNT,
) -> None:
    sheet_entities = entities[entities["sheet"] == sheet]
    primary = sheet_entities[sheet_entities["role"] == "primary"].copy()
    if sharpe_col not in primary.columns:
        return
    primary[sharpe_col] = ensure_numeric(primary[sharpe_col])
    data = primary[[
        "cohort_id",
        "section",
        "Name",
        "Ticker",
        sharpe_col,
    ]].dropna(subset=[sharpe_col])
    if data.empty:
        return
    data["label"] = data.apply(lambda r: compose_label(r["Name"], r["Ticker"]), axis=1)
    top = data.nlargest(top_n, sharpe_col).assign(band="top")
    bottom = data.nsmallest(top_n, sharpe_col).assign(band="bottom")
    summary = pd.concat([top, bottom], ignore_index=True)
    summary_path = sheet_dir / f"{slugify(sheet)}_{slugify(sharpe_col)}_top_bottom.csv"
    summary.to_csv(summary_path, index=False)
    create_bar_chart(
        top,
        sharpe_col,
        f"{sheet}: Top {top_n} Sharpe ratio (1Y)",
        sheet_dir / f"{slugify(sheet)}_{slugify(sharpe_col)}_top_{top_n}.png",
        ascending=False,
        color=BAR_COLOR_POSITIVE,
    )
    create_bar_chart(
        bottom,
        sharpe_col,
        f"{sheet}: Bottom {top_n} Sharpe ratio (1Y)",
        sheet_dir / f"{slugify(sheet)}_{slugify(sharpe_col)}_bottom_{top_n}.png",
        ascending=True,
        color=BAR_COLOR_NEGATIVE,
    )


def run_analysis(
    workbook: Path | str,
    sheet_names: Optional[Sequence[str]] | None = None,
    export_dir: Path | None = None,
    highlight: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, pd.DataFrame]:
    path_obj = Path(workbook)
    if not path_obj.exists():
        raise FileNotFoundError(f"Workbook not found: {path_obj}")

    xl = pd.ExcelFile(path_obj)
    sheets = list(sheet_names) if sheet_names else xl.sheet_names

    all_entities: List[pd.DataFrame] = []
    all_over: List[pd.DataFrame] = []

    export_dir_path = Path(export_dir) if export_dir else None
    if export_dir_path:
        export_dir_path.mkdir(parents=True, exist_ok=True)

    for sheet in sheets:
        df = load_sheet(path_obj, sheet)
        cohorts = parse_cohorts(df)
        numeric_cols = gather_numeric_columns(df)
        if not quiet:
            print_summary(cohorts, numeric_cols, highlight=highlight)
        if not cohorts:
            continue
        entities = build_entity_table(cohorts, numeric_cols)
        over = build_over_under_table(cohorts, numeric_cols)
        all_entities.append(entities)
        all_over.append(over)
        if export_dir_path:
            sheet_dir = export_dir_path / slugify(sheet)
            sheet_dir.mkdir(parents=True, exist_ok=True)
            generate_primary_performance_outputs(entities, sheet, sheet_dir)
            generate_over_under_outputs(over, sheet, sheet_dir, metrics=KEY_METRICS)
            generate_rank_outputs(entities, sheet, sheet_dir)
            generate_sharpe_outputs(entities, sheet, sheet_dir)

    combined_entities = pd.concat(all_entities, ignore_index=True) if all_entities else pd.DataFrame()
    combined_over = pd.concat(all_over, ignore_index=True) if all_over else pd.DataFrame()

    if export_dir_path:
        if not combined_entities.empty:
            entities_path = export_dir_path / "entities.csv"
            combined_entities.to_csv(entities_path, index=False)
            if not quiet:
                print(f"\nSaved entity table to {entities_path}")
        if not combined_over.empty:
            over_path = export_dir_path / "over_under.csv"
            combined_over.to_csv(over_path, index=False)
            if not quiet:
                print(f"Saved over/under table to {over_path}")

    return {"entities": combined_entities, "over_under": combined_over}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workbook",
        nargs="?",
        default="C:/Users/nscott/Downloads/Daily Fund ETF Peer Relative Benchmark Performance_Crossborder & UK 29Sept25.xlsx",
        help="Path to the Excel workbook to analyse.",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        help="Sheet name(s) to process. Defaults to all sheets.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Directory to write CSV outputs and charts.",
    )
    parser.add_argument(
        "--highlight",
        help="Optional substring to highlight a primary fund in the console summary.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress cohort summaries in the console.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        workbook=args.workbook,
        sheet_names=args.sheet,
        export_dir=args.export_dir,
        highlight=args.highlight,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

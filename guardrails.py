# guardrails.py
#
# Primary metric:
#   - LTV7 (revenue day_offset 0..7 per user)
#
# Guardrails:
#   - D1 retention
#   - D7 retention
#   - Economy inflation diagnostics:
#       * avg_end_balance (inflation proxy)
#       * avg_net_currency (earned - spent per active user-day)
#       * sink_coverage_ratio (total_spent / total_earned)
#
# Usage:
#   python guardrails.py --input events.parquet --outdir outputs/guardrails_report
#
# Notes:
# - Metrics are computed at USER level (LTV7) and USER-DAY level (retention/economy).
# - This report is descriptive (no CIs). Pair it with the bootstrap report for uncertainty.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EXPECTED_COLS = [
    "event_time", "user_id", "install_time", "event_name", "session_id",
    "level", "currency_delta", "currency_balance", "iap_usd", "ad_usd",
    "segment", "ab_group", "cohort_day",
]


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def fmt_money(x: float) -> str:
    return f"${float(x):,.4f}"

def fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0

def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    out = df.copy()
    if len(out) > max_rows:
        out = out.head(max_rows).copy()
    try:
        return out.to_markdown(index=False)
    except Exception:
        return out.to_string(index=False)


# -----------------------------
# Load + derive
# -----------------------------

def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_parquet(path)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns:\n  - " + "\n  - ".join(missing))

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["install_time"] = pd.to_datetime(df["install_time"], errors="coerce")

    df["event_date"] = df["event_time"].dt.date
    df["install_date"] = df["install_time"].dt.date
    df["day_offset"] = (pd.to_datetime(df["event_date"]) - pd.to_datetime(df["install_date"])).dt.days

    df["iap_usd"] = pd.to_numeric(df["iap_usd"], errors="coerce").fillna(0.0)
    df["ad_usd"] = pd.to_numeric(df["ad_usd"], errors="coerce").fillna(0.0)
    df["revenue_usd"] = df["iap_usd"] + df["ad_usd"]

    df["currency_delta"] = pd.to_numeric(df["currency_delta"], errors="coerce").fillna(0.0)
    df["currency_balance"] = pd.to_numeric(df["currency_balance"], errors="coerce").fillna(0.0)

    df["ab_group"] = df["ab_group"].astype(str)
    df["segment"] = df["segment"].astype(str)
    df["event_name"] = df["event_name"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df["session_id"] = df["session_id"].astype(str)

    return df


# -----------------------------
# Build user_daily (activity + economy + revenue)
# -----------------------------

def make_user_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Active definition: session_start preferred; else any event
    session_starts = df[df["event_name"] == "session_start"].copy()
    active_base = session_starts if len(session_starts) > 0 else df.copy()

    daily_active = (
        active_base.groupby(["user_id", "event_date"], as_index=False)
        .agg(dummy=("event_name", "size"))
    )
    daily_active["is_active"] = 1
    daily_active = daily_active.drop(columns=["dummy"])

    # Sessions/day
    if len(session_starts) > 0:
        daily_sessions = (
            session_starts.groupby(["user_id", "event_date"], as_index=False)
            .agg(sessions=("session_id", "nunique"))
        )
    else:
        daily_sessions = (
            df.groupby(["user_id", "event_date"], as_index=False)
            .agg(sessions=("session_id", "nunique"))
        )

    # Economy per user-day
    econ_df = df[df["event_name"].isin(["currency_earn", "currency_spend"])].copy()
    daily_econ = (
        econ_df.groupby(["user_id", "event_date"], as_index=False)
        .agg(
            currency_earned=("currency_delta", lambda x: float(x[x > 0].sum()) if len(x) else 0.0),
            currency_spent=("currency_delta", lambda x: float(-x[x < 0].sum()) if len(x) else 0.0),
            net_currency=("currency_delta", lambda x: float(x.sum()) if len(x) else 0.0),
        )
    )

    # End-of-day balance: last event of that day per user
    df_sorted = df.sort_values(["user_id", "event_time"])
    eod = (
        df_sorted.groupby(["user_id", "event_date"], as_index=False)
        .tail(1)[["user_id", "event_date", "currency_balance"]]
        .rename(columns={"currency_balance": "end_balance"})
    )

    # Revenue per user-day
    daily_rev = (
        df.groupby(["user_id", "event_date"], as_index=False)
        .agg(iap_usd=("iap_usd", "sum"), ad_usd=("ad_usd", "sum"))
    )
    daily_rev["total_usd"] = daily_rev["iap_usd"] + daily_rev["ad_usd"]

    # User static info
    user_static = (
        df.groupby("user_id", as_index=False)
        .agg(install_date=("install_date", "min"),
             ab_group=("ab_group", "first"),
             segment=("segment", "first"))
    )

    out = daily_active.merge(daily_sessions, on=["user_id", "event_date"], how="left")
    out = out.merge(daily_econ, on=["user_id", "event_date"], how="left")
    out = out.merge(eod, on=["user_id", "event_date"], how="left")
    out = out.merge(daily_rev, on=["user_id", "event_date"], how="left")
    out = out.merge(user_static, on="user_id", how="left")

    for c in ["sessions", "currency_earned", "currency_spent", "net_currency", "end_balance", "iap_usd", "ad_usd", "total_usd"]:
        out[c] = out[c].fillna(0.0)

    out["day_offset"] = (pd.to_datetime(out["event_date"]) - pd.to_datetime(out["install_date"])).dt.days
    return out


# -----------------------------
# Primary metric (LTV7) + guardrails
# -----------------------------

def compute_ltv7_by_group(df: pd.DataFrame) -> pd.DataFrame:
    # user-level revenue within day_offset 0..7, averaged by group
    w = df[(df["day_offset"] >= 0) & (df["day_offset"] <= 7)].copy()
    user_ltv7 = w.groupby("user_id", as_index=False).agg(ltv7=("revenue_usd", "sum"))
    user_group = df.groupby("user_id", as_index=False).agg(ab_group=("ab_group", "first"))
    user_ltv7 = user_ltv7.merge(user_group, on="user_id", how="left")

    out = (
        user_ltv7.groupby("ab_group", as_index=False)
        .agg(
            installs=("user_id", "nunique"),
            mean_ltv7=("ltv7", "mean"),
            median_ltv7=("ltv7", "median"),
            p90_ltv7=("ltv7", lambda x: float(np.quantile(x, 0.90))),
        )
    )
    return out.sort_values("ab_group")


def retention_at(user_daily: pd.DataFrame, day: int) -> pd.DataFrame:
    installs = user_daily[["user_id", "ab_group"]].drop_duplicates().groupby("ab_group", as_index=False).agg(installs=("user_id", "nunique"))
    act = (
        user_daily[(user_daily["is_active"] == 1) & (user_daily["day_offset"] == day)]
        .groupby("ab_group", as_index=False)
        .agg(active_users=("user_id", "nunique"))
    )
    out = installs.merge(act, on="ab_group", how="left").fillna({"active_users": 0})
    out[f"d{day}_retention"] = out["active_users"] / out["installs"].replace(0, np.nan)
    out[f"d{day}_retention"] = out[f"d{day}_retention"].fillna(0.0)
    return out[["ab_group", "installs", "active_users", f"d{day}_retention"]].sort_values("ab_group")


def economy_daily_by_group(user_daily: pd.DataFrame) -> pd.DataFrame:
    # economy diagnostics per date and group (only active user-days)
    active = user_daily[user_daily["is_active"] == 1].copy()
    daily = (
        active.groupby(["event_date", "ab_group"], as_index=False)
        .agg(
            dau=("user_id", "nunique"),
            avg_end_balance=("end_balance", "mean"),
            avg_net_currency=("net_currency", "mean"),
            total_earned=("currency_earned", "sum"),
            total_spent=("currency_spent", "sum"),
        )
        .sort_values(["event_date", "ab_group"])
    )
    daily["sink_coverage_ratio"] = daily["total_spent"] / daily["total_earned"].replace(0, np.nan)
    daily["sink_coverage_ratio"] = daily["sink_coverage_ratio"].fillna(0.0)
    return daily


# -----------------------------
# Plots
# -----------------------------

def plot_ltv7_bar(ltv7_tbl: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(7.5, 4.5))
    x = ltv7_tbl["ab_group"].tolist()
    y = ltv7_tbl["mean_ltv7"].to_numpy()
    plt.bar(x, y)
    plt.ylabel("Mean LTV7 (USD)")
    plt.title("Primary Metric: Mean LTV7 by A/B Group")
    save_fig(outpath)


def plot_retention_bars(r1: pd.DataFrame, r7: pd.DataFrame, outpath: Path) -> None:
    # Merge
    m = r1[["ab_group", "d1_retention"]].merge(r7[["ab_group", "d7_retention"]], on="ab_group", how="inner")
    plt.figure(figsize=(7.5, 4.5))
    idx = np.arange(len(m))
    width = 0.35
    plt.bar(idx - width/2, m["d1_retention"], width, label="D1 retention")
    plt.bar(idx + width/2, m["d7_retention"], width, label="D7 retention")
    plt.xticks(idx, m["ab_group"].tolist())
    plt.ylabel("Retention")
    plt.title("Guardrails: D1 and D7 Retention by A/B Group")
    plt.legend()
    save_fig(outpath)


def plot_economy_timeseries(daily: pd.DataFrame, metric: str, title: str, outpath: Path) -> None:
    plt.figure(figsize=(8.5, 4.8))
    for g in ["control", "treatment"]:
        sub = daily[daily["ab_group"] == g].copy()
        if len(sub) == 0:
            continue
        plt.plot(pd.to_datetime(sub["event_date"]), sub[metric], marker="o", label=g)
    plt.xlabel("Date")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    save_fig(outpath)


# -----------------------------
# Report writer
# -----------------------------

def write_report(
    outdir: Path,
    ltv7_tbl: pd.DataFrame,
    r1: pd.DataFrame,
    r7: pd.DataFrame,
    econ_daily: pd.DataFrame
) -> Path:
    figs = outdir / "figures"
    ensure_dir(figs)

    # Save tables
    ltv7_tbl.to_csv(outdir / "ltv7_by_group.csv", index=False)
    r1.to_csv(outdir / "d1_retention_by_group.csv", index=False)
    r7.to_csv(outdir / "d7_retention_by_group.csv", index=False)
    econ_daily.to_csv(outdir / "economy_daily_by_group.csv", index=False)

    # Figures
    plot_ltv7_bar(ltv7_tbl, figs / "primary_ltv7_bar.png")
    plot_retention_bars(r1.rename(columns={"d1_retention": "d1_retention"}),
                        r7.rename(columns={"d7_retention": "d7_retention"}),
                        figs / "guardrail_retention_bars.png")
    plot_economy_timeseries(econ_daily, "avg_end_balance",
                            "Guardrail: Economy Inflation Proxy (Average End Balance)",
                            figs / "guardrail_econ_avg_end_balance.png")
    plot_economy_timeseries(econ_daily, "avg_net_currency",
                            "Guardrail: Economy Net Flow (Avg Earned − Spent per Active User-Day)",
                            figs / "guardrail_econ_net_flow.png")
    plot_economy_timeseries(econ_daily, "sink_coverage_ratio",
                            "Guardrail: Sink Coverage Ratio (Total Spent / Total Earned)",
                            figs / "guardrail_econ_sink_coverage.png")

    # Prepare markdown tables with formatting
    ltv7_fmt = ltv7_tbl.copy()
    for c in ["mean_ltv7", "median_ltv7", "p90_ltv7"]:
        ltv7_fmt[c] = ltv7_fmt[c].map(fmt_money)

    r1_fmt = r1.copy()
    r1_fmt["d1_retention"] = r1_fmt["d1_retention"].map(fmt_pct)
    r7_fmt = r7.copy()
    r7_fmt["d7_retention"] = r7_fmt["d7_retention"].map(fmt_pct)

    # Simple interpretation helpers (descriptive)
    # (You can tighten thresholds later)
    def delta(a: float, b: float) -> float:
        return a - b

    ltv7_c = float(ltv7_tbl.loc[ltv7_tbl["ab_group"] == "control", "mean_ltv7"].iloc[0])
    ltv7_t = float(ltv7_tbl.loc[ltv7_tbl["ab_group"] == "treatment", "mean_ltv7"].iloc[0])
    ltv7_delta = ltv7_t - ltv7_c
    ltv7_delta_pct = (ltv7_t / ltv7_c - 1.0) if ltv7_c else 0.0

    d1_c = float(r1.loc[r1["ab_group"] == "control", "d1_retention"].iloc[0])
    d1_t = float(r1.loc[r1["ab_group"] == "treatment", "d1_retention"].iloc[0])
    d7_c = float(r7.loc[r7["ab_group"] == "control", "d7_retention"].iloc[0])
    d7_t = float(r7.loc[r7["ab_group"] == "treatment", "d7_retention"].iloc[0])

    # Guardrail check: if retention drops by > 1pp, flag
    d1_drop_pp = 100.0 * (d1_t - d1_c)
    d7_drop_pp = 100.0 * (d7_t - d7_c)

    guardrail_flags = []
    if d1_drop_pp < -1.0:
        guardrail_flags.append(f"D1 retention decreased by {d1_drop_pp:.2f} pp (guardrail risk).")
    if d7_drop_pp < -1.0:
        guardrail_flags.append(f"D7 retention decreased by {d7_drop_pp:.2f} pp (guardrail risk).")
    if not guardrail_flags:
        guardrail_flags.append("No major retention guardrail breaches detected under the current heuristic thresholds.")

    md = []
    md.append("# Primary Metric & Guardrails Report\n\n")
    md.append("This report defines the primary metric and guardrails for an A/B test and summarizes them descriptively.\n\n")

    md.append("## 1. Metric Definitions\n\n")
    md.append("### Primary Metric\n")
    md.append("- **LTV7**: total revenue per user accumulated over day offsets **0 to 7** after install, averaged across users.\n\n")

    md.append("### Guardrails\n")
    md.append("- **D1 retention**: share of users active on day offset 1 after install.\n")
    md.append("- **D7 retention**: share of users active on day offset 7 after install.\n")
    md.append("- **Economy inflation diagnostics**:\n")
    md.append("  - **Average end balance** (inflation proxy): mean currency balance at the end of each active user-day.\n")
    md.append("  - **Average net flow**: mean (currency earned − currency spent) per active user-day.\n")
    md.append("  - **Sink coverage ratio**: total spent / total earned per day (values < 1 imply currency accumulation).\n\n")

    md.append("## 2. Primary Metric Results (LTV7)\n\n")
    md.append(markdown_table(ltv7_fmt, max_rows=10) + "\n\n")
    md.append(f"![Primary LTV7]({Path('figures') / 'primary_ltv7_bar.png'})\n\n")
    md.append(f"**LTV7 delta (Treatment - Control):** {fmt_money(ltv7_delta)} ({fmt_pct(ltv7_delta_pct)})\n\n")

    md.append("## 3. Guardrail Results: Retention\n\n")
    md.append("### D1 retention\n\n")
    md.append(markdown_table(r1_fmt, max_rows=10) + "\n\n")
    md.append("### D7 retention\n\n")
    md.append(markdown_table(r7_fmt, max_rows=10) + "\n\n")
    md.append(f"![Retention guardrails]({Path('figures') / 'guardrail_retention_bars.png'})\n\n")

    md.append("## 4. Guardrail Results: Economy Inflation Diagnostics\n\n")
    md.append("### Average End Balance (Inflation Proxy)\n\n")
    md.append(f"![Avg end balance]({Path('figures') / 'guardrail_econ_avg_end_balance.png'})\n\n")
    md.append("### Average Net Flow (Earned − Spent)\n\n")
    md.append(f"![Net flow]({Path('figures') / 'guardrail_econ_net_flow.png'})\n\n")
    md.append("### Sink Coverage Ratio (Spent / Earned)\n\n")
    md.append(f"![Sink coverage]({Path('figures') / 'guardrail_econ_sink_coverage.png'})\n\n")

    md.append("## 5. Guardrail Check Summary (Heuristic)\n\n")
    for item in guardrail_flags:
        md.append(f"- {item}\n")
    md.append("\n")
    md.append("**Note:** These guardrail checks are descriptive and threshold-based. For production decisions, pair with confidence intervals (bootstrap) and consider longer horizons.\n")

    report_path = outdir / "guardrails_report.md"
    report_path.write_text("".join(md), encoding="utf-8")
    return report_path


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="events.parquet", help="Path to events.parquet")
    parser.add_argument("--outdir", type=str, default="outputs/guardrails_report", help="Output directory")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_events(in_path)
    user_daily = make_user_daily(df)

    # Tables
    ltv7_tbl = compute_ltv7_by_group(df)

    r1 = retention_at(user_daily, 1).rename(columns={"d1_retention": "d1_retention"})
    r7 = retention_at(user_daily, 7).rename(columns={"d7_retention": "d7_retention"})

    # Economy guardrails
    econ_daily = economy_daily_by_group(user_daily)

    report_path = write_report(outdir, ltv7_tbl, r1, r7, econ_daily)

    print(f"[OK] Guardrails report written: {report_path}")
    print(f"[OK] Outputs saved under: {outdir.resolve()}")


if __name__ == "__main__":
    main()

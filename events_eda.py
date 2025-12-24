# events_eda.py
# Report-style EDA for events.parquet
#
# Usage:
#   python events_eda.py --input events.parquet --outdir outputs/eda_report --max_day 14
#
# Outputs:
#   outputs/eda_report/eda_report.md
#   outputs/eda_report/figures/*.png
#   outputs/eda_report/*.csv
#   outputs/eda_report/user_daily.parquet

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0

def fmt_int(n: int) -> str:
    return f"{int(n):,}"

def fmt_money(x: float) -> str:
    return f"${float(x):,.4f}"

def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    out = df.copy()
    if len(out) > max_rows:
        out = out.head(max_rows).copy()
    return out.to_markdown(index=False)


@dataclass
class KeyMetrics:
    n_rows: int
    n_users: int
    n_sessions: int
    start_time: str
    end_time: str
    n_days: int
    total_iap: float
    total_ads: float
    total_revenue: float
    payer_users: int
    payer_conversion: float
    arpu: float
    arppu: float


# -----------------------------
# Load and validate
# -----------------------------

EXPECTED_COLS = [
    "event_time", "user_id", "install_time", "event_name", "session_id",
    "level", "currency_delta", "currency_balance", "iap_usd", "ad_usd",
    "segment", "ab_group", "cohort_day",
]

def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_parquet(path)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing expected columns in parquet:\n  - " + "\n  - ".join(missing)
            + "\n\nIf your file uses different names, rename columns accordingly."
        )

    # types
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["install_time"] = pd.to_datetime(df["install_time"], errors="coerce")
    for c in ["event_name", "segment", "ab_group", "user_id", "session_id"]:
        df[c] = df[c].astype(str)

    # derived
    df["event_date"] = df["event_time"].dt.date
    df["install_date"] = df["install_time"].dt.date
    df["day_offset"] = (pd.to_datetime(df["event_date"]) - pd.to_datetime(df["install_date"])).dt.days

    # quick sanity: no negative offsets (should be rare if timestamps are correct)
    # keep them but report later
    return df


# -----------------------------
# Quality + key metrics
# -----------------------------

def basic_quality(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    miss = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_rate"})
    )

    dup_count = int(df.duplicated().sum())
    dup = pd.DataFrame([{
        "duplicate_rows": dup_count,
        "duplicate_rate": safe_div(dup_count, len(df))
    }])

    dtypes = (
        df.dtypes.astype(str)
        .reset_index()
        .rename(columns={"index": "column", 0: "dtype"})
    )

    # day_offset sanity
    neg_offsets = int((df["day_offset"] < 0).sum())
    offset_tbl = pd.DataFrame([{
        "negative_day_offset_rows": neg_offsets,
        "negative_day_offset_rate": safe_div(neg_offsets, len(df)),
        "min_day_offset": int(df["day_offset"].min()) if pd.notna(df["day_offset"].min()) else None,
        "max_day_offset": int(df["day_offset"].max()) if pd.notna(df["day_offset"].max()) else None,
    }])

    return {"missingness": miss, "duplicates": dup, "dtypes": dtypes, "day_offset_sanity": offset_tbl}


def compute_key_metrics(df: pd.DataFrame) -> KeyMetrics:
    n_rows = int(len(df))
    n_users = int(df["user_id"].nunique())

    # sessions: prefer counting session_start unique session_id
    session_starts = df[df["event_name"] == "session_start"]
    if len(session_starts) > 0:
        n_sessions = int(session_starts["session_id"].nunique())
    else:
        n_sessions = int(df["session_id"].nunique())

    start_time = str(df["event_time"].min())
    end_time = str(df["event_time"].max())
    n_days = int(pd.to_datetime(df["event_date"]).nunique())

    total_iap = float(df["iap_usd"].fillna(0).sum())
    total_ads = float(df["ad_usd"].fillna(0).sum())
    total_revenue = total_iap + total_ads

    payer_users = int(df.loc[df["iap_usd"] > 0, "user_id"].nunique())
    payer_conversion = safe_div(payer_users, n_users)

    arpu = safe_div(total_revenue, n_users)
    arppu = safe_div(total_revenue, payer_users)

    return KeyMetrics(
        n_rows=n_rows,
        n_users=n_users,
        n_sessions=n_sessions,
        start_time=start_time,
        end_time=end_time,
        n_days=n_days,
        total_iap=total_iap,
        total_ads=total_ads,
        total_revenue=total_revenue,
        payer_users=payer_users,
        payer_conversion=payer_conversion,
        arpu=arpu,
        arppu=arppu,
    )


# -----------------------------
# Derived table: user_daily
# -----------------------------

def make_user_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Active definition: session_start preferred; fallback to any event
    session_starts = df[df["event_name"] == "session_start"].copy()
    active_base = session_starts if len(session_starts) > 0 else df.copy()

    daily_active = (
        active_base.groupby(["user_id", "event_date"], as_index=False)
        .agg(dummy=("event_name", "size"))
    )
    daily_active["is_active"] = 1
    daily_active = daily_active.drop(columns=["dummy"])

    # sessions/day
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

    # progression
    daily_levels = (
        df[df["event_name"] == "level_complete"]
        .groupby(["user_id", "event_date"], as_index=False)
        .agg(levels_completed=("event_name", "size"), max_level=("level", "max"))
    )

    # economy: earned/spent/net
    econ_df = df[df["event_name"].isin(["currency_earn", "currency_spend"])].copy()
    daily_econ = (
        econ_df.groupby(["user_id", "event_date"], as_index=False)
        .agg(
            currency_earned=("currency_delta", lambda x: float(x[x > 0].sum()) if len(x) else 0.0),
            currency_spent=("currency_delta", lambda x: float(-x[x < 0].sum()) if len(x) else 0.0),
            net_currency=("currency_delta", lambda x: float(x.sum()) if len(x) else 0.0),
        )
    )

    # end-of-day balance: last event of day per user
    df_sorted = df.sort_values(["user_id", "event_time"])
    eod_balance = (
        df_sorted.groupby(["user_id", "event_date"], as_index=False)
        .tail(1)[["user_id", "event_date", "currency_balance"]]
        .rename(columns={"currency_balance": "end_balance"})
    )

    # revenue
    daily_rev = (
        df.groupby(["user_id", "event_date"], as_index=False)
        .agg(iap_usd=("iap_usd", "sum"), ad_usd=("ad_usd", "sum"))
    )
    daily_rev["total_usd"] = daily_rev["iap_usd"] + daily_rev["ad_usd"]

    # merge all
    out = daily_active.merge(daily_sessions, on=["user_id", "event_date"], how="left")
    out = out.merge(daily_levels, on=["user_id", "event_date"], how="left")
    out = out.merge(daily_econ, on=["user_id", "event_date"], how="left")
    out = out.merge(eod_balance, on=["user_id", "event_date"], how="left")
    out = out.merge(daily_rev, on=["user_id", "event_date"], how="left")

    # fill NAs
    for c in ["sessions", "levels_completed", "max_level", "currency_earned", "currency_spent",
              "net_currency", "end_balance", "iap_usd", "ad_usd", "total_usd"]:
        out[c] = out[c].fillna(0)

    # user static
    user_static = (
        df.groupby("user_id", as_index=False)
        .agg(
            install_date=("install_date", "min"),
            ab_group=("ab_group", "first"),
            segment=("segment", "first"),
        )
    )
    out = out.merge(user_static, on="user_id", how="left")
    out["day_offset"] = (pd.to_datetime(out["event_date"]) - pd.to_datetime(out["install_date"])).dt.days

    return out


# -----------------------------
# Metrics tables
# -----------------------------

def retention_curve(user_daily: pd.DataFrame, max_day: int = 14) -> pd.DataFrame:
    cohorts = user_daily[["user_id", "install_date"]].drop_duplicates()
    cohort_sizes = cohorts.groupby("install_date", as_index=False).agg(installs=("user_id", "nunique"))

    active = user_daily[user_daily["is_active"] == 1].copy()
    active = active[(active["day_offset"] >= 0) & (active["day_offset"] <= max_day)]

    cohort_active = (
        active.groupby(["install_date", "day_offset"], as_index=False)
        .agg(active_users=("user_id", "nunique"))
        .merge(cohort_sizes, on="install_date", how="left")
    )
    cohort_active["retention"] = cohort_active["active_users"] / cohort_active["installs"]

    # overall weighted
    overall = (
        cohort_active.groupby("day_offset", as_index=False)
        .apply(lambda g: pd.Series({
            "active_users": g["active_users"].sum(),
            "installs": g["installs"].sum(),
            "retention": safe_div(g["active_users"].sum(), g["installs"].sum())
        }))
        .reset_index()
    )
    overall["install_date"] = "ALL"

    return pd.concat([cohort_active, overall], ignore_index=True)


def ltv_by_day(user_daily: pd.DataFrame, max_day: int = 14) -> pd.DataFrame:
    cohorts = user_daily[["user_id", "install_date"]].drop_duplicates()
    cohort_sizes = cohorts.groupby("install_date", as_index=False).agg(installs=("user_id", "nunique"))

    tmp = user_daily[(user_daily["day_offset"] >= 0) & (user_daily["day_offset"] <= max_day)].copy()
    rev = tmp.groupby(["install_date", "day_offset"], as_index=False).agg(rev=("total_usd", "sum"))
    rev = rev.merge(cohort_sizes, on="install_date", how="left").sort_values(["install_date", "day_offset"])
    rev["cum_rev"] = rev.groupby("install_date")["rev"].cumsum()
    rev["ltv"] = rev["cum_rev"] / rev["installs"]

    # overall weighted
    overall = (
        rev.groupby("day_offset", as_index=False)
        .apply(lambda g: pd.Series({"rev": g["rev"].sum(), "installs": g["installs"].sum()}))
        .reset_index()
        .sort_values("day_offset")
    )
    overall["cum_rev"] = overall["rev"].cumsum()
    overall["ltv"] = overall["cum_rev"] / overall["installs"].replace(0, np.nan)
    overall["ltv"] = overall["ltv"].fillna(0)
    overall["install_date"] = "ALL"

    return pd.concat([rev, overall], ignore_index=True)


def economy_indices(user_daily: pd.DataFrame) -> pd.DataFrame:
    active = user_daily[user_daily["is_active"] == 1].copy()
    daily = (
        active.groupby("event_date", as_index=False)
        .agg(
            dau=("user_id", "nunique"),
            avg_end_balance=("end_balance", "mean"),
            avg_net_currency=("net_currency", "mean"),
            total_earned=("currency_earned", "sum"),
            total_spent=("currency_spent", "sum"),
        )
        .sort_values("event_date")
    )
    daily["sink_coverage_ratio"] = daily["total_spent"] / daily["total_earned"].replace(0, np.nan)
    daily["sink_coverage_ratio"] = daily["sink_coverage_ratio"].fillna(0)
    return daily


def ab_compare(user_daily: pd.DataFrame) -> pd.DataFrame:
    users = user_daily[["user_id", "ab_group"]].drop_duplicates()
    installs = users.groupby("ab_group", as_index=False).agg(installs=("user_id", "nunique"))

    active = user_daily[user_daily["is_active"] == 1].copy()

    def retention_at(day: int) -> pd.DataFrame:
        act = active[active["day_offset"] == day].groupby("ab_group", as_index=False).agg(active_users=("user_id", "nunique"))
        out = installs.merge(act, on="ab_group", how="left").fillna({"active_users": 0})
        out[f"d{day}_retention"] = out["active_users"] / out["installs"].replace(0, np.nan)
        out[f"d{day}_retention"] = out[f"d{day}_retention"].fillna(0)
        return out[["ab_group", f"d{day}_retention"]]

    r1 = retention_at(1)
    r7 = retention_at(7)

    total_rev = user_daily.groupby("ab_group", as_index=False).agg(total_rev=("total_usd", "sum"))
    payers = user_daily[user_daily["iap_usd"] > 0].groupby("ab_group", as_index=False).agg(payers=("user_id", "nunique"))

    out = installs.merge(total_rev, on="ab_group", how="left").merge(payers, on="ab_group", how="left").fillna({"total_rev": 0, "payers": 0})
    out["payer_conversion"] = out["payers"] / out["installs"].replace(0, np.nan)
    out["payer_conversion"] = out["payer_conversion"].fillna(0)
    out["arpu"] = out["total_rev"] / out["installs"].replace(0, np.nan)
    out["arpu"] = out["arpu"].fillna(0)
    out["arppu"] = out["total_rev"] / out["payers"].replace(0, np.nan)
    out["arppu"] = out["arppu"].fillna(0)

    h = user_daily[(user_daily["day_offset"] >= 0) & (user_daily["day_offset"] <= 7)].copy()
    ltv7 = h.groupby("ab_group", as_index=False).agg(rev_0_7=("total_usd", "sum"))
    out = out.merge(ltv7, on="ab_group", how="left").fillna({"rev_0_7": 0})
    out["ltv7"] = out["rev_0_7"] / out["installs"].replace(0, np.nan)
    out["ltv7"] = out["ltv7"].fillna(0)

    out = out.merge(r1, on="ab_group", how="left").merge(r7, on="ab_group", how="left")

    out = out.rename(columns={"d1_retention": "d1_retention", "d7_retention": "d7_retention"})
    out = out[["ab_group", "installs", "d1_retention", "d7_retention", "payer_conversion", "arpu", "arppu", "ltv7"]]
    return out.sort_values("ab_group")


# -----------------------------
# Plotting
# -----------------------------

def plot_event_mix(df: pd.DataFrame, outdir: Path) -> Path:
    counts = df["event_name"].value_counts().reset_index()
    counts.columns = ["event_name", "count"]
    counts = counts.sort_values("count", ascending=True)

    plt.figure(figsize=(8, 4.5))
    plt.barh(counts["event_name"], counts["count"])
    plt.xlabel("Count")
    plt.title("Event Mix (Counts by event_name)")
    path = outdir / "event_mix.png"
    save_fig(path)
    return path


def plot_retention(ret_df: pd.DataFrame, outdir: Path, max_day: int) -> Path:
    overall = ret_df[ret_df["install_date"] == "ALL"].sort_values("day_offset")

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(overall["day_offset"], overall["retention"], marker="o")
    plt.xlabel("Day Offset from Install")
    plt.ylabel("Retention")
    plt.title(f"Overall Retention Curve (0–{max_day})")
    path = outdir / "retention_curve.png"
    save_fig(path)
    return path


def plot_ltv(ltv_df: pd.DataFrame, outdir: Path, max_day: int) -> Path:
    overall = ltv_df[ltv_df["install_date"] == "ALL"].sort_values("day_offset")

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(overall["day_offset"], overall["ltv"], marker="o")
    plt.xlabel("Day Offset from Install")
    plt.ylabel("Cumulative LTV")
    plt.title(f"Overall LTV Curve (0–{max_day})")
    path = outdir / "ltv_curve.png"
    save_fig(path)
    return path


def plot_economy(daily_econ: pd.DataFrame, outdir: Path) -> Tuple[Path, Path, Path]:
    # Avg end balance
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(pd.to_datetime(daily_econ["event_date"]), daily_econ["avg_end_balance"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Average End Balance (Active Users)")
    plt.title("Economy: Average End Balance (Inflation Proxy)")
    p1 = outdir / "economy_avg_end_balance.png"
    save_fig(p1)

    # Net flow
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(pd.to_datetime(daily_econ["event_date"]), daily_econ["avg_net_currency"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Average Net Currency per Active User-Day")
    plt.title("Economy: Net Currency Flow (Earned − Spent)")
    p2 = outdir / "economy_net_flow.png"
    save_fig(p2)

    # Sink coverage
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(pd.to_datetime(daily_econ["event_date"]), daily_econ["sink_coverage_ratio"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Total Spent / Total Earned")
    plt.title("Economy: Sink Coverage Ratio")
    p3 = outdir / "economy_sink_coverage.png"
    save_fig(p3)

    return p1, p2, p3


# -----------------------------
# Report writer
# -----------------------------

def write_report(
    df: pd.DataFrame,
    user_daily: pd.DataFrame,
    outdir: Path,
    max_day: int
) -> Path:
    figs_dir = outdir / "figures"
    ensure_dir(figs_dir)

    qm = compute_key_metrics(df)
    q = basic_quality(df)
    ret = retention_curve(user_daily, max_day=max_day)
    ltv = ltv_by_day(user_daily, max_day=max_day)
    econ = economy_indices(user_daily)
    ab = ab_compare(user_daily)

    # Save tables
    q["missingness"].to_csv(outdir / "missingness.csv", index=False)
    q["dtypes"].to_csv(outdir / "dtypes.csv", index=False)
    q["duplicates"].to_csv(outdir / "duplicates.csv", index=False)
    q["day_offset_sanity"].to_csv(outdir / "day_offset_sanity.csv", index=False)
    ret.to_csv(outdir / "retention_curve.csv", index=False)
    ltv.to_csv(outdir / "ltv_curve.csv", index=False)
    econ.to_csv(outdir / "economy_daily.csv", index=False)
    ab.to_csv(outdir / "ab_comparison.csv", index=False)

    # Figures
    f_event = plot_event_mix(df, figs_dir)
    f_ret = plot_retention(ret, figs_dir, max_day=max_day)
    f_ltv = plot_ltv(ltv, figs_dir, max_day=max_day)
    f_e1, f_e2, f_e3 = plot_economy(econ, figs_dir)

    # Dist tables (users)
    top_events = df["event_name"].value_counts().reset_index()
    top_events.columns = ["event_name", "count"]
    top_events["share"] = top_events["count"] / top_events["count"].sum()

    seg_dist = df[["user_id", "segment"]].drop_duplicates()["segment"].value_counts().reset_index()
    seg_dist.columns = ["segment", "users"]
    seg_dist["share"] = seg_dist["users"] / seg_dist["users"].sum()

    ab_dist = df[["user_id", "ab_group"]].drop_duplicates()["ab_group"].value_counts().reset_index()
    ab_dist.columns = ["ab_group", "users"]
    ab_dist["share"] = ab_dist["users"] / ab_dist["users"].sum()

    # DAU
    dau = (
        user_daily[user_daily["is_active"] == 1]
        .groupby("event_date", as_index=False)
        .agg(
            dau=("user_id", "nunique"),
            sessions=("sessions", "sum"),
            revenue=("total_usd", "sum"),
        )
        .sort_values("event_date")
    )

    # Format some columns for markdown readability
    seg_tbl = seg_dist.copy()
    seg_tbl["share"] = seg_tbl["share"].map(lambda v: f"{v:.2%}")

    ab_tbl = ab_dist.copy()
    ab_tbl["share"] = ab_tbl["share"].map(lambda v: f"{v:.2%}")

    events_tbl = top_events.copy()
    events_tbl["share"] = events_tbl["share"].map(lambda v: f"{v:.2%}")

    overall_ret = ret[ret["install_date"] == "ALL"][["day_offset", "retention"]].sort_values("day_offset")
    overall_ret["retention"] = overall_ret["retention"].map(lambda v: f"{v:.2%}")

    overall_ltv = ltv[ltv["install_date"] == "ALL"][["day_offset", "ltv"]].sort_values("day_offset")
    overall_ltv["ltv"] = overall_ltv["ltv"].map(fmt_money)

    ab_fmt = ab.copy()
    ab_fmt["d1_retention"] = ab_fmt["d1_retention"].map(lambda v: f"{v:.2%}")
    ab_fmt["d7_retention"] = ab_fmt["d7_retention"].map(lambda v: f"{v:.2%}")
    ab_fmt["payer_conversion"] = ab_fmt["payer_conversion"].map(lambda v: f"{v:.2%}")
    ab_fmt["arpu"] = ab_fmt["arpu"].map(fmt_money)
    ab_fmt["arppu"] = ab_fmt["arppu"].map(fmt_money)
    ab_fmt["ltv7"] = ab_fmt["ltv7"].map(fmt_money)

    report = []
    report.append("# Game Telemetry EDA Report\n\n")
    report.append("This report summarizes exploratory data analysis (EDA) for a synthetic mobile game telemetry dataset.\n\n")

    report.append("## 1. Dataset Overview\n")
    report.append(f"- Rows (events): **{fmt_int(qm.n_rows)}**\n")
    report.append(f"- Unique users: **{fmt_int(qm.n_users)}**\n")
    report.append(f"- Unique sessions: **{fmt_int(qm.n_sessions)}**\n")
    report.append(f"- Time range: **{qm.start_time}** to **{qm.end_time}**\n")
    report.append(f"- Unique event days: **{fmt_int(qm.n_days)}**\n\n")

    report.append("### Revenue Snapshot\n")
    report.append(f"- Total IAP revenue: **{fmt_money(qm.total_iap)}**\n")
    report.append(f"- Total Ad revenue: **{fmt_money(qm.total_ads)}**\n")
    report.append(f"- Total revenue: **{fmt_money(qm.total_revenue)}**\n")
    report.append(f"- Payer users: **{fmt_int(qm.payer_users)}**\n")
    report.append(f"- Payer conversion: **{qm.payer_conversion:.2%}**\n")
    report.append(f"- ARPU: **{fmt_money(qm.arpu)}**\n")
    report.append(f"- ARPPU: **{fmt_money(qm.arppu)}**\n\n")

    report.append("## 2. Data Quality Checks\n")
    report.append("### Column Types\n")
    report.append(markdown_table(q["dtypes"], max_rows=60) + "\n\n")

    report.append("### Missingness\n")
    report.append("Missingness is expected for optional fields (e.g., `level` for non-level events).\n\n")
    report.append(markdown_table(q["missingness"], max_rows=60) + "\n\n")

    report.append("### Duplicates and Day Offset Sanity\n")
    report.append(markdown_table(q["duplicates"], max_rows=10) + "\n\n")
    report.append(markdown_table(q["day_offset_sanity"], max_rows=10) + "\n\n")

    report.append("## 3. Event Composition\n")
    report.append(markdown_table(events_tbl, max_rows=30) + "\n\n")
    report.append(f"![Event mix]({Path('figures') / f_event.name})\n\n")

    report.append("## 4. User Segments and A/B Groups\n")
    report.append("### Segment Distribution (Users)\n")
    report.append(markdown_table(seg_tbl, max_rows=30) + "\n\n")
    report.append("### A/B Group Distribution (Users)\n")
    report.append(markdown_table(ab_tbl, max_rows=30) + "\n\n")

    report.append("## 5. Engagement: Daily Active Users (DAU)\n")
    report.append("DAU is computed from daily activity (preferably `session_start`).\n\n")
    report.append(markdown_table(dau.head(30), max_rows=30) + "\n\n")

    report.append("## 6. Retention Analysis\n")
    report.append("Retention is the share of users active on a given day offset after install.\n\n")
    report.append(markdown_table(overall_ret, max_rows=50) + "\n\n")
    report.append(f"![Retention curve]({Path('figures') / f_ret.name})\n\n")

    report.append("## 7. Monetization: LTV Curve\n")
    report.append("LTV is cumulative revenue up to each day offset, normalized by installs.\n\n")
    report.append(markdown_table(overall_ltv, max_rows=50) + "\n\n")
    report.append(f"![LTV curve]({Path('figures') / f_ltv.name})\n\n")

    report.append("## 8. Economy Diagnostics\n")
    report.append("We report three simple economy indices:\n")
    report.append("- **Average End Balance** (inflation proxy)\n")
    report.append("- **Average Net Currency Flow** (earned − spent per active user-day)\n")
    report.append("- **Sink Coverage Ratio** (total spent / total earned)\n\n")
    report.append(markdown_table(econ.head(30), max_rows=30) + "\n\n")
    report.append(f"![Avg end balance]({Path('figures') / f_e1.name})\n\n")
    report.append(f"![Net flow]({Path('figures') / f_e2.name})\n\n")
    report.append(f"![Sink coverage ratio]({Path('figures') / f_e3.name})\n\n")

    report.append("## 9. A/B Comparison (Control vs Treatment)\n")
    report.append("High-level metric comparison by A/B group (descriptive, no confidence intervals yet).\n\n")
    report.append(markdown_table(ab_fmt, max_rows=10) + "\n\n")

    report.append("## 10. Notes and Next Steps\n")
    report.append("- If **average end balance** grows steadily while retention is stable, the economy may be too generous (insufficient sinks).\n")
    report.append("- If **net currency** is strongly negative and retention drops, the economy may be too punishing (excessive sinks).\n")
    report.append("- Next step: add statistical uncertainty for A/B deltas (bootstrap confidence intervals).\n")

    report_path = outdir / "eda_report.md"
    report_path.write_text("".join(report), encoding="utf-8")
    return report_path


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="events.parquet", help="Path to events.parquet")
    parser.add_argument("--outdir", type=str, default="outputs/eda_report", help="Output directory")
    parser.add_argument("--max_day", type=int, default=14, help="Max day offset for retention/LTV curves")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_events(in_path)
    user_daily = make_user_daily(df)

    # Save derived table for reuse
    user_daily.to_parquet(outdir / "user_daily.parquet", index=False)

    report_path = write_report(df, user_daily, outdir, max_day=args.max_day)
    print(f"[OK] Report written: {report_path}")
    print(f"[OK] Outputs saved under: {outdir.resolve()}")

if __name__ == "__main__":
    main()

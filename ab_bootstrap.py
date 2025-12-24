# ab_bootstrap_report.py
# Bootstrap 95% CI for A/B deltas (Treatment - Control) on ARPU and LTV7.
#
# Usage:
#   python ab_bootstrap_report.py --input events.parquet --outdir outputs/ab_bootstrap_report --B 3000 --seed 42
#
# Notes:
# - Bootstrapping is done at the USER level (resampling users with replacement within each group).
# - ARPU here = mean total revenue per user (IAP + Ads) over the full dataset window.
# - LTV7 here = mean revenue per user accumulated from day_offset 0..7.

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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fmt_money(x: float) -> str:
    return f"${float(x):,.4f}"


def fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


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

    # Revenue per event
    df["iap_usd"] = pd.to_numeric(df["iap_usd"], errors="coerce").fillna(0.0)
    df["ad_usd"] = pd.to_numeric(df["ad_usd"], errors="coerce").fillna(0.0)
    df["revenue_usd"] = df["iap_usd"] + df["ad_usd"]

    df["ab_group"] = df["ab_group"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    return df


def build_user_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # One row per user with:
    # - ab_group
    # - total_rev (full window)
    # - rev_0_7 (LTV7 numerator per user)
    user_static = df.groupby("user_id", as_index=False).agg(ab_group=("ab_group", "first"))

    total_rev = df.groupby("user_id", as_index=False).agg(total_rev=("revenue_usd", "sum"))

    w = df[(df["day_offset"] >= 0) & (df["day_offset"] <= 7)].copy()
    rev_0_7 = w.groupby("user_id", as_index=False).agg(rev_0_7=("revenue_usd", "sum"))

    out = user_static.merge(total_rev, on="user_id", how="left").merge(rev_0_7, on="user_id", how="left")
    out["total_rev"] = out["total_rev"].fillna(0.0)
    out["rev_0_7"] = out["rev_0_7"].fillna(0.0)
    return out


def point_estimates(user_metrics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    # returns metrics by group and deltas
    groups = {}
    for g in ["control", "treatment"]:
        arr_total = user_metrics.loc[user_metrics["ab_group"] == g, "total_rev"].to_numpy()
        arr_ltv7 = user_metrics.loc[user_metrics["ab_group"] == g, "rev_0_7"].to_numpy()

        groups[g] = {
            "n_users": float(len(arr_total)),
            "arpu": float(arr_total.mean()) if len(arr_total) else 0.0,
            "ltv7": float(arr_ltv7.mean()) if len(arr_ltv7) else 0.0,
        }

    delta = {
        "arpu_delta": groups["treatment"]["arpu"] - groups["control"]["arpu"],
        "ltv7_delta": groups["treatment"]["ltv7"] - groups["control"]["ltv7"],
        "arpu_delta_pct": (groups["treatment"]["arpu"] / groups["control"]["arpu"] - 1.0) if groups["control"]["arpu"] else 0.0,
        "ltv7_delta_pct": (groups["treatment"]["ltv7"] / groups["control"]["ltv7"] - 1.0) if groups["control"]["ltv7"] else 0.0,
    }
    return {"groups": groups, "delta": delta}


def bootstrap_delta(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    B: int,
    seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_c = len(control_values)
    n_t = len(treatment_values)

    # Pre-allocate
    deltas = np.empty(B, dtype=float)

    # Bootstrap: resample users WITHIN each group
    for b in range(B):
        c_sample = control_values[rng.integers(0, n_c, size=n_c)]
        t_sample = treatment_values[rng.integers(0, n_t, size=n_t)]
        deltas[b] = t_sample.mean() - c_sample.mean()

    return deltas


def plot_delta_distribution(deltas: np.ndarray, ci: Tuple[float, float], point: float, title: str, outpath: Path) -> None:
    plt.figure(figsize=(7.5, 4.5))
    plt.hist(deltas, bins=40)
    plt.axvline(point, linestyle="--")
    plt.axvline(ci[0], linestyle="--")
    plt.axvline(ci[1], linestyle="--")
    plt.xlabel("Delta (Treatment - Control)")
    plt.ylabel("Bootstrap Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_point_with_ci(
    metric_name: str,
    control: float,
    treatment: float,
    delta: float,
    ci: Tuple[float, float],
    outpath: Path
) -> None:
    # Simple bar-like plot using points + error bar for delta
    plt.figure(figsize=(7.5, 4.5))

    # Two points for group metrics
    x = np.array([0, 1])
    y = np.array([control, treatment])
    plt.plot(x, y, marker="o", linestyle="-")
    plt.xticks([0, 1], ["Control", "Treatment"])
    plt.ylabel(metric_name)
    plt.title(f"{metric_name}: Group Means + Delta CI")

    # Delta CI in an inset-like line at x=1.6
    # We'll draw a vertical line representing CI and a point for delta
    x_delta = 1.6
    plt.plot([x_delta, x_delta], [ci[0], ci[1]], linestyle="-")
    plt.plot([x_delta], [delta], marker="o")
    plt.xticks([0, 1, x_delta], ["Control", "Treatment", "Δ (T-C)"])

    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def make_markdown_report(
    outdir: Path,
    B: int,
    seed: int,
    pe: Dict[str, Dict[str, float]],
    arpu_ci: Tuple[float, float],
    ltv7_ci: Tuple[float, float]
) -> Path:
    figs_dir = outdir / "figures"

    c = pe["groups"]["control"]
    t = pe["groups"]["treatment"]
    d = pe["delta"]

    # Decision heuristics (simple)
    arpu_inconclusive = (arpu_ci[0] <= 0.0 <= arpu_ci[1])
    ltv7_inconclusive = (ltv7_ci[0] <= 0.0 <= ltv7_ci[1])

    if (not ltv7_inconclusive) and (d["ltv7_delta"] > 0):
        decision = "Ship (treatment increases LTV7 with CI excluding 0)."
    elif (not ltv7_inconclusive) and (d["ltv7_delta"] < 0):
        decision = "Do not ship (treatment decreases LTV7 with CI excluding 0)."
    else:
        decision = "Inconclusive (LTV7 CI includes 0). Consider more data or a longer test."

    md = []
    md.append("# A/B Bootstrap Report (ARPU & LTV7)\n\n")
    md.append("This report estimates uncertainty for A/B deltas using nonparametric bootstrap confidence intervals (95%).\n\n")

    md.append("## 1. Setup\n")
    md.append(f"- Bootstrap iterations (B): **{B}**\n")
    md.append(f"- Random seed: **{seed}**\n")
    md.append("- Unit of resampling: **users** (resample users with replacement within each A/B group)\n\n")

    md.append("## 2. Point Estimates\n")
    md.append("| Metric | Control | Treatment | Delta (T-C) | Delta % |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    md.append(f"| ARPU | {fmt_money(c['arpu'])} | {fmt_money(t['arpu'])} | {fmt_money(d['arpu_delta'])} | {fmt_pct(d['arpu_delta_pct'])} |\n")
    md.append(f"| LTV7 | {fmt_money(c['ltv7'])} | {fmt_money(t['ltv7'])} | {fmt_money(d['ltv7_delta'])} | {fmt_pct(d['ltv7_delta_pct'])} |\n\n")

    md.append("## 3. Bootstrap 95% Confidence Intervals for Deltas\n")
    md.append("| Delta | 95% CI (Lower) | 95% CI (Upper) | Includes 0? |\n")
    md.append("|---|---:|---:|:---:|\n")
    md.append(f"| ARPU Δ | {fmt_money(arpu_ci[0])} | {fmt_money(arpu_ci[1])} | {'Yes' if arpu_inconclusive else 'No'} |\n")
    md.append(f"| LTV7 Δ | {fmt_money(ltv7_ci[0])} | {fmt_money(ltv7_ci[1])} | {'Yes' if ltv7_inconclusive else 'No'} |\n\n")

    md.append("## 4. Visual Diagnostics\n")
    md.append("### ARPU Delta Bootstrap Distribution\n")
    md.append(f"![ARPU delta bootstrap]({Path('figures') / 'arpu_delta_bootstrap.png'})\n\n")
    md.append("### LTV7 Delta Bootstrap Distribution\n")
    md.append(f"![LTV7 delta bootstrap]({Path('figures') / 'ltv7_delta_bootstrap.png'})\n\n")

    md.append("### Point Estimates + CI\n")
    md.append(f"![ARPU point+ci]({Path('figures') / 'arpu_point_ci.png'})\n\n")
    md.append(f"![LTV7 point+ci]({Path('figures') / 'ltv7_point_ci.png'})\n\n")

    md.append("## 5. Recommendation (Based on LTV7)\n")
    md.append(f"**Decision:** {decision}\n\n")

    md.append("## 6. Notes\n")
    md.append("- Bootstrap CI reflects sampling uncertainty under the assumption that users are independent.\n")
    md.append("- If you run sequential tests or multiple metrics, consider corrections / sequential methods.\n")
    md.append("- Add guardrails (D1/D7 retention, economy indices) before final product decisions.\n")

    report_path = outdir / "ab_bootstrap_report.md"
    report_path.write_text("".join(md), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="events.parquet", help="Path to events.parquet")
    parser.add_argument("--outdir", type=str, default="outputs/ab_bootstrap_report", help="Output directory")
    parser.add_argument("--B", type=int, default=3000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    figs_dir = outdir / "figures"
    ensure_dir(outdir)
    ensure_dir(figs_dir)

    df = load_events(in_path)
    user_metrics = build_user_level_metrics(df)

    # sanity: ensure both groups exist
    groups_present = set(user_metrics["ab_group"].unique().tolist())
    if not {"control", "treatment"}.issubset(groups_present):
        raise ValueError(f"Expected ab_group to include 'control' and 'treatment'. Found: {groups_present}")

    pe = point_estimates(user_metrics)

    # Arrays for bootstrapping
    control_total = user_metrics.loc[user_metrics["ab_group"] == "control", "total_rev"].to_numpy()
    treat_total = user_metrics.loc[user_metrics["ab_group"] == "treatment", "total_rev"].to_numpy()

    control_ltv7 = user_metrics.loc[user_metrics["ab_group"] == "control", "rev_0_7"].to_numpy()
    treat_ltv7 = user_metrics.loc[user_metrics["ab_group"] == "treatment", "rev_0_7"].to_numpy()

    # Bootstrap deltas
    arpu_deltas = bootstrap_delta(control_total, treat_total, B=args.B, seed=args.seed)
    ltv7_deltas = bootstrap_delta(control_ltv7, treat_ltv7, B=args.B, seed=args.seed + 1)

    arpu_ci = percentile_ci(arpu_deltas, alpha=0.05)
    ltv7_ci = percentile_ci(ltv7_deltas, alpha=0.05)

    # Save delta samples for transparency
    pd.DataFrame({"arpu_delta": arpu_deltas}).to_csv(outdir / "arpu_delta_bootstrap_samples.csv", index=False)
    pd.DataFrame({"ltv7_delta": ltv7_deltas}).to_csv(outdir / "ltv7_delta_bootstrap_samples.csv", index=False)

    # Figures
    plot_delta_distribution(
        deltas=arpu_deltas,
        ci=arpu_ci,
        point=pe["delta"]["arpu_delta"],
        title="Bootstrap Distribution: ARPU Delta (Treatment - Control)",
        outpath=figs_dir / "arpu_delta_bootstrap.png",
    )
    plot_delta_distribution(
        deltas=ltv7_deltas,
        ci=ltv7_ci,
        point=pe["delta"]["ltv7_delta"],
        title="Bootstrap Distribution: LTV7 Delta (Treatment - Control)",
        outpath=figs_dir / "ltv7_delta_bootstrap.png",
    )

    plot_point_with_ci(
        metric_name="ARPU",
        control=pe["groups"]["control"]["arpu"],
        treatment=pe["groups"]["treatment"]["arpu"],
        delta=pe["delta"]["arpu_delta"],
        ci=arpu_ci,
        outpath=figs_dir / "arpu_point_ci.png",
    )
    plot_point_with_ci(
        metric_name="LTV7",
        control=pe["groups"]["control"]["ltv7"],
        treatment=pe["groups"]["treatment"]["ltv7"],
        delta=pe["delta"]["ltv7_delta"],
        ci=ltv7_ci,
        outpath=figs_dir / "ltv7_point_ci.png",
    )

    # Report
    report_path = make_markdown_report(
        outdir=outdir,
        B=args.B,
        seed=args.seed,
        pe=pe,
        arpu_ci=arpu_ci,
        ltv7_ci=ltv7_ci,
    )

    print(f"[OK] Bootstrap report written: {report_path}")
    print(f"[OK] Figures saved under: {figs_dir.resolve()}")


if __name__ == "__main__":
    main()

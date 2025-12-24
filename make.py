from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("\n[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def write_master_report(out_root: Path) -> Path:
    master = out_root / "master_report.md"
    lines = []
    lines.append("# Mobile Game Economy & Monetization Sandbox — Reports\n\n")
    lines.append("This file links to automatically generated reports.\n\n")

    # expected outputs
    eda = out_root / "eda_report" / "eda_report.md"
    boot = out_root / "ab_bootstrap_report" / "ab_bootstrap_report.md"
    guard = out_root / "guardrails_report" / "guardrails_report.md"

    def link(p: Path) -> str:
        # relative link from out_root
        return p.relative_to(out_root).as_posix()

    lines.append("## Reports\n\n")
    lines.append(f"- EDA report: `{link(eda)}`\n")
    lines.append(f"- A/B bootstrap CI report: `{link(boot)}`\n")
    lines.append(f"- Primary metric + guardrails report: `{link(guard)}`\n\n")

    lines.append("## Figures\n\n")
    lines.append("- see each report's `figures/` folder.\n")

    master.write_text("".join(lines), encoding="utf-8")
    return master


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=str, default="events.parquet", help="path to events parquet")
    parser.add_argument("--run-generate", action="store_true", help="run create_dataset.py even if events exists")
    parser.add_argument("--max-day", type=int, default=14, help="max day offset for EDA retention/LTV curves")
    parser.add_argument("--B", type=int, default=3000, help="bootstrap iterations for ab_bootstrap_report.py")
    parser.add_argument("--seed", type=int, default=42, help="random seed for bootstrap report")
    parser.add_argument("--outroot", type=str, default="outputs", help="root output directory")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    py = sys.executable

    events_path = (repo_root / args.events).resolve()
    out_root = (repo_root / args.outroot).resolve()

    # scripts
    gen_script = repo_root / "create_dataset.py"
    eda_script = repo_root / "events_eda.py"
    boot_script = repo_root / "ab_bootstrap.py"
    guard_script = repo_root / "guardrails.py"

    # sanity checks
    for s in [eda_script, boot_script, guard_script]:
        if not s.exists():
            raise FileNotFoundError(f"missing script: {s}")

    # step 0: generate (optional)
    if args.run_generate or (not events_path.exists()):
        if not gen_script.exists():
            raise FileNotFoundError(
                f"events file not found ({events_path}) and generator script missing: {gen_script}"
            )
        run_cmd([py, str(gen_script)], cwd=repo_root)

        # generator likely writes to repo_root/events.parquet
        default_events = (repo_root / "events.parquet").resolve()
        if not default_events.exists():
            raise FileNotFoundError("create_dataset.py finished but events.parquet was not created.")

        # if user requested a different events path, move it
        if default_events != events_path:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            default_events.replace(events_path)

    if not events_path.exists():
        raise FileNotFoundError(f"events parquet not found: {events_path}")

    # step 1: eda report
    eda_out = out_root / "eda_report"
    eda_out.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [py, str(eda_script), "--input", str(events_path), "--outdir", str(eda_out), "--max_day", str(args.max_day)],
        cwd=repo_root,
    )

    # step 2: bootstrap ci report
    boot_out = out_root / "ab_bootstrap_report"
    boot_out.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [py, str(boot_script), "--input", str(events_path), "--outdir", str(boot_out), "--B", str(args.B), "--seed", str(args.seed)],
        cwd=repo_root,
    )

    # step 3: guardrails report
    guard_out = out_root / "guardrails_report"
    guard_out.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [py, str(guard_script), "--input", str(events_path), "--outdir", str(guard_out)],
        cwd=repo_root,
    )

    # step 4: master report
    master = write_master_report(out_root)

    print("\n[done] outputs:")
    print(f"- master: {master}")
    print(f"- eda:    {eda_out / 'eda_report.md'}")
    print(f"- boot:   {boot_out / 'ab_bootstrap_report.md'}")
    print(f"- guard:  {guard_out / 'guardrails_report.md'}")


if __name__ == "__main__":
    main()

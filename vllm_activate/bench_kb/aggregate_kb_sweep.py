#!/usr/bin/env python3
"""KB sweep 결과 (JSON 여러개) → 표 1장."""
import json, glob, sys, argparse
from pathlib import Path

def main(out_dir, fmt):
    files = sorted(glob.glob(f"{out_dir}/*.json"))
    if not files:
        print(f"No JSON in {out_dir}"); sys.exit(1)
    rows = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr); continue
        cfg = d.get("config", {})
        s = d.get("summary", {}) or {}
        steps = s.get("steps", {}) or {}
        turn = s.get("turn", {}) or {}
        thr = s.get("throughput", {}) or {}

        # step별 TTFT/TPOT 평균 (3 step 평균)
        if steps:
            ttft_p95 = sum(st.get("ttft_p95", 0) for st in steps.values()) / len(steps)
            ttft_mean = sum(st.get("ttft_mean", 0) for st in steps.values()) / len(steps)
            tpot_mean = sum(st.get("tpot_mean", 0) for st in steps.values()) / len(steps)
        else:
            ttft_p95 = ttft_mean = tpot_mean = 0

        rows.append({
            "mode": cfg.get("mode", "?"),
            "pattern": cfg.get("pattern", "?"),
            "ch": cfg.get("channels", 0),
            "ttft_mean": ttft_mean,
            "ttft_p95": ttft_p95,
            "turn_mean": turn.get("mean", 0),
            "turn_p95": turn.get("p95", 0),
            "turn_p99": turn.get("p99", 0),
            "tpot_mean": tpot_mean,
            "tps_in": thr.get("tps_in", 0),
            "tps_out": thr.get("tps_out", 0),
            "calls_s": thr.get("calls_per_s", 0),
            "err_pct": s.get("err_rate", 0),
        })

    rows.sort(key=lambda r: (r["mode"], r["pattern"], r["ch"]))

    if fmt == "csv":
        import csv
        w = csv.writer(sys.stdout)
        w.writerow(["mode","pattern","ch","ttft_mean_ms","ttft_p95_ms","turn_mean_ms","turn_p95_ms","turn_p99_ms","tpot_mean_ms","tps_in","tps_out","calls_s","err_pct"])
        for r in rows:
            w.writerow([r["mode"], r["pattern"], r["ch"],
                        f"{r[\"ttft_mean\"]:.0f}", f"{r[\"ttft_p95\"]:.0f}",
                        f"{r[\"turn_mean\"]:.0f}", f"{r[\"turn_p95\"]:.0f}", f"{r[\"turn_p99\"]:.0f}",
                        f"{r[\"tpot_mean\"]:.1f}",
                        f"{r[\"tps_in\"]:.0f}", f"{r[\"tps_out\"]:.1f}",
                        f"{r[\"calls_s\"]:.2f}", f"{r[\"err_pct\"]:.1f}"])
        return

    # md table
    H = ["mode","pat","ch","TTFT μ","TTFT p95","Turn μ","Turn p95","Turn p99","TPOT μ","TPS_in","TPS_out","call/s","err%"]
    print("| " + " | ".join(H) + " |")
    print("|" + "|".join("---" for _ in H) + "|")
    for r in rows:
        print(f"| {r[\"mode\"]:<10} | {r[\"pattern\"]:<7} | {r[\"ch\"]:>3} | "
              f"{r[\"ttft_mean\"]:>6.0f} | {r[\"ttft_p95\"]:>7.0f} | "
              f"{r[\"turn_mean\"]:>6.0f} | {r[\"turn_p95\"]:>7.0f} | {r[\"turn_p99\"]:>7.0f} | "
              f"{r[\"tpot_mean\"]:>5.1f} | {r[\"tps_in\"]:>6.0f} | {r[\"tps_out\"]:>6.1f} | "
              f"{r[\"calls_s\"]:>5.2f} | {r[\"err_pct\"]:>4.1f} |")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="bench 결과 디렉터리")
    ap.add_argument("--csv", action="store_true", help="CSV 출력 (default markdown)")
    args = ap.parse_args()
    main(args.dir, "csv" if args.csv else "md")

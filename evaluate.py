#!/usr/bin/env python3
"""SwiftSNP: correctness check against snp-dists 1.2.0 + benchmark.

    python3 evaluate.py              # build, random-input check, benchmark-set check + timing
    python3 evaluate.py --holdout    # the same on the four holdout sets
    python3 evaluate.py --quick      # build + random-input check only

The gate has two parts, and both must pass:
  * Fuzz: fresh random tiny inputs every run (new seed each time, printed), random snp-dists flags, random
    candidate thread counts. The candidate's stdout must equal `snp-dists -q -j 1 <same flags>` byte for byte
    (the top-left corner label may differ). When the reference exits non-zero, the candidate must too.
  * Benchmark sets: candidate stdout for `-q -b -j T` must match the recorded sha256 of the reference output.
Timing: wall clock of the full command (parse + compute + write), best of 3 when a run takes < 5 s.
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gen_data  # noqa: E402

MANIFEST = ROOT / "bench" / "manifest.json"
REF = ROOT / "ext" / "snp-dists" / "snp-dists"
CAND = ROOT / "solver" / "swiftsnp"
FAIL_DIR = ROOT / "failures"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise(out: bytes, flags):
    """Drop the top-left corner label (tool name/version) of matrix output unless -b was given."""
    if "-m" in flags or "-b" in flags:
        return out
    sep = b"," if "-c" in flags else b"\t"
    nl = out.find(b"\n")
    i = out.find(sep)
    if i == -1 or (nl != -1 and i > nl):
        return out[nl:] if nl != -1 else b""
    return out[i:]


def run(cmd, cwd, timeout):
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", b"", b""


def fuzz_gate(n_cases, seed, max_fail=5):
    rng = random.Random(seed)
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        for k in range(n_cases):
            data, suffix, flags = gen_data.fuzz_case(rng)
            fname = f"in_{rng.randrange(16**8):08x}{suffix}"
            p = Path(td) / fname
            p.write_bytes(data)
            threads = rng.choice([1, 2, 3, 4, 7])
            rc_r, out_r, _ = run([str(REF), "-q", *flags, fname], td, 20)
            rc_c, out_c, err_c = run([str(CAND), "-q", "-j", str(threads), *flags, fname], td, 20)
            if rc_r != 0:
                ok = rc_c != 0 and rc_c != "timeout"
            else:
                ok = rc_c == 0 and normalise(out_c, flags) == normalise(out_r, flags)
            if not ok:
                fails += 1
                FAIL_DIR.mkdir(exist_ok=True)
                tag = f"fuzz_{seed}_{k}"
                (FAIL_DIR / (tag + suffix)).write_bytes(data)
                (FAIL_DIR / (tag + ".json")).write_text(json.dumps(dict(
                    flags=flags, threads=threads, ref_rc=rc_r, cand_rc=str(rc_c),
                    ref_stdout_head=out_r[:600].decode(errors="replace"),
                    cand_stdout_head=out_c[:600].decode(errors="replace"),
                    cand_stderr_head=err_c[:600].decode(errors="replace")), indent=1))
                print(f"FUZZ FAIL case {k}: flags={flags} -j{threads} ref_rc={rc_r} cand_rc={rc_c} -> failures/{tag}.*")
                if fails >= max_fail:
                    break
    return fails


def bench(rows, threads):
    results, fails = [], 0
    with tempfile.TemporaryDirectory() as td:
        for row in rows:
            src = ROOT / row["path"]
            link = Path(td) / (f"x{random.randrange(16**8):08x}" + "".join(src.suffixes))
            os.symlink(src, link)
            out = Path(td) / "out.tsv"
            timeout = max(60.0, 5.0 * row["ref_j1_s"])
            times, ok = [], True
            for rep in range(3):
                t0 = time.perf_counter()
                with open(out, "wb") as f:
                    try:
                        r = subprocess.run([str(CAND), "-q", "-b", "-j", str(threads), link.name], cwd=td,
                                           stdout=f, stderr=subprocess.PIPE, timeout=timeout)
                        rc = r.returncode
                    except subprocess.TimeoutExpired:
                        rc = "timeout"
                dt = time.perf_counter() - t0
                if rc != 0 or out.stat().st_size != row["expected_bytes"] or sha256_file(out) != row["expected_sha256"]:
                    ok = False
                    print(f"BENCH FAIL {row['name']}: rc={rc} bytes={out.stat().st_size} want={row['expected_bytes']}")
                    break
                times.append(dt)
                if dt >= 5.0:
                    break
            if not ok:
                fails += 1
                results.append(dict(name=row["name"], ok=False))
                continue
            best = min(times)
            results.append(dict(name=row["name"], ok=True, seconds=round(best, 6),
                                speedup=round(row["ref_j1_s"] / best, 3)))
    return results, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fuzz", type=int, default=300)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    bad = [f for f, h in manifest["harness_sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        print(f"HARNESS MODIFIED: {bad}. Restore them (git checkout) before evaluating.")
        sys.exit(2)

    t_start = time.time()
    CAND.unlink(missing_ok=True)
    b = subprocess.run(["bash", str(ROOT / "solver" / "build.sh")], cwd=ROOT, capture_output=True, text=True,
                       timeout=600)
    if b.returncode != 0 or not CAND.exists():
        print(b.stdout[-3000:], b.stderr[-3000:])
        print("---\ngate:             BUILD_FAIL")
        sys.exit(1)

    threads = manifest["threads"]
    seed = random.SystemRandom().randrange(10**9)
    fuzz_fail = fuzz_gate(args.fuzz, seed)
    summary = dict(mode="quick" if args.quick else ("holdout" if args.holdout else "dev"),
                   fuzz_seed=seed, fuzz_cases=args.fuzz, fuzz_failures=fuzz_fail, threads=threads)
    if not args.quick:
        rows = manifest["holdout" if args.holdout else "dev"]
        results, bench_fail = bench(rows, threads) if fuzz_fail == 0 else ([], -1)
        summary.update(bench_failures=bench_fail, per_dataset=results)
        if fuzz_fail == 0 and bench_fail == 0:
            gm = math.exp(sum(math.log(r["seconds"]) for r in results) / len(results))
            sp = math.exp(sum(math.log(r["speedup"]) for r in results) / len(results))
            summary.update(geomean_s=round(gm, 6), speedup_vs_ref=round(sp, 3))
    gate = fuzz_fail == 0 and (args.quick or summary.get("bench_failures") == 0)
    summary.update(gate="PASS" if gate else "FAIL", eval_seconds=round(time.time() - t_start, 1))
    (ROOT / ("last_eval_holdout.json" if args.holdout else "last_eval.json")).write_text(json.dumps(summary, indent=1))

    print("---")
    print(f"gate:             {summary['gate']}")
    print(f"mode:             {summary['mode']}")
    print(f"fuzz:             {args.fuzz} cases, seed {seed}, failures {fuzz_fail}")
    if "geomean_s" in summary:
        print(f"geomean_s:        {summary['geomean_s']:.6f}")
        print(f"speedup_vs_ref:   {summary['speedup_vs_ref']:.3f}")
        print("per_dataset:      " + "  ".join(f"{r['name']}={r['seconds']:.3f}s({r['speedup']:.1f}x)"
                                               for r in summary["per_dataset"]))
    print(f"threads:          {threads}")
    print(f"eval_seconds:     {summary['eval_seconds']}")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SwiftSNP: one-time setup for evaluate.py.

    python3 prepare.py            # build the reference, generate datasets, record expected outputs + timings

Steps:
  1. Build the reference oracle: snp-dists pinned to commit 3cd3988 (v1.2.0) from GitHub, building zlib from
     GitHub source if the system headers are missing.
  2. Generate the dev and holdout alignments (gen_data.py, fixed seeds).
  3. Run `snp-dists -q -b -j 1` on each dataset; record sha256 of stdout, its size and the wall time.
     Single-threaded is the only trustworthy reference: v1.2.0's multithreaded TSV output interleaves rows
     (prepare.py records that as evidence, ref_jT_matches_j1).
  4. Write bench/manifest.json (expected outputs, timings, environment, script hashes).
Re-running is safe: finished steps are skipped.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "ext"
BENCH = ROOT / "bench"
DATA = BENCH / "data"
SOLVER = ROOT / "solver"
SNPDISTS_COMMIT = "3cd3988"
REF = EXT / "snp-dists" / "snp-dists"
HARNESS_FILES = ["prepare.py", "evaluate.py", "gen_data.py"]


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_numpy():
    try:
        import numpy  # noqa: F401
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy"])
        if r.returncode != 0:  # PEP 668 "externally managed" images
            sh([sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", "numpy"])


def have_zlib_headers():
    test = EXT / "zt.c"
    test.write_text("#include <zlib.h>\nint main(){return zlibVersion()[0]==0;}\n")
    r = subprocess.run(["gcc", str(test), "-lz", "-o", str(EXT / "zt")], capture_output=True)
    return r.returncode == 0


def build_reference():
    EXT.mkdir(exist_ok=True)
    if REF.exists():
        return
    extra_cflags, extra_libs = "", ""
    if not have_zlib_headers():
        print("system zlib headers missing; building zlib from GitHub source")
        if not (EXT / "zlib").exists():
            sh(["git", "clone", "-q", "--depth", "1", "https://github.com/madler/zlib.git", str(EXT / "zlib")])
        sh("./configure --static --prefix=" + str(EXT / "zlib-install") + " >/dev/null && make -s install",
           cwd=EXT / "zlib")
        extra_cflags = f"-I{EXT/'zlib-install'/'include'}"
        extra_libs = f"-L{EXT/'zlib-install'/'lib'}"
    d = EXT / "snp-dists"
    if not d.exists():
        sh(["git", "clone", "-q", "https://github.com/tseemann/snp-dists.git", str(d)])
    sh(["git", "checkout", "-q", SNPDISTS_COMMIT], cwd=d)
    sh(f"gcc -Wall -Wextra -Ofast -std=c99 -fopenmp {extra_cflags} -o snp-dists main.c {extra_libs} -lz -lm",
       cwd=d)
    out = subprocess.run([str(REF), "-v"], capture_output=True, text=True).stdout.strip()
    assert out == "snp-dists 1.2.0", out
    # upstream's own regression tests (single-threaded)
    for args, res in [("good.aln", "good.res"), ("ambig.aln", "ambig.res"), ("lowercase.aln", "lowercase.res"),
                      ("-a ambig.aln", "ambig-a.res"), ("-k lowercase.aln", "lowercase-k.res")]:
        a = args.split()
        got = subprocess.run([str(REF), "-q", "-b", *a[:-1], str(d / "test" / a[-1])],
                             capture_output=True, text=True).stdout
        want = (d / "test" / res).read_text()
        assert [l.split() for l in got.splitlines() if l.strip()] == \
               [l.split() for l in want.splitlines() if l.strip()], f"upstream test {args} failed"
    print("reference snp-dists 1.2.0 built and passes upstream tests")


def time_ref(path, threads, out_path):
    t0 = time.perf_counter()
    with open(out_path, "wb") as f:
        subprocess.run([str(REF), "-q", "-b", "-j", str(threads), str(path)], stdout=f, check=True)
    return time.perf_counter() - t0


def prepare_sets(specs, kind, threads):
    import gen_data  # noqa: E402  (needs numpy)
    rows = []
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = BENCH / "tmp_out.tsv"
    for name, spec in specs.items():
        path = DATA / (name + (".fa.gz" if spec.get("gz") else ".fa"))
        if not path.exists():
            t0 = time.perf_counter()
            shape = gen_data.write_dataset(spec, path)
            print(f"generated {name} {shape} in {time.perf_counter()-t0:.1f}s", flush=True)
        t1 = time_ref(path, 1, tmp)
        digest, size = sha256_file(tmp), tmp.stat().st_size
        with open(tmp, "rb") as f:
            header = f.readline().rstrip(b"\n").split(b"\t")
        row = dict(name=name, kind=kind, path=str(path.relative_to(ROOT)), n=len(header) - 1,
                   expected_sha256=digest, expected_bytes=size, ref_j1_s=round(t1, 4))
        if kind == "dev" and threads > 1:
            tT = time_ref(path, threads, tmp)
            row.update(ref_jT_s=round(tT, 4), ref_jT_matches_j1=(sha256_file(tmp) == digest))
        print(json.dumps(row), flush=True)
        rows.append(row)
    tmp.unlink(missing_ok=True)
    return rows


def main():
    os.chdir(ROOT)
    ensure_numpy()
    sys.path.insert(0, str(ROOT))
    build_reference()
    threads = os.cpu_count() or 1
    manifest_path = BENCH / "manifest.json"
    if manifest_path.exists():
        print("bench/manifest.json exists; delete bench/ to regenerate")
    else:
        import gen_data
        dev = prepare_sets(gen_data.DEV_SPECS, "dev", threads)
        hold = prepare_sets(gen_data.HOLDOUT_SPECS, "holdout", threads)
        cpu = ""
        try:
            cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        except Exception:
            pass
        flags = ""
        try:
            flags = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("flags"))
        except Exception:
            pass
        manifest = dict(
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            reference=dict(tool="snp-dists", version="1.2.0", commit=SNPDISTS_COMMIT, cmd="snp-dists -q -b -j 1"),
            threads=threads, cpu=cpu,
            simd=[f for f in ["sse4_2", "popcnt", "avx2", "avx512f", "avx512_vpopcntdq", "bmi2"] if f in flags.split()],
            gcc=subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
            python=platform.python_version(),
            dev=dev, holdout=hold,
            harness_sha256={f: sha256_file(ROOT / f) for f in HARNESS_FILES},
        )
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"wrote {manifest_path}")
    print("prepare: done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SwiftSNP: deterministic test alignments.

Two jobs:
  * write_dataset(spec, path): the large benchmark alignments (dev + holdout), seeded and reproducible.
  * fuzz_case(rng): tiny adversarial FASTA/FASTQ inputs + snp-dists flags for the correctness tests.

Model for the large sets: clonal expansion. Each new isolate copies a random earlier isolate and gains
Poisson(mu) substitutions drawn from a pool of variable sites, which loosely resembles a bacterial outbreak
or core-genome alignment. Missing data (N blocks), gaps, IUPAC codes and soft-masked (lowercase) regions are
layered on top, because real pipeline output (snippy-core, etc.) contains all of them.
"""
import gzip
import random

import numpy as np

ACGT = np.frombuffer(b"ACGT", dtype=np.uint8)
IUPAC = np.frombuffer(b"RYKMSWBDHV", dtype=np.uint8)

# name: (n, L, kwargs). L is the simulated length; for snp_only sets the written length is the number of
# variable columns (capped at L_out).
DEV_SPECS = {
    "outbreak_full":  dict(seed=101, n=200,  L=1_000_000, pool=3_000,   mu=12,  nfrac=0.02, gapfrac=0.005, wrap=60),
    "core_snps":      dict(seed=102, n=1500, L=40_000,    pool=40_000,  mu=40,  nfrac=0.01, gapfrac=0.002, snp_only=8_000, wrap=0),
    "large_n_snps":   dict(seed=103, n=4000, L=10_000,    pool=10_000,  mu=8,   nfrac=0.01, gapfrac=0.0,   snp_only=1_500, wrap=0),
    "diverse_full":   dict(seed=104, n=60,   L=3_000_000, pool=150_000, mu=3_000, nfrac=0.03, gapfrac=0.01, iupac=1e-4, lower=0.3, wrap=80),
    "gz_wrapped":     dict(seed=105, n=300,  L=300_000,   pool=5_000,   mu=25,  nfrac=0.02, gapfrac=0.005, wrap=60, gz=True),
}

# Never used for keep/discard decisions. Different seeds and shapes, checked at checkpoints only.
HOLDOUT_SPECS = {
    "h_mid_full":     dict(seed=201, n=700,  L=200_000,   pool=4_000,   mu=6,   nfrac=0.03, gapfrac=0.01, wrap=70),
    "h_snps_2500":    dict(seed=202, n=2500, L=20_000,    pool=20_000,  mu=15,  nfrac=0.05, gapfrac=0.0,  snp_only=3_000, iupac=5e-4, wrap=0),
    "h_heavy_n":      dict(seed=203, n=100,  L=2_000_000, pool=50_000,  mu=400, nfrac=0.20, gapfrac=0.05, lower=0.5, wrap=0, gz=True),
    "h_identical":    dict(seed=204, n=500,  L=100_000,   pool=50,      mu=0.3, nfrac=0.0,  gapfrac=0.0,  wrap=60),
}


def _mask_blocks(row, frac, char, rng, mean_len=500):
    """Overwrite about frac of the row with runs of `char` (missing coverage / deletions)."""
    L = row.shape[0]
    if frac <= 0 or L == 0:
        return
    target = int(frac * L)
    done = 0
    while done < target:
        blen = int(min(L, max(1, rng.geometric(1.0 / mean_len))))
        start = int(rng.integers(0, max(1, L - blen + 1)))
        row[start:start + blen] = char
        done += blen


def simulate(spec):
    rng = np.random.default_rng(spec["seed"])
    n, L = spec["n"], spec["L"]
    seqs = np.empty((n, L), dtype=np.uint8)
    seqs[0] = ACGT[rng.integers(0, 4, L)]
    pool = rng.choice(L, size=min(spec["pool"], L), replace=False)
    for i in range(1, n):
        seqs[i] = seqs[int(rng.integers(0, i))]
        k = int(rng.poisson(spec["mu"]))
        if k:
            sites = pool[rng.integers(0, len(pool), k)]
            cur = np.searchsorted(ACGT, seqs[i, sites])
            seqs[i, sites] = ACGT[(cur + rng.integers(1, 4, k)) % 4]
    if spec.get("snp_only"):
        variable = np.flatnonzero((seqs != seqs[0]).any(axis=0))
        seqs = np.ascontiguousarray(seqs[:, variable[: spec["snp_only"]]])
    for i in range(n):
        row = seqs[i]
        _mask_blocks(row, spec.get("nfrac", 0.0), ord("N"), rng)
        _mask_blocks(row, spec.get("gapfrac", 0.0), ord("-"), rng, mean_len=50)
        if spec.get("iupac"):
            m = int(rng.binomial(row.shape[0], spec["iupac"]))
            if m:
                row[rng.integers(0, row.shape[0], m)] = IUPAC[rng.integers(0, len(IUPAC), m)]
        if spec.get("lower") and rng.random() < spec["lower"]:
            a = int(rng.integers(0, row.shape[0]))
            b = min(row.shape[0], a + int(rng.integers(1, max(2, row.shape[0] // 4))))
            seg = row[a:b]
            letters = (seg >= 65) & (seg <= 90)
            seg[letters] += 32
    return seqs


def write_dataset(spec, path):
    seqs = simulate(spec)
    wrap = spec.get("wrap", 0)
    opener = (lambda p: gzip.open(p, "wb", compresslevel=1)) if spec.get("gz") else (lambda p: open(p, "wb"))
    rng = random.Random(spec["seed"])
    with opener(path) as f:
        for i in range(seqs.shape[0]):
            desc = f" sample={rng.randrange(10**6)}" if rng.random() < 0.2 else ""
            f.write(f">iso{i:05d}{desc}\n".encode())
            s = seqs[i].tobytes()
            if wrap:
                f.write(b"\n".join(s[j:j + wrap] for j in range(0, len(s), wrap)) + b"\n")
            else:
                f.write(s + b"\n")
    return seqs.shape


# ---------------------------------------------------------------- fuzz cases for the correctness gate

_FUZZ_CHARS = "AAAACCCCGGGGTTTTacgtNNn--.RYKMSWBDHVrykx?*"


def fuzz_case(rng: random.Random):
    """Return (file_bytes, filename_suffix, flags). Small, adversarial, but mostly valid alignments."""
    n = rng.choice([1, 2, 2, 3, 4, 5, 7, 12, 25, 40])
    L = rng.choice([0, 1, 2, 3, 5, 8, 16, 31, 63, 64, 65, 100, 255, 256, 257, 1000])
    style = rng.random()
    base = [rng.choice("ACGT") for _ in range(L)]
    seqs = []
    for _ in range(n):
        if style < 0.15:  # fully random
            s = [rng.choice(_FUZZ_CHARS) for _ in range(L)]
        else:  # mutate the shared base so distances are small and varied
            s = base[:]
            for _ in range(rng.randrange(0, max(1, L // 3) + 1)):
                if L:
                    s[rng.randrange(L)] = rng.choice(_FUZZ_CHARS)
        seqs.append("".join(s))
    if rng.random() < 0.03 and n >= 2 and L >= 1:  # unequal lengths: reference must error, so must candidate
        k = rng.randrange(1, n)
        seqs[k] = seqs[k][:-1] if rng.random() < 0.5 else seqs[k] + "A"
    names = []
    for i in range(n):
        r = rng.random()
        if r < 0.02:
            nm = ""
        elif r < 0.05 and names:
            nm = rng.choice(names)  # duplicate names are allowed
        else:
            nm = rng.choice(["s", "iso_", "Sample-", "x.y|"]) + str(rng.randrange(10**rng.randrange(1, 7)))
        names.append(nm)
    fastq = rng.random() < 0.03
    wrap = rng.choice([0, 0, 0, 1, 7, 60])
    blank = rng.random() < 0.1
    crlf = rng.random() < 0.05
    out = []
    for nm, s in zip(names, seqs):
        desc = (rng.choice([" ", "\t"]) + "desc " + str(rng.randrange(100))) if rng.random() < 0.2 else ""
        if fastq:
            out.append(f"@{nm}{desc}\n{s}\n+\n{'I' * len(s)}\n")
            continue
        body = s if not wrap or not s else "\n".join(s[j:j + wrap] for j in range(0, len(s), wrap))
        out.append(f">{nm}{desc}\n{body}\n" + ("\n" if blank and rng.random() < 0.5 else ""))
    text = "".join(out)
    if rng.random() < 0.05:
        text = text.rstrip("\n")  # no trailing newline
    if crlf:
        text = text.replace("\n", "\r\n")
    data = text.encode()
    suffix = ".fq" if fastq else ".fa"
    if rng.random() < 0.1:
        data = gzip.compress(data, compresslevel=1)
        suffix += ".gz"
    flags = []
    for f in ["-a", "-k", "-c", "-b", "-L", "-m", "-t"]:
        if rng.random() < 0.25:
            flags.append(f)
    if rng.random() < 0.15:
        flags += ["-x", str(rng.randrange(1, 6))]
    rng.shuffle(flags)
    return data, suffix, flags

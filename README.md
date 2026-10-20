# swiftsnp

swiftsnp is a drop-in replacement for [snp-dists](https://github.com/tseemann/snp-dists) 1.2.0. It computes the same
pairwise SNP distance matrix from a FASTA alignment, byte for byte, much faster.

On five published alignments (*Vibrio cholerae*, *Mycobacterium tuberculosis*, SARS-CoV-2; 100 to 8,596 genomes) it was
336× faster than the fastest other tool that writes the same matrix, and 2,894× faster than `snp-dists -j 1`
(geometric means, 2 threads, AVX-512 CPU). On 8,596 SARS-CoV-2 genomes snp-dists took 27 minutes and swiftsnp 0.23 s.
Without AVX-512 the gain is smaller: 53× at one thread on an AVX2 laptop. Details are in the preprint (see Citation).

## Build

Requirements: Linux on x86-64, gcc with OpenMP, zlib headers (`zlib1g-dev` on Debian/Ubuntu). Tested with gcc 13.3.0
on Ubuntu 24.04.

```
bash solver/build.sh        # produces solver/swiftsnp
```

The build uses `-march=native`, so the binary is tuned for the build machine's CPU. The fastest code paths need
AVX-512 BW; on other x86-64 CPUs portable paths give the same output, more slowly.

## Usage

swiftsnp takes the same options as snp-dists 1.2.0 and writes the same output:

```
solver/swiftsnp [opts] aligned.fasta[.gz] > matrix.tsv
  -j INT   threads (default 1)
  -q       quiet          -a  count all differences, not just ACGT
  -k       keep case      -m  molten output       -t  header for molten output
  -L       lower triangle (including the diagonal)
  -c       comma separator  -b  blank top-left corner cell
  -x INT   cap distances at INT (default 9999999)
  -v       print version
```

- Output is byte-identical to `snp-dists -q -j 1` with the same options, at every `-j`. (snp-dists 1.2.0 writes
  corrupt output with `-j` > 1; swiftsnp does not.)
- swiftsnp prints the snp-dists corner label and version string, so its output matches exactly.
- Input: plain or gzip-compressed FASTA or FASTQ.
- `-a` is supported but uses a slower generic path.

## Check it yourself

```
python3 prepare.py              # builds snp-dists 1.2.0 (commit 3cd3988), generates test alignments,
                                # records snp-dists' outputs (needs gcc, zlib, numpy, network; 5-15 min)
bash solver/build.sh
python3 evaluate.py             # 300 random inputs and five benchmark sets compared byte for byte, plus timing
python3 evaluate.py --holdout   # the four holdout sets
```

`data/` holds the exact SARS-CoV-2 sets used in the preprint; `bash data/fetch_real_data.sh` downloads the other
published alignments (see `data/README.txt`).

## Licence

GPL-3.0 (`LICENSE`), as a derivative of snp-dists. Vendored code: libdeflate (MIT) and kseq.h (MIT); see `LICENSES/`.

## Citation

Christensen J. Claude Opus 5.5 for bioinformatics: SwiftSNP, a byte-identical snp-dists that is 336-fold faster than
the fastest equivalent tool. Preprint, 2026. Please also cite snp-dists (Seemann T, Klötzl F, Page AJ;
https://github.com/tseemann/snp-dists). See `CITATION.cff`.

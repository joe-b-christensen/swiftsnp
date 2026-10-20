#!/bin/bash
# Download the five published alignments used in the SwiftSNP preprint into data/real_data/.
#   - V. cholerae and M. tuberculosis: versioned Zenodo records (CC BY 4.0).
#   - SARS-CoV-2: Nextstrain open builds. Nextstrain regenerates these regularly, so a fresh download is a different
#     sequence set from the one in the preprint (downloaded 11 Oct 2026).
#     sarscov2_n8596 is the union of the seven regional builds, keeping the first occurrence of each name.
# Requirements: curl, xz, python3. About 1.5 GB of temporary disk space.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/real_data
mkdir -p "$OUT"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

fetch() { echo "downloading $1"; curl -fL --retry 3 -o "$2" "$1"; }

# 1. Vibrio cholerae O1, Snippy core-genome alignment (Wilkinson, Zenodo, doi:10.5281/zenodo.16909942, CC BY 4.0)
fetch "https://zenodo.org/records/16909942/files/cholera_global_n100_2025.09.01.fasta?download=1" "$T/vc.fa"
xz -9e -T0 -c "$T/vc.fa" > "$OUT/vcholerae_n100.fa.xz"; rm "$T/vc.fa"

# 2. Mycobacterium tuberculosis, 1,788 isolates, variable sites (Mixão et al., Zenodo, doi:10.5281/zenodo.7772652,
#    CC BY 4.0)
fetch "https://zenodo.org/records/7772652/files/MTb_original_align_profile.fasta?download=1" "$T/mtb.fa"
xz -9e -T0 -c "$T/mtb.fa" > "$OUT/mtb_n1788_snps.fa.xz"; rm "$T/mtb.fa"

# 3-5. SARS-CoV-2, Nextstrain open builds (GenBank/INSDC-derived; see DATA_SOURCES.txt)
for r in global europe africa asia north-america oceania south-america; do
  fetch "https://data.nextstrain.org/files/ncov/open/$r/aligned.fasta.xz" "$T/$r.fa.xz"
done
cp "$T/europe.fa.xz" "$OUT/sarscov2_europe_n2012.fa.xz"
cp "$T/global.fa.xz" "$OUT/sarscov2_global_n4637.fa.xz"
python3 - "$T" "$OUT/sarscov2_n8596.fa" << 'PY'
import lzma, sys
src, dst = sys.argv[1], sys.argv[2]
seen, keep, n = set(), False, 0
with open(dst, "w") as out:
    for r in ["global", "europe", "africa", "asia", "north-america", "oceania", "south-america"]:
        with lzma.open(f"{src}/{r}.fa.xz", "rt") as f:
            for line in f:
                if line.startswith(">"):
                    name = line[1:].split()[0] if line[1:].strip() else ""
                    keep = name not in seen
                    if keep:
                        seen.add(name); n += 1
                if keep:
                    out.write(line)
print(f"union: {n} sequences")
PY
xz -9e -T0 "$OUT/sarscov2_n8596.fa"

echo "Done. Files in $OUT:"; ls -l "$OUT"
echo "Note: the SARS-CoV-2 files differ from the sets used in the preprint (see the header)."

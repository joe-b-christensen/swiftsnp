#!/bin/bash
# Builds solver/swiftsnp.
set -euo pipefail
cd "$(dirname "$0")"
LD=libdeflate/lib
OBJ=$(mktemp -d)
trap 'rm -rf "$OBJ"' EXIT
for f in deflate_decompress gzip_decompress utils crc32 x86/cpu_features; do
  gcc -O3 -march=native -std=gnu11 -w -c $LD/$f.c -o "$OBJ/$(basename $f).o" &
done
wait
# static link: saves ~0.3-0.6 ms of dynamic loading per run (falls back to dynamic if static libs are missing)
gcc -Wall -Wextra -O3 -march=native -std=gnu11 -fopenmp -static -o swiftsnp main.c "$OBJ"/*.o -lz -lm 2>/dev/null || \
  gcc -Wall -Wextra -O3 -march=native -std=gnu11 -fopenmp -o swiftsnp main.c "$OBJ"/*.o -lz -lm

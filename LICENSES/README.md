# Licences

swiftsnp is a derivative of snp-dists 1.2.0 (Torsten Seemann, Fabian Klötzl, Andrew J. Page), which is licensed
under the GNU General Public License v3.0. swiftsnp is therefore distributed under the GPL-3.0; the full text is in
`../LICENSE`.

Vendored third-party code (MIT licence, GPL-3.0-compatible):

| component | where | licence text |
|---|---|---|
| libdeflate 1.26 (Eric Biggers; decompression sources only, patched as listed in `solver/LICENSE_NOTE.txt`) | `solver/libdeflate/` | `libdeflate-MIT.txt` (also `solver/libdeflate/COPYING`) |
| kseq.h (Attractive Chaos / Heng Li, klib, 2017-02-11 version, as shipped with snp-dists) | `solver/kseq.h` | `kseq-MIT.txt` (also in the file header) |

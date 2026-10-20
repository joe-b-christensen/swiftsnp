/*
 * swiftsnp — a drop-in, byte-identical, faster replacement for snp-dists 1.2.0
 *
 * Derived from snp-dists (Torsten Seemann, Fabian Klötzl, Andrew J. Page), GPL-3.0.
 * The reference main.c is the specification; every output byte must equal `snp-dists -q -j 1` with the same
 * flags (only the top-left corner label may differ, and here it does not).
 *
 * Pipeline:
 *   1. load   — the whole input in memory (mmap for plain files, gzread for gzip), then a fast FASTA parser that
 *               reproduces kseq's semantics for clean input. Anything unusual (CR, NUL, FASTQ '+' lines,
 *               unequal lengths, no records, read errors) bails out to the reference loader (kseq over gzread,
 *               verbatim), which also produces the reference's errors.
 *   2. scan   — per-column state over all records to find informative columns (two different non-'.' bytes).
 *   3. gather — transformed bytes of the informative columns only.
 *   4. encode — bitplanes (code planes + valid plane), 64 columns per word.
 *   5. kernel — d = popcount(OR_p(a_p ^ b_p) & va & vb), upper triangle once, min(d, maxdiff).
 *   6. output — parallel chunked formatting, ordered writes.
 */
#define _GNU_SOURCE
#include "kseq.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <zlib.h>
#include <omp.h>
#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include <sys/resource.h>
#include <immintrin.h>
#include "libdeflate/libdeflate.h"

#define VERSION "1.2.0"
#define EXENAME "snp-dists"
#define GITHUB_URL "https://github.com/tseemann/snp-dists"

const int MAX_SEQ = 1000000;
const char IGNORE_CHAR = '.';

int cpus = 1;

KSEQ_INIT(gzFile, gzread)

//------------------------------------------------------------------------ profiling (stderr, opt-in)
static int g_prof = 0;
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double g_t0, g_tl, g_twrite;
static void prof(const char* what)
{
  if (!g_prof) return;
  double t = now_s();
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  fprintf(stderr, "[prof] %-10s %8.4f s  (total %8.4f)  minflt %ld\n", what, t - g_tl, t - g_t0, ru.ru_minflt);
  g_tl = t;
}

//------------------------------------------------------------------------ output helpers
static const char DIG2[201] =
  "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
  "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
  "8081828384858687888990919293949596979899";

static inline char* put_u32(char* p, uint32_t v)
{
  if (v < 10) { *p = (char)('0' + v); return p + 1; }
  if (v < 100) { memcpy(p, DIG2 + 2 * v, 2); return p + 2; }
  char tmp[12]; char* e = tmp + 12; char* q = e;
  while (v >= 100) { uint32_t r = v % 100; v /= 100; q -= 2; memcpy(q, DIG2 + 2 * r, 2); }
  if (v < 10) *--q = (char)('0' + v); else { q -= 2; memcpy(q, DIG2 + 2 * v, 2); }
  size_t n = (size_t)(e - q); memcpy(p, q, n); return p + n;
}

static uint64_t SEPTAB[1000];   // "<sep><digits>" in the low bytes, its length in the top byte (one load per value)
static uint64_t PAIRTAB[128 * 128];   // two values < 100 ("<sep>a<sep>b", <= 6 bytes) + length; 0 = not covered

// SIMD formatting of 16-bit distances: for a group of 4 values stored as "<sep>d2d1d0" dwords, the pshufb control
// that keeps <sep>, d2 only if the value is >= 100, d1 only if >= 10, and d0; index = (>= 10 bits) | (>= 100 bits) << 4
static uint8_t FMT_SHUF[256][16] __attribute__((aligned(64)));
static uint8_t FMT_LEN[256];
static void fmt_init(void)
{
  for (int ix = 0; ix < 256; ix++) {
    int p = 0;
    for (int k = 0; k < 4; k++) {
      int c = ((ix >> k) & 1) + ((ix >> (4 + k)) & 1);
      FMT_SHUF[ix][p++] = (uint8_t)(4 * k);
      if (c == 2) FMT_SHUF[ix][p++] = (uint8_t)(4 * k + 1);
      if (c >= 1) FMT_SHUF[ix][p++] = (uint8_t)(4 * k + 2);
      FMT_SHUF[ix][p++] = (uint8_t)(4 * k + 3);
    }
    FMT_LEN[ix] = (uint8_t)p;
    for (; p < 16; p++) FMT_SHUF[ix][p] = 0x80;
  }
}

static void write_all(const char* buf, size_t n)
{
  while (n > 0) {
    ssize_t w = write(1, buf, n);
    if (w < 0) { if (errno == EINTR) continue; perror("write"); exit(EXIT_FAILURE); }
    buf += w; n -= (size_t)w;
  }
}

static void* xmalloc(size_t n) { void* p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); } return p; }
static void* xcalloc(size_t n, size_t s) { void* p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); } return p; }


// Large anonymous buffers: mmap + MADV_HUGEPAGE (THP is in "madvise" mode on this kernel), so a 100 MB buffer
// costs ~50 page faults instead of ~25,000. Below 256 KB, 4 KB pages: a huge page is zeroed in full on first touch
// (~0.1 ms), which dominated small inputs. The memory comes zero-filled; callers do not clear it again.
static size_t hp_round(size_t n) { return (n + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1); }
static void* big_alloc(size_t n)
{
  size_t want = n;
  n = hp_round(n ? n : 1);
  void* p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return NULL;
  if (want >= (256u << 10)) madvise(p, n, MADV_HUGEPAGE);
  return p;
}
static void* big_realloc(void* p, size_t old, size_t n)
{
  void* q = mremap(p, hp_round(old ? old : 1), hp_round(n), MREMAP_MAYMOVE);
  if (q == MAP_FAILED) return NULL;
  madvise(q, hp_round(n), MADV_HUGEPAGE);
  return q;
}

//------------------------------------------------------------------------ records
typedef struct {
  const char* name;            // not NUL-terminated in the fast path
  size_t namelen;              // printed length (reference: strcpy + %s, i.e. up to the first NUL)
  const unsigned char* seq;    // fast path: raw bytes in the input buffer; fallback: transformed row
  size_t W;                    // fast path: line width of a regularly wrapped record (0 = contiguous)
  unsigned char* own;          // owned buffer (materialised irregular record or fallback row)
} rec_t;

static rec_t* R = NULL;
static size_t RN = 0, RM = 0;
static size_t SEQLEN = 0;      // common sequence length L
static int PRE_TRANSFORMED = 0; // 1 when records come from the reference loader (already transformed)

static void push_rec(rec_t r)
{
  if (RN == RM) { RM = RM ? 2 * RM : 1024; R = realloc(R, RM * sizeof(rec_t)); if (!R) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); } }
  R[RN++] = r;
}

static void free_recs(void)
{
  for (size_t i = 0; i < RN; i++) free(R[i].own);
  free(R); R = NULL; RN = RM = 0;
}

//------------------------------------------------------------------------ the reference loader (exact)
// Verbatim semantics of snp-dists 1.2.0: kseq over gzread, strcpy, toupper, clean. Used whenever the fast
// path sees anything it does not model, and it is where every input error is reported.
static void load_reference(const char* fasta, int keepcase, int allchars, int quiet)
{
  gzFile fp = gzopen(fasta, "r");
  if (!fp) {
    fprintf(stderr, "ERROR: Could not open filename '%s'\n", fasta);
    exit(EXIT_FAILURE);
  }
  ssize_t l, N = 0, L = -1;
  kseq_t* kseq = kseq_init(fp);
  while ((l = kseq_read(kseq)) >= 0) {
    if (L < 0) L = l;
    if (l != L) {
      fprintf(stderr, "ERROR: sequence #%zu '%s' has length %zu but expected %zu\n", N + 1, kseq->name.s, l, L);
      exit(EXIT_FAILURE);
    }
    if (N >= MAX_SEQ) {
      fprintf(stderr, "ERROR: %s can only handle %d sequences at most. Please change MAX_SEQ and recompile.\n", EXENAME, MAX_SEQ);
      exit(EXIT_FAILURE);
    }
    char* s = (char*)xcalloc(kseq->seq.l + 1, sizeof(char));
    strcpy(s, kseq->seq.s);
    char* nm = (char*)xcalloc(kseq->name.l + 1, sizeof(char));
    strcpy(nm, kseq->name.s);
    if (!keepcase) for (char* c = s; *c; c++) *c = toupper(*c);
    if (!allchars) for (char* c = s; *c; c++) if (*c != 'A' && *c != 'T' && *c != 'C' && *c != 'G') *c = IGNORE_CHAR;
    rec_t r = { nm, strlen(nm), (const unsigned char*)s, 0, (unsigned char*)s };
    push_rec(r);
    // the name buffer is kept alive for output; it is freed with the process
    N++;
  }
  kseq_destroy(kseq);
  gzclose(fp);
  if (N < 1) {
    fprintf(stderr, "ERROR: file contained no sequences\n");
    exit(EXIT_FAILURE);
  }
  if (!quiet) fprintf(stderr, "Read %zu sequences of length %zu\n", N, L);
  SEQLEN = (size_t)L;
  PRE_TRANSFORMED = 1;
}

//------------------------------------------------------------------------ whole-file input
typedef struct { unsigned char* p; size_t n; int mapped; size_t cap; } inbuf_t;

// Returns 0 on success; nonzero means "let the reference loader handle it" (including open errors).

//------------------------------------------------------------------------ two-thread gzip decoding
// One gzip member is decoded by two threads. Thread 1 looks for a deflate block boundary near the middle of
// the compressed data (dynamic-Huffman header that passes a strict validity check, then one block actually
// decoded), publishes it, and decodes from there to the end with a 32 KB history of zero bytes. Thread 0
// decodes from the start and stops exactly at that boundary; if the boundary is not a real block boundary of
// the stream, thread 0 simply passes it and decodes everything (serial result). Every byte thread 1 produced
// from unknown history is a copy of a zero, so all non-zero bytes are already correct; the prefix up to the
// last zero byte is decoded again with the real history. CRC-32 and ISIZE of the member are verified; any
// failure returns 1 and the caller falls back to the serial decoder (and, on real errors, to zlib/kseq).

static size_t gz_header_len(const unsigned char* z, size_t n)
{
  if (n < 18 || z[0] != 0x1f || z[1] != 0x8b || z[2] != 8) return 0;
  unsigned flg = z[3];
  if (flg & 0xE0) return 0;
  size_t p = 10;
  if (flg & 4) { if (p + 2 > n) return 0; size_t xl = (size_t)z[p] | (size_t)z[p + 1] << 8; p += 2 + xl; }
  if (flg & 8) { while (p < n && z[p]) p++; p++; }
  if (flg & 16) { while (p < n && z[p]) p++; p++; }
  if (flg & 2) p += 2;
  return p < n ? p : 0;
}

static inline uint64_t bits_at(const unsigned char* z, uint64_t bp)
{
  uint64_t w; memcpy(&w, z + (bp >> 3), 8);
  return w >> (bp & 7);   // at least 57 valid bits
}

// Kraft sum of code lengths scaled to 2^15; returns -1 if over-subscribed
static long kraft15(const unsigned char* len, int n)
{
  long sum = 0;
  for (int i = 0; i < n; i++) if (len[i]) sum += 1L << (15 - len[i]);
  return sum > (1L << 15) ? -1 : sum;
}

// Is there a plausible non-final dynamic-Huffman block header at bit position bp? (caller guarantees 64 bytes
// of readable input after bp)
static int plausible_block(const unsigned char* z, uint64_t bp)
{
  uint64_t w = bits_at(z, bp);
  if ((w & 7) != 4) return 0;                        // BFINAL = 0, BTYPE = 2
  unsigned hlit = (unsigned)(w >> 3) & 31, hdist = (unsigned)(w >> 8) & 31, hclen = (unsigned)(w >> 13) & 15;
  if (hlit > 29 || hdist > 29) return 0;
  static const unsigned char perm[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
  unsigned char cl[19] = {0};
  bp += 17;
  w = bits_at(z, bp);
  for (unsigned i = 0; i < hclen + 4; i++) cl[perm[i]] = (unsigned char)((w >> (3 * i)) & 7);
  bp += 3 * (hclen + 4);
  if (kraft15(cl, 19) != (1L << 15)) return 0;      // precode must be complete
  // canonical codes for the precode, bit-reversed into a 128-entry table
  unsigned short tab[128]; unsigned char tl[128];
  memset(tl, 0, sizeof tl);
  {
    int cnt[8] = {0}, next[8] = {0};
    for (int i = 0; i < 19; i++) cnt[cl[i]]++;
    cnt[0] = 0;
    int code = 0;
    for (int b = 1; b < 8; b++) { code = (code + cnt[b - 1]) << 1; next[b] = code; }
    for (int sym = 0; sym < 19; sym++) {
      int L = cl[sym]; if (!L) continue;
      int c = next[L]++, r = 0;
      for (int b = 0; b < L; b++) r |= ((c >> b) & 1) << (L - 1 - b);
      for (int f = r; f < 128; f += 1 << L) { tab[f] = (unsigned short)sym; tl[f] = (unsigned char)L; }
    }
  }
  unsigned total = hlit + 257 + hdist + 1;
  unsigned char lens[320];
  unsigned k = 0;
  while (k < total) {
    w = bits_at(z, bp);
    unsigned e = (unsigned)(w & 127);
    if (!tl[e]) return 0;
    unsigned sym = tab[e]; bp += tl[e]; w >>= tl[e];
    if (sym < 16) { lens[k++] = (unsigned char)sym; continue; }
    unsigned rep; unsigned char val;
    if (sym == 16) { if (k == 0) return 0; rep = 3 + (unsigned)(w & 3); bp += 2; val = lens[k - 1]; }
    else if (sym == 17) { rep = 3 + (unsigned)(w & 7); bp += 3; val = 0; }
    else { rep = 11 + (unsigned)(w & 127); bp += 7; val = 0; }
    if (k + rep > total) return 0;
    while (rep--) lens[k++] = val;
  }
  if (lens[256] == 0) return 0;                       // end-of-block must be codable
  if (kraft15(lens, hlit + 257) != (1L << 15)) return 0;
  long kd = kraft15(lens + hlit + 257, hdist + 1);
  if (kd < 0) return 0;
  return 1;
}


// u16 marker values: 0x8000 | i names byte i of the 32 KB history window (i = 0 oldest, 32767 newest)
// Thread 1's u16 output -> bytes: values < 256 are bytes, 0x8000 | i is byte i of the 32 KB window before the
// split. One 64K-entry byte table covers both cases (built per thread once the window is known), so every element
// is a plain table load: ~1.2 cycles per element vs ~1.8 with masked vpgatherdd (28% markers, measured).
static void marker_table(const unsigned char* win, unsigned char* tab)
{
  memset(tab, 0, 65536);
  for (int v = 0; v < 256; v++) tab[v] = (unsigned char)v;
  memcpy(tab + 0x8000, win, 32768);
}
static void resolve_markers(const uint16_t* src, size_t n, const unsigned char* tab, unsigned char* dst)
{
  size_t k = 0;
  for (; k + 8 <= n; k += 8) {
    uint64_t a, b;
    memcpy(&a, src + k, 8); memcpy(&b, src + k + 4, 8);
    uint64_t o = (uint64_t)tab[a & 0xffff] | (uint64_t)tab[(a >> 16) & 0xffff] << 8 |
                 (uint64_t)tab[(a >> 32) & 0xffff] << 16 | (uint64_t)tab[a >> 48] << 24 |
                 (uint64_t)tab[b & 0xffff] << 32 | (uint64_t)tab[(b >> 16) & 0xffff] << 40 |
                 (uint64_t)tab[(b >> 32) & 0xffff] << 48 | (uint64_t)tab[b >> 48] << 56;
    memcpy(dst + k, &o, 8);
  }
  for (; k < n; k++) dst[k] = tab[src[k]];
}

static int gz_parallel(const unsigned char* z, size_t n, unsigned char** outp, size_t* outn, size_t* outcap, size_t* used_in)
{
  size_t ds = gz_header_len(z, n);
  if (!ds || n < ds + 8192) return 1;
  size_t hint = (size_t)z[n - 4] | (size_t)z[n - 3] << 8 | (size_t)z[n - 2] << 16 | (size_t)z[n - 1] << 24;
  size_t cap = hint + (1u << 20);
  if (cap < 4 * n) cap = 4 * n;
  const size_t DICT = 32768;
  unsigned char* fin = big_alloc(cap + 64);
  uint16_t* b1 = big_alloc((DICT + cap + 64) * sizeof(uint16_t));
  struct libdeflate_decompressor* d0 = libdeflate_alloc_decompressor();
  struct libdeflate_decompressor* d1 = libdeflate_alloc_decompressor();
  if (!fin || !b1 || !d0 || !d1) goto fail;
  {
    uint16_t* out1 = b1 + DICT;
    for (size_t i = 0; i < DICT; i++) b1[i] = (uint16_t)(0x8000 | i);
    swx_configure(d0, 0, 0, 0, 0);
    enum libdeflate_result r0 = LIBDEFLATE_BAD_DATA, r1 = LIBDEFLATE_BAD_DATA;
    size_t n0 = 0, n1 = 0, in0 = 0;
    uint64_t B = 0;           // split, absolute bit position in z
    uint64_t endbit = 0;      // absolute bit position where thread 1's deflate stream ended
    int ncand = 0;
    double t0done = 0, t1done = 0, tsplit = 0, tpost = 0;
    // Post-processing in 1 MB chunks shared by both threads: CRC of thread 0's output ("A") and marker
    // resolution + CRC of thread 1's output ("B"). Thread 1 decodes in pieces and publishes how much is final,
    // so B chunks can be resolved by thread 0 while thread 1 is still decoding.
    const size_t CH = 1u << 20;
    size_t PIECE = 4u << 20;
    { const char* e = getenv("SWIFTSNP_GZPIECE"); if (e && atoll(e) > 0) PIECE = (size_t)atoll(e); }   // test knob
    size_t nchunk_max = cap / CH + 4;
    uint32_t* ccrc = xmalloc(2 * nchunk_max * sizeof(uint32_t));
    int done0 = 0, done1 = 0, split_ok0 = 0;
    size_t kA = 0, kB = 0, nextA = 0, nextB = 0, prod1 = 0;
    double frac = 0.50;
    { const char* e = getenv("SWIFTSNP_GZSPLIT"); if (e) frac = atof(e); }   // tuning knob (tests only)
#pragma omp parallel num_threads(2)
    {
      int t = omp_get_thread_num();
      if (t == 0) {
        r0 = libdeflate_deflate_decompress_ex(d0, z + ds, n - ds, fin, cap, &in0, &n0);
        t0done = now_s() - g_t0;
        kA = r0 == LIBDEFLATE_SUCCESS ? (n0 + CH - 1) / CH : 0;
        if (kA > nchunk_max) kA = 0;
        split_ok0 = r0 == LIBDEFLATE_SUCCESS && B && swx_stopped(d0) &&
                    swx_consumed_bits(d0) == B - (uint64_t)ds * 8 && n0 >= DICT;
        __atomic_store_n(&done0, 1, __ATOMIC_RELEASE);
      } else {
        uint64_t lo = (uint64_t)ds * 8 + (uint64_t)((double)(n - ds) * 8 * frac);
        uint64_t hi = lo + (uint64_t)(4u << 20) * 8;
        if (hi > (uint64_t)(n - 4096) * 8) hi = (uint64_t)(n - 4096) * 8;
        for (uint64_t bp = lo; bp < hi; bp++) {
          if (!plausible_block(z, bp)) continue;
          ncand++;
          // validate by decoding one block (markers stand in for the unknown history)
          swx_configure(d1, DICT, 0, 1, (unsigned)(bp & 7));
          size_t ib = 0, ob = 0;
          enum libdeflate_result rv = swx_deflate_decompress_u16(d1, z + (bp >> 3), n - (bp >> 3), out1, cap, &ib, &ob);
          if (rv != LIBDEFLATE_SUCCESS || !swx_stopped(d1) || ob == 0) continue;
          B = bp;
          break;
        }
        tsplit = now_s() - g_t0;
        if (B) {
          swx_publish_stop_bits(d0, B - (uint64_t)ds * 8);
          uint64_t at = B;           // absolute input bit of the next piece
          size_t produced = 0;
          for (;;) {
            swx_configure(d1, DICT + produced, 0, PIECE, (unsigned)(at & 7));
            size_t ib = 0, ob = 0;
            r1 = swx_deflate_decompress_u16(d1, z + (at >> 3), n - (at >> 3), out1 + produced, cap - produced, &ib, &ob);
            if (r1 != LIBDEFLATE_SUCCESS) break;
            produced += ob;
            at = (at >> 3) * 8 + swx_consumed_bits(d1);
            __atomic_store_n(&prod1, produced, __ATOMIC_RELEASE);
            if (!swx_stopped(d1)) break;          // final block reached
          }
          n1 = produced; endbit = at;
        }
        t1done = now_s() - g_t0;
        kB = r1 == LIBDEFLATE_SUCCESS ? (n1 + CH - 1) / CH : 0;
        if (kB > nchunk_max) kB = 0;
        __atomic_store_n(&done1, 1, __ATOMIC_RELEASE);
      }
      // shared post-processing
      unsigned char* mtab = NULL;
      for (;;) {
        if (!__atomic_load_n(&done0, __ATOMIC_ACQUIRE)) { _mm_pause(); continue; }
        size_t i = __atomic_fetch_add(&nextA, 1, __ATOMIC_RELAXED);
        if (i < kA) {
          size_t len = n0 - i * CH < CH ? n0 - i * CH : CH;
          ccrc[i] = libdeflate_crc32(0, fin + i * CH, len);
          continue;
        }
        if (!split_ok0) break;
        int d1v = __atomic_load_n(&done1, __ATOMIC_ACQUIRE);
        size_t pr = __atomic_load_n(&prod1, __ATOMIC_ACQUIRE);
        size_t avail = d1v ? kB : pr / CH;
        size_t j = __atomic_load_n(&nextB, __ATOMIC_ACQUIRE);
        if (j < avail) {
          if (!__atomic_compare_exchange_n(&nextB, &j, j + 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) continue;
          size_t total1 = d1v ? n1 : pr;
          size_t len = total1 - j * CH < CH ? total1 - j * CH : CH;
          if (n0 + j * CH + len > cap) { ccrc[nchunk_max + j] = 0; continue; }   // caught by the checks below
          if (!mtab) { mtab = xmalloc(65536); marker_table(fin + n0 - DICT, mtab); }
          resolve_markers(out1 + j * CH, len, mtab, fin + n0 + j * CH);
          ccrc[nchunk_max + j] = libdeflate_crc32(0, fin + n0 + j * CH, len);
          continue;
        }
        if (d1v && j >= kB) break;
        _mm_pause();
      }
      free(mtab);
    }
    tpost = now_s() - g_t0;
    if (g_prof) fprintf(stderr, "[gzpar] split found %.4f, t0 done %.4f, t1 done %.4f, post done %.4f\n", tsplit, t0done, t1done, tpost);
    if (g_prof) fprintf(stderr, "[gzpar] %d candidates, B=%llu (%.1f%%) r0=%d stopped=%d missed=%d n0=%zu r1=%d n1=%zu\n",
                        ncand, (unsigned long long)B, 100.0 * B / (8.0 * n), (int)r0, swx_stopped(d0), swx_missed(d0), n0, (int)r1, n1);
    if (r0 != LIBDEFLATE_SUCCESS || (kA == 0 && n0 > 0)) { free(ccrc); goto fail; }
    size_t total, tb;
    uint32_t crc = 0;
    for (size_t i = 0; i < kA; i++) {
      size_t len = n0 - i * CH < CH ? n0 - i * CH : CH;
      crc = (uint32_t)crc32_combine(crc, ccrc[i], (z_off_t)len);
    }
    if (split_ok0) {
      if (r1 != LIBDEFLATE_SUCCESS || n0 + n1 > cap) { free(ccrc); goto fail; }
      for (size_t i = 0; i < kB; i++) {
        size_t len = n1 - i * CH < CH ? n1 - i * CH : CH;
        crc = (uint32_t)crc32_combine(crc, ccrc[nchunk_max + i], (z_off_t)len);
      }
      total = n0 + n1;
      tb = (size_t)((endbit + 7) / 8);      // deflate end, rounded up to a byte: the trailer
    } else {
      if (B && swx_stopped(d0)) { free(ccrc); goto fail; }   // stopped at the split but n0 < 32 KB
      total = n0;                                     // no usable split: thread 0 decoded the whole member
      tb = ds + in0;
    }
    free(ccrc);
    if (tb + 8 > n) goto fail;
    uint32_t want = (uint32_t)z[tb] | (uint32_t)z[tb + 1] << 8 | (uint32_t)z[tb + 2] << 16 | (uint32_t)z[tb + 3] << 24;
    uint32_t isz = (uint32_t)z[tb + 4] | (uint32_t)z[tb + 5] << 8 | (uint32_t)z[tb + 6] << 16 | (uint32_t)z[tb + 7] << 24;
    if (g_prof) fprintf(stderr, "[gzpar] tb=%zu n=%zu crc=%08x want=%08x total=%zu isz=%u t=%.4f\n", tb, n, crc, want, total, isz, now_s() - g_t0);
    if (crc != want || (uint32_t)total != isz) goto fail;
    if (g_prof) fprintf(stderr, "[gzpar] resolved + crc ok, t=%.4f\n", now_s() - g_t0);
    *used_in = tb + 8;
    libdeflate_free_decompressor(d0); libdeflate_free_decompressor(d1);
    munmap(b1, hp_round((DICT + cap + 64) * sizeof(uint16_t)));
    *outp = fin; *outn = total; *outcap = cap + 64;
    return 0;
  }
fail:
  if (d0) libdeflate_free_decompressor(d0);
  if (d1) libdeflate_free_decompressor(d1);
  if (fin) munmap(fin, hp_round(cap + 64));
  if (b1) munmap(b1, hp_round((DICT + cap + 64) * sizeof(uint16_t)));
  return 1;
}

static int read_input(const char* path, inbuf_t* in)
{
  in->p = NULL; in->n = 0; in->mapped = 0;
  int fd = open(path, O_RDONLY);
  if (fd < 0) return 1;
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) { close(fd); return 1; }
  size_t n = (size_t)st.st_size;
  unsigned char magic[2] = {0, 0};
  int gz = 0;
  if (n >= 2 && pread(fd, magic, 2, 0) == 2 && magic[0] == 0x1f && magic[1] == 0x8b) gz = 1;
  if (!gz) {
    if (n == 0) { close(fd); return 1; }
    void* m = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);   // pages are faulted in by the parallel scan
    close(fd);
    if (m == MAP_FAILED) return 1;
    in->p = (unsigned char*)m; in->n = n; in->mapped = 1;
    return 0;
  }
  // gzip: decode member by member with libdeflate, reproducing gzread's stream rules: concatenated members
  // are joined; after a member, anything that does not start with the gzip magic is trailing garbage and is
  // ignored. Any decode error bails out to the reference loader (zlib gzread), so error behaviour stays exact.
  unsigned char* z = mmap(NULL, n, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  close(fd);
  if (z == MAP_FAILED) return 1;
  size_t hint = n >= 4 ? ((size_t)z[n - 4] | (size_t)z[n - 3] << 8 | (size_t)z[n - 2] << 16 | (size_t)z[n - 1] << 24) : 0;
  size_t cap = hint + 64;
  if (cap < 4 * n) cap = 4 * n;
  unsigned char* b = big_alloc(cap);
  struct libdeflate_decompressor* dz = libdeflate_alloc_decompressor();
  if (!b || !dz) { munmap(z, n); if (b) munmap(b, hp_round(cap)); if (dz) libdeflate_free_decompressor(dz); return 1; }
  size_t ip = 0, op = 0;
  int ok = 1;
  size_t par_min = 4u << 20;                       // below this, two-thread inflate is not worth it
  { const char* e = getenv("SWIFTSNP_GZPAR_MIN"); if (e) par_min = (size_t)atoll(e); }   // test knob
  if (cpus >= 2 && n >= par_min) {
    unsigned char* pb = NULL; size_t pn = 0, pcap = 0, used = 0;
    if (gz_parallel(z, n, &pb, &pn, &pcap, &used) == 0) {
      munmap(b, hp_round(cap));
      b = pb; cap = pcap; op = pn; ip = used;
    }
  }
  while (ip + 1 < n && z[ip] == 0x1f && z[ip + 1] == 0x8b) {
    size_t in_used = 0, out_got = 0;
    enum libdeflate_result r = libdeflate_gzip_decompress_ex(dz, z + ip, n - ip, b + op, cap - op, &in_used, &out_got);
    if (r == LIBDEFLATE_INSUFFICIENT_SPACE) {
      unsigned char* nb = big_realloc(b, cap, cap * 2);
      if (!nb) { ok = 0; break; }
      b = nb; cap *= 2;
      continue;
    }
    if (r != LIBDEFLATE_SUCCESS) { ok = 0; break; }
    ip += in_used; op += out_got;
  }
  libdeflate_free_decompressor(dz);
  munmap(z, n);
  if (!ok) { munmap(b, hp_round(cap)); return 1; }
  in->p = b; in->n = op; in->mapped = 2; in->cap = cap;
  return 0;
}

static void free_input(inbuf_t* in)
{
  if (!in->p) return;
  if (in->mapped == 1) munmap(in->p, in->n); else if (in->mapped == 2) munmap(in->p, hp_round(in->cap)); else free(in->p);
  in->p = NULL;
}

//------------------------------------------------------------------------ fast parser
static unsigned char IS_SPACE[256];   // C-locale isspace: ' ', \t, \n, \v, \f, \r

// Parse the structure of every record (names, where its sequence lines are). Content is not transformed here.
// Returns 0 on success, 1 to bail out to the reference loader.
static int fast_structure(const unsigned char* buf, size_t n, int predict)
{
  size_t p = 0;
  // kseq skips everything up to the first '>' or '@', wherever it is
  while (p < n && buf[p] != '>' && buf[p] != '@') p++;
  if (p >= n) return 1;  // no records: the reference reports the error
  size_t L = (size_t)-1;
  size_t Wprev = 0;            // line width of the previous record (prediction hint)
  while (p < n) {
    p++;                       // the header char
    if (p >= n) break;         // a lone '>' at EOF is not a record
    size_t ns = p;
    while (p < n && !IS_SPACE[buf[p]]) p++;
    size_t nlen = p - ns;
    const char* nm = (const char*)buf + ns;
    const char* z = memchr(nm, 0, nlen);
    size_t shown = z ? (size_t)(z - nm) : nlen;
    if (p < n) {
      unsigned char d = buf[p++];
      if (d != '\n') {
        const unsigned char* q = memchr(buf + p, '\n', n - p);
        p = q ? (size_t)(q - buf) + 1 : n;
      }
    }
    // Optimistic path for records after the first: measure the first line (width W), assume every line but the
    // last has W bytes, jump to the predicted end and check that a header or EOF follows. The interior of the
    // block (no '\n' inside lines, '\n' at every predicted line end, no '>' '@' '+' at line starts) is verified
    // during the column scan; any misprediction re-parses the whole file line by line.
    if (predict && L != (size_t)-1 && L > 0 && p < n) {
      unsigned char c0 = buf[p];
      if (c0 != '\n' && c0 != '>' && c0 != '@' && c0 != '+') {
        size_t lim = n - p; if (lim > L + 1) lim = L + 1;
        size_t W;
        // guess the previous record's line width (the scan verifies every byte of the layout anyway);
        // otherwise measure the first line
        if (Wprev && Wprev <= L && (Wprev < lim ? buf[p + Wprev] == '\n' : (Wprev == lim && p + Wprev == n)))
          W = Wprev;
        else {
          const unsigned char* q = memchr(buf + p, '\n', lim);
          W = q ? (size_t)(q - (buf + p)) : lim;
        }
        if (W >= 1 && W <= L) {
          size_t nl = L / W + (L % W ? 1 : 0);
          size_t e = p + L + nl;     // just past the last line's '\n'
          int ok = 0; size_t nx = 0;
          if (e <= n && buf[e - 1] == '\n') { ok = 1; nx = e; }
          else if (e == n + 1) { ok = 1; nx = n; }   // last line ends at EOF without '\n'
          if (ok) {
            while (nx < n && buf[nx] == '\n') nx++;
            if (nx < n && buf[nx] != '>' && buf[nx] != '@') ok = 0;
          }
          if (ok && RN < (size_t)MAX_SEQ) {
            rec_t r = { nm, shown, buf + p, W, NULL };
            push_rec(r);
            Wprev = W;
            p = nx;
            continue;
          }
        }
      }
    }
    // Optimistic path for the first record (length still unknown): measure the first line (width W) and hop
    // from line end to line end, looking only at the byte where the next '\n' should be; the first line that is
    // not W long must be the last one. Like predicted records, the interior of every line and every line start
    // are verified by the column scan (a misprediction re-parses the file line by line).
    if (predict && L == (size_t)-1 && p < n && RN < (size_t)MAX_SEQ) {
      unsigned char c0 = buf[p];
      if (c0 != '\n' && c0 != '>' && c0 != '@' && c0 != '+') {
        const unsigned char* q = memchr(buf + p, '\n', n - p);
        size_t W = q ? (size_t)(q - (buf + p)) : n - p;
        size_t e = p + W, col = W, nx = 0;   // e: end of the current line ('\n' position or n)
        int ok = 1;
        for (;;) {
          if (e >= n) { nx = n; break; }                       // sequence ends at EOF
          size_t ls = e + 1;                                   // next line start
          if (ls >= n) { nx = n; break; }
          unsigned char c = buf[ls];
          if (c == '>' || c == '@') { nx = ls; break; }        // next record
          if (c == '\n' || c == '+') { ok = 0; break; }        // blank line or '+': line-by-line parser
          if (ls + W < n && buf[ls + W] == '\n') { e = ls + W; col += W; continue; }
          // not a full line: it must be the last one (shorter than W, or W long and ending at EOF)
          size_t lim = n - ls < W + 1 ? n - ls : W + 1;
          const unsigned char* qq = memchr(buf + ls, '\n', lim);
          size_t len = qq ? (size_t)(qq - (buf + ls)) : n - ls;
          if (len > W || (!qq && n - ls > W)) { ok = 0; break; }
          col += len;
          nx = qq ? ls + len + 1 : n;
          while (nx < n && buf[nx] == '\n') nx++;
          if (nx < n && buf[nx] != '>' && buf[nx] != '@') ok = 0;
          break;
        }
        if (ok) {
          rec_t r = { nm, shown, buf + p, W, NULL };
          push_rec(r);
          L = col; Wprev = W;
          p = nx;
          continue;
        }
      }
    }
    // sequence lines
    size_t col = 0, W = 0, start = 0;
    int irregular = 0, short_seen = 0, gap = 0;
    size_t seg_cap = 0, seg_n = 0;
    size_t* segs = NULL;       // (offset, len) pairs, only kept until we know the record is regular
    while (p < n) {
      unsigned char c = buf[p];
      if (c == '\n') { p++; if (col > 0) gap = 1; continue; }
      if (c == '>' || c == '@') break;
      if (c == '+') { free(segs); return 1; }      // FASTQ: let kseq handle it
      const unsigned char* q = memchr(buf + p, '\n', n - p);
      size_t e = q ? (size_t)(q - buf) : n;
      size_t len = e - p;
      if (col == 0) { W = len; start = p; }
      else {
        if (gap || short_seen || len > W) irregular = 1;
        if (len < W) short_seen = 1;
      }
      if (seg_n + 2 > seg_cap) { seg_cap = seg_cap ? 2 * seg_cap : 64; segs = realloc(segs, seg_cap * sizeof(size_t)); }
      segs[seg_n++] = p; segs[seg_n++] = len;
      col += len;
      p = q ? e + 1 : n;
    }
    if (L == (size_t)-1) L = col;
    if (col != L) { free(segs); return 1; }        // unequal lengths: the reference reports the error
    if (RN >= (size_t)MAX_SEQ) { free(segs); return 1; }
    rec_t r = { nm, shown, buf + start, 0, NULL };
    if (col == 0) { r.seq = buf; r.W = 0; }
    else if (!irregular) { r.W = W; Wprev = W; }
    else {
      unsigned char* own = xmalloc(col);
      size_t o = 0;
      for (size_t k = 0; k < seg_n; k += 2) { memcpy(own + o, buf + segs[k], segs[k + 1]); o += segs[k + 1]; }
      r.seq = own; r.own = own; r.W = 0;
    }
    free(segs);
    push_rec(r);
  }
  if (RN < 1) return 1;
  SEQLEN = L;
  return 0;
}

// byte at column c of record r (raw for fast-path records)
static inline unsigned char rec_byte(const rec_t* r, size_t c)
{
  if (r->W == 0) return r->seq[c];
  return r->seq[(c / r->W) * (r->W + 1) + c % r->W];
}



//------------------------------------------------------------------------ column scan (4-letter modes)
// OR one-hot codes of `cnt` raw bytes into sn[0..cnt). Returns nonzero if a NUL or CR byte was seen (those need
// the reference loader's exact strcpy / CR semantics).
static unsigned scan_onehot(const unsigned char* sq, size_t cnt, unsigned char* sn, int keepcase)
{
  uint64_t bad = 0, nlm = 0;
  size_t i = 0;
#if defined(__AVX512BW__)
  // low nibbles of A C G T are 1 3 7 4. EXP[lo] is the only byte with that low nibble that is valid; unused
  // entries hold a value whose low nibble differs from the index, so they never match.
  const __m512i exp = _mm512_broadcast_i32x4(_mm_setr_epi8(0x01, 'A', 0x00, 'C', 'T', 0x00, 0x00, 'G',
                                                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
  const __m512i oht = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 0, 2, 8, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0));
  const __m512i m0f = _mm512_set1_epi8(0x0f);
  const __m512i mdf = _mm512_set1_epi8((char)(keepcase ? 0xff : 0xdf));
  const __m512i zero = _mm512_setzero_si512();
  const __m512i cr = _mm512_set1_epi8('\r');
  const __m512i nlv = _mm512_set1_epi8('\n');
  for (; i < cnt; i += 64) {
    __mmask64 m = cnt - i >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << (cnt - i)) - 1);
    __m512i b = _mm512_maskz_loadu_epi8(m, sq + i);
    __m512i u = _mm512_and_si512(b, mdf);
    __m512i lo = _mm512_and_si512(u, m0f);
    __mmask64 valid = _mm512_mask_cmpeq_epi8_mask(m, u, _mm512_shuffle_epi8(exp, lo));
    __m512i oh = _mm512_maskz_shuffle_epi8(valid, oht, lo);
    __m512i sv = _mm512_maskz_loadu_epi8(m, sn + i);
    _mm512_mask_storeu_epi8(sn + i, m, _mm512_or_si512(sv, oh));
    bad |= _mm512_mask_cmpeq_epi8_mask(m, b, zero) | _mm512_mask_cmpeq_epi8_mask(m, b, cr);
    nlm |= _mm512_mask_cmpeq_epi8_mask(m, b, nlv);
  }
#else
  for (; i < cnt; i++) {
    unsigned char b = sq[i];
    unsigned char u = keepcase ? b : (unsigned char)(b & 0xdf);
    unsigned char o = u == 'A' ? 1 : u == 'C' ? 2 : u == 'G' ? 4 : u == 'T' ? 8 : 0;
    sn[i] |= o;
    bad |= (b == 0) | (b == '\r');
    nlm |= (b == '\n');
  }
#endif
  return (bad ? 1u : 0u) | (nlm ? 2u : 0u);
}


#if defined(__AVX512BW__)
// 64 raw bytes (lanes outside m are ignored) -> plane words: [0] = bit 1, [1] = bit 2, [2] = valid (A/C/G/T).
static inline void encode64_acgt(const unsigned char* tmp, __mmask64 m, int keepcase, uint64_t* pl)
{
  const __m512i exp = _mm512_broadcast_i32x4(_mm_setr_epi8(0x01, 'A', 0x00, 'C', 'T', 0x00, 0x00, 'G',
                                                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
  const __m512i m0f = _mm512_set1_epi8(0x0f);
  __m512i b = _mm512_maskz_loadu_epi8(m, tmp);
  __m512i u = keepcase ? b : _mm512_and_si512(b, _mm512_set1_epi8((char)0xdf));
  __mmask64 valid = _mm512_mask_cmpeq_epi8_mask(m, u, _mm512_shuffle_epi8(exp, _mm512_and_si512(u, m0f)));
  pl[0] = _mm512_mask_test_epi8_mask(valid, u, _mm512_set1_epi8(0x02));
  pl[1] = _mm512_mask_test_epi8_mask(valid, u, _mm512_set1_epi8(0x04));
  pl[2] = valid;
}
#endif


#if defined(__AVX512BW__)
// Scan chunks [q0, q1) (64 bytes each) of one record's raw byte range of length RL into a raw-indexed seen array.
// Returns 1 for NUL/CR (reference loader needed), 2 for a layout misprediction.
static unsigned scan4_raw(const unsigned char* rec, size_t q0, size_t q1, size_t RL, unsigned char* seen,
                          const uint64_t* expnl, const uint64_t* expst, int keepcase)
{
  const __m512i exp = _mm512_broadcast_i32x4(_mm_setr_epi8(0x01, 'A', 0x00, 'C', 'T', 0x00, 0x00, 'G',
                                                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
  const __m512i oht = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 0, 2, 8, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0));
  // byte classes NUL \n \r + > @ (bits 0..5) via low/high nibble tables
  const __m512i lot = _mm512_broadcast_i32x4(_mm_setr_epi8(0x21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0x08, 0, 0x04, 0x10, 0));
  const __m512i hit = _mm512_broadcast_i32x4(_mm_setr_epi8(0x07, 0, 0x08, 0x10, 0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  const __m512i m0f = _mm512_set1_epi8(0x0f);
  const __m512i mdf = _mm512_set1_epi8((char)(keepcase ? 0xff : 0xdf));
  unsigned fl = 0;
  for (size_t q = q0; q < q1; q++) {
    size_t o = q * 64;
    __mmask64 m = RL - o >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << (RL - o)) - 1);
    // software prefetch 2 KB ahead: the loop has ~18 uops per 64 bytes, so the reorder buffer alone keeps too few
    // cache-line misses in flight to reach the memory bandwidth (measured: outbreak -11%, diverse -8%)
    _mm_prefetch((const char*)(rec + o + 2048), _MM_HINT_T0);
    __m512i b = _mm512_maskz_loadu_epi8(m, rec + o);
    __m512i u = _mm512_and_si512(b, mdf);
    __m512i lo = _mm512_and_si512(u, m0f);
    __mmask64 valid = _mm512_mask_cmpeq_epi8_mask(m, u, _mm512_shuffle_epi8(exp, lo));
    __m512i oh = _mm512_maskz_shuffle_epi8(valid, oht, lo);
    __m512i* sp = (__m512i*)(seen + o);
    _mm512_store_si512(sp, _mm512_or_si512(_mm512_load_si512(sp), oh));
    __m512i cls = _mm512_and_si512(_mm512_shuffle_epi8(lot, _mm512_and_si512(b, m0f)),
                                   _mm512_shuffle_epi8(hit, _mm512_and_si512(_mm512_srli_epi16(b, 4), m0f)));
    __mmask64 spec = _mm512_mask_test_epi8_mask(m, cls, cls);
    if (__builtin_expect(spec != expnl[q], 0)) {
      size_t cnt = RL - o < 64 ? RL - o : 64;
      for (size_t i = 0; i < cnt; i++) {
        unsigned char c = rec[o + i];
        int e = (expnl[q] >> i) & 1, st = (expst[q] >> i) & 1;
        if (e) { if (c != '\n') fl |= 2; }
        else {
          if (c == '\n') fl |= 2;
          if (c == 0 || c == '\r') fl |= 1;
          if (st && (c == '>' || c == '@' || c == '+')) fl |= 2;
        }
      }
    }
  }
  return fl;
}
#endif

#if defined(__AVX512BW__)
// Raw offsets in [o0, o1) whose one-hot OR has >= 2 bits set (informative columns), in increasing order.
// seen[] must be readable (and zero past the data) up to the next multiple of 64 after o1.
static size_t informative_offsets(const unsigned char* seen, size_t o0, size_t o1, size_t* out)
{
  size_t k = 0;
  const __m512i one = _mm512_set1_epi8(1);
  for (size_t o = o0; o < o1; o += 64) {
    __m512i v = _mm512_loadu_si512((const void*)(seen + o));
    uint64_t mk = _mm512_test_epi8_mask(v, _mm512_sub_epi8(v, one));
    if (o1 - o < 64) mk &= ((uint64_t)1 << (o1 - o)) - 1;
    while (mk) { out[k++] = o + (size_t)__builtin_ctzll(mk); mk &= mk - 1; }
  }
  return k;
}
#endif
// column index of each raw offset in a layout with lines of W1 - 1 bytes plus a newline (offsets increasing)
static void cols_from_roff(const size_t* roff, size_t k, size_t W1, size_t* cols)
{
  if (!k) return;
  const double inv = 1.0 / (double)W1;
  size_t line = 0, nb = 0;
  for (size_t i = 0; i < k; i++) {
    size_t o = roff[i];
    if (i == 0 || o >= nb) {
      line = (size_t)((double)o * inv);
      while (line && line * W1 > o) line--;
      while ((line + 1) * W1 <= o) line++;
      nb = (line + 1) * W1;
    }
    cols[i] = o - line;
  }
}

//------------------------------------------------------------------------ pairwise kernel
#define KERNEL_ROWS 8
#if defined(__AVX512BW__)
#include <immintrin.h>
// rows j0..j0+nr-1 against every group g >= j0/8. Per word: OR over code planes of (a ^ b), AND both valid
// planes, byte popcount via nibble table, bytes accumulated for <= 31 words, then vpsadbw into 64-bit lanes.
// difference mask of word k of this block: OR over code planes of (b ^ a), AND both valid planes
#define DIFFM(k) diffm(bw + (k) * WS, aw + (k) * WS, B)
#define PC8(x) pc8((x), lut, m0f)
static inline __attribute__((always_inline)) __m512i diffm(const uint64_t* bw, const uint64_t* aw, const int B)
{
  __m512i x = _mm512_xor_si512(_mm512_load_si512((const void*)bw), _mm512_set1_epi64((long long)aw[0]));
  for (int pl = 1; pl < B; pl++)
    x = _mm512_ternarylogic_epi64(x, _mm512_load_si512((const void*)(bw + pl * 8)), _mm512_set1_epi64((long long)aw[pl * 8]), 0xF6);
  return _mm512_ternarylogic_epi64(x, _mm512_load_si512((const void*)(bw + B * 8)), _mm512_set1_epi64((long long)aw[B * 8]), 0x80);
}
static inline __attribute__((always_inline)) __m512i pc8(__m512i x, __m512i lut, __m512i m0f)
{
  __m512i lo = _mm512_and_si512(x, m0f);
  __m512i hi = _mm512_and_si512(_mm512_srli_epi16(x, 4), m0f);
  return _mm512_add_epi8(_mm512_shuffle_epi8(lut, lo), _mm512_shuffle_epi8(lut, hi));
}
static inline __attribute__((always_inline)) void kernel_block_T(const uint64_t* enc, size_t W64, size_t P, const int B,
    size_t G, size_t N, size_t j0, size_t nr, size_t cap, void* Dv, int d16, int addmode, size_t gfrom, size_t rowoff,
    size_t ld, size_t col0)
{
  const __m512i lut = _mm512_set_epi8(4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0, 4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
                                      4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0, 4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
  const __m512i m0f = _mm512_set1_epi8(0x0f);
  const __m512i vcap = _mm512_set1_epi64((long long)cap);
  const __m512i zero = _mm512_setzero_si512();
  const uint64_t* arow[KERNEL_ROWS];
  for (size_t r = 0; r < KERNEL_ROWS; r++) {
    size_t j = j0 + (r < nr ? r : 0);
    arow[r] = enc + (j / 8) * W64 * P * 8 + (j % 8);
  }
  const size_t WS = P * 8;
  for (size_t g = gfrom; g < G; g++) {
    const uint64_t* gb = enc + g * W64 * WS;
    __m512i acc[KERNEL_ROWS], c8[KERNEL_ROWS];
    for (size_t r = 0; r < KERNEL_ROWS; r++) { acc[r] = zero; c8[r] = zero; }
    size_t w = 0;
    int nblk = 0;
    // 7 words at a time: a carry-save adder tree (vpternlog 0x96 = xor3, 0xE8 = majority) folds the 7 difference
    // masks into ones/twos/fours, so 3 nibble-table popcounts replace 7. Per byte the block adds <= 8+16+32 = 56,
    // so the byte accumulators are folded into 64-bit lanes (vpsadbw) every 4 blocks.
    for (; w + 7 <= W64; w += 7) {
      for (size_t r = 0; r < KERNEL_ROWS; r++) {
        const uint64_t* bw = gb + w * WS;
        const uint64_t* aw = arow[r] + w * WS;
        __m512i m1 = DIFFM(0), m2 = DIFFM(1), m3 = DIFFM(2);
        __m512i s1 = _mm512_ternarylogic_epi64(m1, m2, m3, 0x96), k1 = _mm512_ternarylogic_epi64(m1, m2, m3, 0xE8);
        __m512i m4 = DIFFM(3), m5 = DIFFM(4), m6 = DIFFM(5);
        __m512i s2 = _mm512_ternarylogic_epi64(m4, m5, m6, 0x96), k2 = _mm512_ternarylogic_epi64(m4, m5, m6, 0xE8);
        __m512i m7 = DIFFM(6);
        __m512i ones = _mm512_ternarylogic_epi64(s1, s2, m7, 0x96), k3 = _mm512_ternarylogic_epi64(s1, s2, m7, 0xE8);
        __m512i twos = _mm512_ternarylogic_epi64(k1, k2, k3, 0x96), fours = _mm512_ternarylogic_epi64(k1, k2, k3, 0xE8);
        __m512i t = _mm512_add_epi8(PC8(ones), _mm512_add_epi8(_mm512_slli_epi16(PC8(twos), 1), _mm512_slli_epi16(PC8(fours), 2)));
        c8[r] = _mm512_add_epi8(c8[r], t);
      }
      if (++nblk == 4) {
        for (size_t r = 0; r < KERNEL_ROWS; r++) { acc[r] = _mm512_add_epi64(acc[r], _mm512_sad_epu8(c8[r], zero)); c8[r] = zero; }
        nblk = 0;
      }
    }
    for (; w < W64; w++) {                 // <= 6 remaining words: plain popcounts (adds <= 48 per byte)
      const uint64_t* bw = gb + w * WS;
      for (size_t r = 0; r < KERNEL_ROWS; r++) {
        const uint64_t* aw = arow[r] + w * WS;
        c8[r] = _mm512_add_epi8(c8[r], PC8(DIFFM(0)));
      }
    }
    for (size_t r = 0; r < KERNEL_ROWS; r++) acc[r] = _mm512_add_epi64(acc[r], _mm512_sad_epu8(c8[r], zero));
    size_t i0 = g * 8;
    __mmask8 lanes = (__mmask8)(N - i0 >= 8 ? 0xFF : ((1u << (N - i0)) - 1));
    for (size_t r = 0; r < nr; r++) {
      __m512i d = _mm512_min_epu64(acc[r], vcap);
      size_t j = j0 + r;
      __mmask8 lm = lanes;
      if (addmode == 1) {                 // only i > j, added to what the sparse pass left in D (mode 2: all i)
        if (j >= i0 + 7) lm = 0;
        else if (j >= i0) lm &= (__mmask8)(0xFFu << (j - i0 + 1));
        if (!lm) continue;
      }
      if (d16) {
        uint16_t* dp = (uint16_t*)Dv + (j - rowoff) * ld + (i0 - col0);
        __m128i v = _mm512_cvtepi64_epi16(d);
        if (addmode) v = _mm_add_epi16(v, _mm_maskz_loadu_epi16(lm, dp));
        _mm_mask_storeu_epi16((void*)dp, lm, v);
      } else {
        uint32_t* dp = (uint32_t*)Dv + (j - rowoff) * ld + (i0 - col0);
        __m256i v = _mm512_cvtepi64_epi32(d);
        if (addmode) v = _mm256_add_epi32(v, _mm256_maskz_loadu_epi32(lm, dp));
        _mm256_mask_storeu_epi32((void*)dp, lm, v);
      }
    }
  }
}
// rows j0..j0+nr-1 against groups gfrom..G-1; output row j at Dv + (j - rowoff) * ld, column i at (i - col0)
static void kernel_block(const uint64_t* enc, size_t W64, size_t P, int B, size_t G, size_t N, size_t j0, size_t nr,
                         size_t cap, void* D, int d16, int addmode, size_t gfrom, size_t rowoff, size_t ld, size_t col0)
{
  switch (B) {
    case 1: kernel_block_T(enc, W64, P, 1, G, N, j0, nr, cap, D, d16, addmode, gfrom, rowoff, ld, col0); break;
    case 2: kernel_block_T(enc, W64, P, 2, G, N, j0, nr, cap, D, d16, addmode, gfrom, rowoff, ld, col0); break;
    case 3: kernel_block_T(enc, W64, P, 3, G, N, j0, nr, cap, D, d16, addmode, gfrom, rowoff, ld, col0); break;
    default: kernel_block_T(enc, W64, P, B, G, N, j0, nr, cap, D, d16, addmode, gfrom, rowoff, ld, col0); break;
  }
}
#else
static void kernel_block(const uint64_t* enc, size_t W64, size_t P, int B, size_t G, size_t N, size_t j0, size_t nr,
                         size_t cap, void* D, int d16, int addmode, size_t gfrom, size_t rowoff, size_t ld, size_t col0)
{
  (void)addmode; (void)ld; (void)col0;   // the sparse/dense split (add mode, tiles) is only used with AVX-512
  for (size_t r = 0; r < nr; r++) {
    size_t j = j0 + r;
    const uint64_t* a = enc + (j / 8) * W64 * P * 8 + (j % 8);
    size_t ie = G * 8 < N ? G * 8 : N;
    for (size_t i = gfrom * 8; i < ie; i++) {
      const uint64_t* b = enc + (i / 8) * W64 * P * 8 + (i % 8);
      size_t d = 0;
      for (size_t w = 0; w < W64; w++) {
        uint64_t x = 0;
        for (int pl = 0; pl < B; pl++) x |= a[(w * P + pl) * 8] ^ b[(w * P + pl) * 8];
        d += (size_t)__builtin_popcountll(x & a[(w * P + B) * 8] & b[(w * P + B) * 8]);
      }
      size_t dv = d < cap ? d : cap;
      if (d16) ((uint16_t*)D)[(j - rowoff) * N + i] = (uint16_t)dv; else ((uint32_t*)D)[(j - rowoff) * N + i] = (uint32_t)dv;
    }
  }
}
#endif


#if defined(__AVX512BW__)
//------------------------------------------------------------------------ sparse/dense column split (4-letter modes)
// In phylogenetic data most informative columns have a small minority: the sequences that differ from the
// column's majority symbol (a mutation on a short branch). For such a column c with minority set M_c, every
// pair it separates contains a member x of M_c, so its whole contribution can be added row-wise:
//   for x in M_c:  D[x][y] += 1  for every y valid at c whose symbol differs from x's,
//                  excluding members y < x of M_c (pairs inside M_c are added once, from the smaller index).
// Rows are accumulated in D's full rows (both triangles); dense columns go through the bitplane kernel, which
// adds into the upper triangle; a final pass sets D[i][j] = D[j][i] = min(D[i][j] + D[j][i], cap).
// The split is chosen per column from a cost model (row work |M_c|·N/32 vector ops vs kernel work N²/2·V/512).

// 64x64 bit-matrix transpose: out[c] bit s = in[s] bit c (in and out may alias). Bytes are regrouped with
// vpshufb + vpermw + an 8x8 qword transpose, then each output word is one vpmovb2m of a shifted vector.
static inline void transpose64(const uint64_t* in, uint64_t* out)
{
  const __m512i shuf = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15));
  const __m512i pw = _mm512_set_epi16(31, 23, 15, 7, 30, 22, 14, 6, 29, 21, 13, 5, 28, 20, 12, 4,
                                      27, 19, 11, 3, 26, 18, 10, 2, 25, 17, 9, 1, 24, 16, 8, 0);
  __m512i T[8], A[8], Bv[8], Vv[8];
  for (int i = 0; i < 8; i++)
    T[i] = _mm512_permutexvar_epi16(pw, _mm512_shuffle_epi8(_mm512_loadu_si512((const void*)(in + 8 * i)), shuf));
  for (int i = 0; i < 8; i += 2) { A[i] = _mm512_unpacklo_epi64(T[i], T[i + 1]); A[i + 1] = _mm512_unpackhi_epi64(T[i], T[i + 1]); }
  const __m512i i2a = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0), i2b = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  for (int g = 0; g < 8; g += 4)
    for (int h = 0; h < 2; h++) {
      Bv[g + h] = _mm512_permutex2var_epi64(A[g + h], i2a, A[g + 2 + h]);
      Bv[g + 2 + h] = _mm512_permutex2var_epi64(A[g + h], i2b, A[g + 2 + h]);
    }
  for (int k = 0; k < 4; k++) {
    Vv[k] = _mm512_shuffle_i64x2(Bv[k], Bv[4 + k], 0x44);
    Vv[k + 4] = _mm512_shuffle_i64x2(Bv[k], Bv[4 + k], 0xEE);
  }
  for (int k = 0; k < 8; k++) {
    __m512i v = Vv[k];
    out[8 * k + 7] = _mm512_movepi8_mask(v);
    out[8 * k + 6] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 1));
    out[8 * k + 5] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 2));
    out[8 * k + 4] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 3));
    out[8 * k + 3] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 4));
    out[8 * k + 2] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 5));
    out[8 * k + 1] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 6));
    out[8 * k + 0] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 7));
  }
}

// as transpose64, with row group i (rows 8i..8i+7) read from in + i * gstride
static inline void transpose64_g(const uint64_t* in, size_t gstride, uint64_t* out)
{
  const __m512i shuf = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15));
  const __m512i pw = _mm512_set_epi16(31, 23, 15, 7, 30, 22, 14, 6, 29, 21, 13, 5, 28, 20, 12, 4,
                                      27, 19, 11, 3, 26, 18, 10, 2, 25, 17, 9, 1, 24, 16, 8, 0);
  __m512i T[8], A[8], Bv[8], Vv[8];
  for (int i = 0; i < 8; i++)
    T[i] = _mm512_permutexvar_epi16(pw, _mm512_shuffle_epi8(_mm512_loadu_si512((const void*)(in + gstride * i)), shuf));
  for (int i = 0; i < 8; i += 2) { A[i] = _mm512_unpacklo_epi64(T[i], T[i + 1]); A[i + 1] = _mm512_unpackhi_epi64(T[i], T[i + 1]); }
  const __m512i i2a = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0), i2b = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  for (int g = 0; g < 8; g += 4)
    for (int h = 0; h < 2; h++) {
      Bv[g + h] = _mm512_permutex2var_epi64(A[g + h], i2a, A[g + 2 + h]);
      Bv[g + 2 + h] = _mm512_permutex2var_epi64(A[g + h], i2b, A[g + 2 + h]);
    }
  for (int k = 0; k < 4; k++) {
    Vv[k] = _mm512_shuffle_i64x2(Bv[k], Bv[4 + k], 0x44);
    Vv[k + 4] = _mm512_shuffle_i64x2(Bv[k], Bv[4 + k], 0xEE);
  }
  for (int k = 0; k < 8; k++) {
    __m512i v = Vv[k];
    out[8 * k + 7] = _mm512_movepi8_mask(v);
    out[8 * k + 6] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 1));
    out[8 * k + 5] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 2));
    out[8 * k + 4] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 3));
    out[8 * k + 3] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 4));
    out[8 * k + 2] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 5));
    out[8 * k + 1] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 6));
    out[8 * k + 0] = _mm512_movepi8_mask(_mm512_slli_epi16(v, 7));
  }
}


// 16x16 transpose of u16 elements held in 16 ymm rows
static inline void tr16x16(__m256i r[16])
{
  __m256i t[16], u[16];
  for (int i = 0; i < 16; i += 2) { t[i] = _mm256_unpacklo_epi16(r[i], r[i + 1]); t[i + 1] = _mm256_unpackhi_epi16(r[i], r[i + 1]); }
  for (int i = 0; i < 16; i += 4) {
    u[i] = _mm256_unpacklo_epi32(t[i], t[i + 2]); u[i + 1] = _mm256_unpackhi_epi32(t[i], t[i + 2]);
    u[i + 2] = _mm256_unpacklo_epi32(t[i + 1], t[i + 3]); u[i + 3] = _mm256_unpackhi_epi32(t[i + 1], t[i + 3]);
  }
  for (int i = 0; i < 4; i++) {
    t[2 * i] = _mm256_unpacklo_epi64(u[i], u[i + 4]); t[2 * i + 1] = _mm256_unpackhi_epi64(u[i], u[i + 4]);
    t[2 * i + 8] = _mm256_unpacklo_epi64(u[i + 8], u[i + 12]); t[2 * i + 9] = _mm256_unpackhi_epi64(u[i + 8], u[i + 12]);
  }
  for (int i = 0; i < 8; i++) {
    r[i] = _mm256_permute2x128_si256(t[i], t[i + 8], 0x20);
    r[i + 8] = _mm256_permute2x128_si256(t[i], t[i + 8], 0x31);
  }
}

// D[i][j] = D[j][i] = min(D[i][j] + D[j][i], cap) for all i != j, 16x16 tiles transposed in registers
static void combine16(uint16_t* D, size_t N, size_t cap)
{
  const uint16_t c16 = cap < 65535 ? (uint16_t)cap : 65535;
  const __m256i vcap = _mm256_set1_epi16((short)c16);
  const size_t NT = N / 16;
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t ti = 0; ti < NT; ti++) {
    size_t i0 = ti * 16;
    for (size_t tj = 0; tj < ti; tj++) {
      size_t j0 = tj * 16;
      __m256i L[16], U[16];
      for (int k = 0; k < 16; k++) {
        L[k] = _mm256_loadu_si256((const __m256i*)(D + (i0 + k) * N + j0));
        U[k] = _mm256_loadu_si256((const __m256i*)(D + (j0 + k) * N + i0));
      }
      tr16x16(U);
      for (int k = 0; k < 16; k++) {
        L[k] = _mm256_min_epu16(_mm256_add_epi16(L[k], U[k]), vcap);
        _mm256_storeu_si256((__m256i*)(D + (i0 + k) * N + j0), L[k]);
      }
      tr16x16(L);
      for (int k = 0; k < 16; k++) _mm256_storeu_si256((__m256i*)(D + (j0 + k) * N + i0), L[k]);
    }
    for (size_t i = i0; i < i0 + 16; i++)                       // diagonal tile
      for (size_t j = i0; j < i; j++) {
        size_t v = (size_t)D[i * N + j] + D[j * N + i];
        uint16_t t = (uint16_t)(v < cap ? v : cap);
        D[i * N + j] = t; D[j * N + i] = t;
      }
  }
  for (size_t i = NT * 16; i < N; i++)                          // rows past the last full tile
    for (size_t j = 0; j < i; j++) {
      size_t v = (size_t)D[i * N + j] + D[j * N + i];
      uint16_t t = (uint16_t)(v < cap ? v : cap);
      D[i * N + j] = t; D[j * N + i] = t;
    }
}

// Shared preparation of the sparse/dense column split. enc: group/lane layout with P = 3 planes (bit1, bit2,
// valid) and W64 words. Column-major bit planes, per-column majority symbol and sparse/dense choice, majority
// bitmaps of sparse columns, and the dense columns back in the group/lane layout (enc_dense, NULL if none).
typedef struct {
  size_t N, V, Vp, NB, W64, G, nsparse, nmulti, W64d, encd_words;
  const uint64_t* enc;
  uint64_t* col;                 // planes [bit1 | bit2 | valid | majority], each NB x Vp words (column-major)
  unsigned char* maj;            // majority symbol per column
  unsigned char* sparse;         // 1 = sparse, 2 = sparse with >= 2 distinct minority symbols, 0 = dense
  uint64_t *mj0, *mj1, *spm;     // per 64 columns: majority bit1, majority bit2, sparse mask
  uint64_t* encd;                // dense columns, group/lane layout
  size_t* rl_off;                // row-streamed mode: per sequence, its sparse minority columns (CSR)
  uint32_t* rl_col; unsigned char* rl_sym;   //   column index and the sequence's symbol there
  uint32_t* rl_d4;               //   symbol * Vp + column: index into a chunk's "valid, other symbol" planes
  uint32_t* lcnt; size_t lcb;    //   fused: minority members per (column block of lcb columns, sequence)
  unsigned char* rl_ord;         //   per 64-sequence block, its members by list length (NB x 64; loop trip
                                 //   counts then repeat, so the list loops' exits are mostly predicted)
} split_t;

static void split_prepare(split_t* s, const uint64_t* enc, size_t W64, size_t G, size_t N, size_t V, int rs,
                          uint64_t* colpre)
{
  memset(s, 0, sizeof *s);
  const size_t P = 3, NB = (N + 63) / 64, Vp = W64 * 64;
  s->N = N; s->V = V; s->Vp = Vp; s->NB = NB; s->W64 = W64; s->G = G; s->enc = enc;
  uint64_t* col = colpre ? colpre : big_alloc(4 * Vp * NB * sizeof(uint64_t) + 64);    // bit1, bit2, valid, majority
  if (!col) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
  s->col = col;
  // 1. column-major bit vectors: col[(p * NB + sb) * Vp + c], bit s = sequence 64*sb + s (already built when fused)
#pragma omp parallel for schedule(dynamic, 4) collapse(2)
  for (size_t sb = 0; sb < (colpre ? 0 : NB); sb++)
    for (size_t w = 0; w < W64; w++) {
      uint64_t a[64];
      for (size_t pl = 0; pl < P; pl++) {
        for (size_t q = 0; q < 64; q++) {
          size_t k = sb * 64 + q;
          a[q] = k < N ? enc[(((k >> 3) * W64 + w) * P + pl) * 8 + (k & 7)] : 0;
        }
        transpose64(a, col + (pl * NB + sb) * Vp + w * 64);
      }
    }
  const uint64_t* C0 = col; const uint64_t* C1 = col + Vp * NB; const uint64_t* CV = col + 2 * Vp * NB;
  uint64_t* MJ = col + 3 * Vp * NB;     // sequences holding the column's majority symbol
  prof("sp:transp");
  // 2. majority symbol and minority size per column; cost model decides sparse vs dense
  double dense_cost = (double)N * (double)N / 2.0 / 512.0 * 7.4;   // kernel µops per column
  double row_cost = (double)NB * 6.0 + 10.0;                        // row-update µops per minority member
  unsigned char* maj = xmalloc(Vp);
  unsigned char* sparse = xcalloc(Vp, 1);
  uint32_t* mcount = xmalloc((Vp + 1) * sizeof(uint32_t));
  unsigned char* multi = xcalloc(Vp, 1);
  size_t nsparse = 0, nmulti = 0;
  const size_t CB = 1024;          // columns per block: counts accumulate with unit-stride access per bitplane row
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t cb0 = 0; cb0 < V; cb0 += CB) {
    size_t cb1 = cb0 + CB < V ? cb0 + CB : V, nc = cb1 - cb0;
    uint32_t cnt4[4][1024];
    memset(cnt4, 0, sizeof cnt4);
    for (size_t sb = 0; sb < NB; sb++) {
      const uint64_t* v = CV + sb * Vp + cb0; const uint64_t* b0 = C0 + sb * Vp + cb0; const uint64_t* b1 = C1 + sb * Vp + cb0;
      for (size_t k = 0; k < nc; k++) {
        uint64_t vv = v[k], x0 = b0[k], x1 = b1[k];
        cnt4[0][k] += (uint32_t)__builtin_popcountll(vv & ~x0 & ~x1);
        cnt4[1][k] += (uint32_t)__builtin_popcountll(vv & x0 & ~x1);
        cnt4[2][k] += (uint32_t)__builtin_popcountll(vv & ~x0 & x1);
        cnt4[3][k] += (uint32_t)__builtin_popcountll(vv & x0 & x1);
      }
    }
    for (size_t k = 0; k < nc; k++) {
      size_t c = cb0 + k;
      size_t n4[4] = {cnt4[0][k], cnt4[1][k], cnt4[2][k], cnt4[3][k]};
      int mj = 0;
      for (int q = 1; q < 4; q++) if (n4[q] > n4[mj]) mj = q;
      size_t m = n4[0] + n4[1] + n4[2] + n4[3] - n4[mj];
      maj[c] = (unsigned char)mj;
      mcount[c] = (uint32_t)m;
      int kinds = 0;
      for (int q = 0; q < 4; q++) kinds += q != mj && n4[q] > 0;
      multi[c] = kinds >= 2;
    }
  }
  // sparse/dense choice. Matrix mode: per column, row work vs half-matrix kernel work. Row-streamed mode: dense
  // columns are paid per 64-column word of full-row kernel (~28 ns x N^2/64 per word, measured), sparse ones per
  // minority member (~4.2 ns x N/64); the highest-minority columns fill dense words while a word's sparse cost
  // would exceed its dense cost (large_n: its 15 high-minority columns are cheaper sparse, -10%; core keeps its
  // 3 dense words).
  if (!rs) {
    for (size_t c = 0; c < V; c++)
      if (mcount[c] >= 1 && (double)mcount[c] * row_cost < dense_cost) sparse[c] = 1;
  } else {
    // columns with a minority, by minority size descending (counting sort: sizes are <= N)
    uint64_t* ord = xmalloc((V + 1) * sizeof(uint64_t));
    size_t* bucket = xcalloc(N + 2, sizeof(size_t));
    size_t nc = 0;
    for (size_t c = 0; c < V; c++) if (mcount[c] >= 1) { bucket[N - mcount[c]]++; sparse[c] = 1; nc++; }
    for (size_t q = 0, acc = 0; q <= N; q++) { size_t b = bucket[q]; bucket[q] = acc; acc += b; }
    for (size_t c = 0; c < V; c++) if (mcount[c] >= 1) ord[bucket[N - mcount[c]]++] = (uint64_t)mcount[c] << 32 | c;
    free(bucket);
    const double word_cost = 28.2 * (double)N * (double)N / 64.0, member_cost = 4.2 * (double)NB;
    for (size_t w0 = 0; w0 < nc; w0 += 64) {
      size_t w1 = w0 + 64 < nc ? w0 + 64 : nc;
      double sm = 0;
      for (size_t k = w0; k < w1; k++) sm += (double)(ord[k] >> 32);
      // the first dense word carries the kernel's per-(row, group) overhead; later words cost ~0.6 of it
      if (sm * member_cost <= (w0 == 0 ? word_cost : 0.6 * word_cost)) break;
      for (size_t k = w0; k < w1; k++) sparse[(uint32_t)ord[k]] = 0;
    }
    free(ord);
  }
  for (size_t c = 0; c < V; c++) if (sparse[c]) { if (multi[c]) sparse[c] = 2; nsparse++; nmulti += multi[c]; }
  free(mcount); free(multi);
  // row-streamed + fused: also count each sequence's minority members per column block (lists are built from
  // these counts in rs_lists_col without another pass over the planes)
  size_t ncb = (V + CB - 1) / CB;
  uint32_t* lcnt = (rs && colpre) ? xcalloc(ncb * NB * 64 + 1, sizeof(uint32_t)) : NULL;
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t cb0 = 0; cb0 < V; cb0 += CB) {
    size_t cb1 = cb0 + CB < V ? cb0 + CB : V, nc = cb1 - cb0;
    uint32_t* lc = lcnt ? lcnt + (cb0 / CB) * NB * 64 : NULL;
    for (size_t sb = 0; sb < NB; sb++) {             // majority bitmaps of the sparse columns
      for (size_t k = 0; k < nc; k++) {
        size_t c = cb0 + k;
        if (!sparse[c]) continue;
        uint64_t m0 = (maj[c] & 1) ? ~0ULL : 0, m1 = (maj[c] & 2) ? ~0ULL : 0;
        uint64_t cv = CV[sb * Vp + c], mj = cv & ~(C0[sb * Vp + c] ^ m0) & ~(C1[sb * Vp + c] ^ m1);
        MJ[sb * Vp + c] = mj;
        if (lc) { uint64_t mm = cv & ~mj; while (mm) { lc[sb * 64 + (size_t)__builtin_ctzll(mm)]++; mm &= mm - 1; } }
      }
    }
  }
  s->lcnt = lcnt; s->lcb = CB;
  uint64_t* mj0 = xcalloc(W64 + 1, sizeof(uint64_t));
  uint64_t* mj1 = xcalloc(W64 + 1, sizeof(uint64_t));
  uint64_t* spm = xcalloc(W64 + 1, sizeof(uint64_t));
  for (size_t c = 0; c < V; c++) {
    if (maj[c] & 1) mj0[c >> 6] |= 1ULL << (c & 63);
    if (maj[c] & 2) mj1[c >> 6] |= 1ULL << (c & 63);
    if (sparse[c]) spm[c >> 6] |= 1ULL << (c & 63);
  }
  s->maj = maj; s->sparse = sparse; s->mj0 = mj0; s->mj1 = mj1; s->spm = spm;
  s->nsparse = nsparse; s->nmulti = nmulti;
  prof("sp:classify");
  // 3. dense columns back to the group/lane layout
  size_t Vd = V - nsparse, W64d = (Vd + 63) / 64;
  s->W64d = W64d;
  if (Vd) {
    uint32_t* dc = xmalloc(Vd * sizeof(uint32_t));
    size_t k = 0;
    for (size_t c = 0; c < V; c++) if (!sparse[c]) dc[k++] = (uint32_t)c;
    size_t words = G * W64d * P * 8 + 8;
    uint64_t* e2 = big_alloc(words * sizeof(uint64_t));
    if (!e2) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
#pragma omp parallel for schedule(dynamic, 4) collapse(2)
    for (size_t sb = 0; sb < NB; sb++)
      for (size_t wd = 0; wd < W64d; wd++) {
        uint64_t a[64];
        for (size_t pl = 0; pl < P; pl++) {
          for (size_t q = 0; q < 64; q++) {
            size_t d = wd * 64 + q;
            a[q] = d < Vd ? col[(pl * NB + sb) * Vp + dc[d]] : 0;
          }
          transpose64(a, a);
          for (size_t q = 0; q < 64; q++) {
            size_t kk = sb * 64 + q;
            if (kk < N) e2[(((kk >> 3) * W64d + wd) * P + pl) * 8 + (kk & 7)] = a[q];
          }
        }
      }
    free(dc);
    s->encd = e2; s->encd_words = words;
  }
  prof("sp:dense-enc");
}

static void split_free(split_t* s)
{
  free(s->maj); free(s->sparse); free(s->mj0); free(s->mj1); free(s->spm);
  free(s->rl_off); free(s->rl_col); free(s->rl_sym); free(s->rl_d4); free(s->rl_ord);
  if (s->col) munmap(s->col, hp_round(4 * s->Vp * s->NB * sizeof(uint64_t) + 64));
  if (s->encd) munmap(s->encd, hp_round(s->encd_words * sizeof(uint64_t)));
  s->col = NULL; s->encd = NULL; s->maj = NULL; s->sparse = NULL; s->mj0 = s->mj1 = s->spm = NULL;
  s->rl_off = NULL; s->rl_col = NULL; s->rl_sym = NULL; s->rl_d4 = NULL; s->rl_ord = NULL;
}

// x's sparse minority columns (valid at the column, symbol != majority), from the sequence-major planes
static inline size_t minority_list(const split_t* s, size_t x, uint32_t* list)
{
  const size_t P = 3;
  const uint64_t* ex = s->enc + (x >> 3) * s->W64 * P * 8 + (x & 7);
  size_t nl = 0;
  for (size_t w = 0; w < s->W64; w++) {
    const uint64_t* ew = ex + w * P * 8;
    uint64_t mb = ew[16] & ((ew[0] ^ s->mj0[w]) | (ew[8] ^ s->mj1[w])) & s->spm[w];
    while (mb) { list[nl++] = (uint32_t)(w * 64 + (size_t)__builtin_ctzll(mb)); mb &= mb - 1; }
  }
  return nl;
}

// row x += +1 for every y holding the majority symbol of a sparse column where x is in the minority
// (stores rows[x - rowoff] for 64-sequence blocks; d16 / d32)
static inline void minority_row(const split_t* s, const uint32_t* list, size_t nl, int d16, void* rowp)
{
  const size_t N = s->N, Vp = s->Vp;
  const uint64_t* MJ = s->col + 3 * Vp * s->NB;
  for (size_t sb = 0; sb < s->NB; sb++) {
    const uint64_t* mjb = MJ + sb * Vp;
    size_t y0 = sb * 64, ny = N - y0 < 64 ? N - y0 : 64;
    if (d16) {
      const __m512i one = _mm512_set1_epi16(1);
      __m512i r0 = _mm512_setzero_si512(), r1 = _mm512_setzero_si512();
      for (size_t e = 0; e < nl; e++) {
        uint64_t mk = mjb[list[e]];
        r0 = _mm512_mask_add_epi16(r0, (__mmask32)mk, r0, one);
        r1 = _mm512_mask_add_epi16(r1, (__mmask32)(mk >> 32), r1, one);
      }
      uint16_t* r = (uint16_t*)rowp + y0;
      __mmask32 k0 = (__mmask32)(ny >= 32 ? 0xFFFFFFFFu : ((1u << ny) - 1));
      __mmask32 k1 = (__mmask32)(ny <= 32 ? 0 : (ny >= 64 ? 0xFFFFFFFFu : ((1u << (ny - 32)) - 1)));
      _mm512_mask_storeu_epi16(r, k0, r0); _mm512_mask_storeu_epi16(r + 32, k1, r1);
    } else {
      const __m512i one = _mm512_set1_epi32(1);
      __m512i rq[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
      for (size_t e = 0; e < nl; e++) {
        uint64_t mk = mjb[list[e]];
        for (int q = 0; q < 4; q++) rq[q] = _mm512_mask_add_epi32(rq[q], (__mmask16)(mk >> (16 * q)), rq[q], one);
      }
      uint32_t* r = (uint32_t*)rowp + y0;
      for (int q = 0; q < 4; q++) {
        size_t base = (size_t)q * 16;
        if (base >= ny) break;
        __mmask16 lq = (__mmask16)(ny - base >= 16 ? 0xFFFFu : ((1u << (ny - base)) - 1));
        _mm512_mask_storeu_epi32(r + base, lq, rq[q]);
      }
    }
  }
}

// Matrix mode: sparse contributions into D (rows of minority members, then cross-symbol minority pairs once at
// D[x][y], x < y); D's two triangles are combined afterwards.
static void split_rows_D(split_t* s, int d16, void* D)
{
  const size_t N = s->N, V = s->V, Vp = s->Vp, NB = s->NB;
  const uint64_t* C0 = s->col; const uint64_t* C1 = s->col + Vp * NB; const uint64_t* CV = s->col + 2 * Vp * NB;
  size_t summ = 0;
#pragma omp parallel reduction(+:summ)
  {
    uint32_t* mylist = xmalloc((V + 1) * sizeof(uint32_t));
#pragma omp for schedule(dynamic, 4)
    for (size_t x = 0; x < N; x++) {
      size_t nl = minority_list(s, x, mylist);
      summ += nl;
      if (!nl) continue;
      minority_row(s, mylist, nl, d16, d16 ? (void*)((uint16_t*)D + x * N) : (void*)((uint32_t*)D + x * N));
    }
    free(mylist);
  }
  prof("sp:rows");
  if (s->nmulti) {
#pragma omp parallel
    {
    size_t* who = xmalloc(4 * N * sizeof(size_t)); size_t ng[4];
#pragma omp for schedule(dynamic, 16)
    for (size_t c = 0; c < V; c++) if (s->sparse[c] == 2) {
      uint64_t m0 = (s->maj[c] & 1) ? ~0ULL : 0, m1 = (s->maj[c] & 2) ? ~0ULL : 0;
      ng[0] = ng[1] = ng[2] = ng[3] = 0;
      for (size_t sb = 0; sb < NB; sb++) {
        uint64_t b0 = C0[sb * Vp + c], b1 = C1[sb * Vp + c];
        uint64_t mb = CV[sb * Vp + c] & ((b0 ^ m0) | (b1 ^ m1));
        while (mb) {
          unsigned bit = (unsigned)__builtin_ctzll(mb);
          unsigned sy = (unsigned)(((b0 >> bit) & 1) | (((b1 >> bit) & 1) << 1));
          who[sy * N + ng[sy]++] = sb * 64 + bit;
          mb &= mb - 1;
        }
      }
      for (int ga = 0; ga < 4; ga++)
        for (int gb = ga + 1; gb < 4; gb++)
          for (size_t a = 0; a < ng[ga]; a++)
            for (size_t b = 0; b < ng[gb]; b++) {
              size_t xa = who[ga * N + a], xb = who[gb * N + b];
              size_t lo = xa < xb ? xa : xb, hi = xa < xb ? xb : xa;
              if (d16) __atomic_fetch_add((uint16_t*)D + lo * N + hi, (uint16_t)1, __ATOMIC_RELAXED);
              else __atomic_fetch_add((uint32_t*)D + lo * N + hi, (uint32_t)1, __ATOMIC_RELAXED);
            }
    }
    free(who);
    }
  }
  prof("sp:pairs");
  if (g_prof) fprintf(stderr, "[split] %zu sparse (%zu with >= 2 minority symbols, sum |M| = %zu), %zu dense columns\n",
                      s->nsparse, s->nmulti, summ, V - s->nsparse);
}

// Row-streamed mode (no N x N matrix): final rows are produced per 64-row output chunk. For a sparse column c
// and rows x, y valid at c with different symbols, the pair is counted from x's side when x is in the minority
// and y holds the majority (minority_row), and otherwise from y's side: y is in the minority and x holds any
// valid symbol other than y's ("column side", which also covers two minority members with different symbols).
// The column side for the chunk's 64 rows is accumulated per y in registers over y's minority list and stored
// in a y-major tile T[y][x - r0], then added to the rows with 16x16 transposes. Dense columns go through the
// kernel for full rows (add mode 2).
// per 64-sequence block, its members ordered by list length (stable insertion sort)
static void rl_order(split_t* s)
{
  const size_t N = s->N; const size_t* off = s->rl_off;
  unsigned char* ord = xmalloc(s->NB * 64);
  for (size_t b = 0; b < s->NB; b++) {
    unsigned char* o = ord + b * 64;
    size_t y0 = b * 64, ny = N - y0 < 64 ? N - y0 : 64;
    for (size_t i = 0; i < 64; i++) o[i] = (unsigned char)i;
    for (size_t i = 1; i < ny; i++) {
      unsigned char v = o[i]; size_t lv = off[y0 + v + 1] - off[y0 + v], j = i;
      while (j > 0 && off[y0 + o[j - 1] + 1] - off[y0 + o[j - 1]] > lv) { o[j] = o[j - 1]; j--; }
      o[j] = v;
    }
  }
  s->rl_ord = ord;
}

static void rs_lists(split_t* s)
{
  const size_t N = s->N, V = s->V;
  size_t* off = xmalloc((N + 1) * sizeof(size_t));
  off[0] = 0;
  // pass 1: list sizes; pass 2: column indices and symbols
#pragma omp parallel
  {
    uint32_t* tmp = xmalloc((V + 1) * sizeof(uint32_t));
#pragma omp for schedule(dynamic, 16)
    for (size_t x = 0; x < N; x++) off[x + 1] = minority_list(s, x, tmp);
    free(tmp);
  }
  for (size_t x = 0; x < N; x++) off[x + 1] += off[x];
  uint32_t* rc = xmalloc((off[N] + 1) * sizeof(uint32_t));
  unsigned char* rsy = xmalloc(off[N] + 1);
  const size_t P = 3;
#pragma omp parallel for schedule(dynamic, 16)
  for (size_t x = 0; x < N; x++) {
    size_t nl = minority_list(s, x, rc + off[x]);
    const uint64_t* ex = s->enc + (x >> 3) * s->W64 * P * 8 + (x & 7);
    for (size_t e = 0; e < nl; e++) {
      size_t c = rc[off[x] + e];
      const uint64_t* ew = ex + (c >> 6) * P * 8;
      rsy[off[x] + e] = (unsigned char)(((ew[0] >> (c & 63)) & 1) | (((ew[8] >> (c & 63)) & 1) << 1));
    }
  }
  uint32_t* rd4 = xmalloc((off[N] + 1) * sizeof(uint32_t));
  for (size_t e = 0; e < off[N]; e++) rd4[e] = (uint32_t)(rsy[e] * s->Vp + rc[e]);
  s->rl_off = off; s->rl_col = rc; s->rl_sym = rsy; s->rl_d4 = rd4;
  rl_order(s);
  if (g_prof) fprintf(stderr, "[split] row-streamed: %zu sparse (%zu with >= 2 minority symbols, sum |M| = %zu), %zu dense columns\n",
                      s->nsparse, s->nmulti, off[N], V - s->nsparse);
  prof("sp:lists");
}

// final u16 rows r0..r1-1 (r0 a multiple of 64, r1 - r0 <= 64), capped, into rows[(x - r0) * N + y].
// T: NB * 64 * 64 u16, 64-byte aligned. Work per 64-sequence block of y in an L1 tile: column side
// (transposed from T), dense columns, minority side, cap; one store per row segment.
static void rs_rows(const split_t* s, size_t r0, size_t r1, uint16_t* rows, uint16_t* T, uint64_t* D4, size_t cap)
{
  const size_t N = s->N, Vp = s->Vp, NB = s->NB, sb = r0 / 64;
  // 1. column side: T[y][x - r0] = #{c in y's minority list: x valid at c with a symbol other than y's}.
  //    First the chunk's four "valid and symbol != s" planes (D4[s * Vp + c]), so each list entry is one load;
  //    T is computed per block of 64 y below.
  const uint64_t* C0 = s->col + sb * Vp; const uint64_t* C1 = s->col + (NB + sb) * Vp; const uint64_t* CV = s->col + (2 * NB + sb) * Vp;
  for (size_t c = 0; c < Vp; c += 8) {
    __m512i w0 = _mm512_loadu_si512((const void*)(C0 + c)), w1 = _mm512_loadu_si512((const void*)(C1 + c));
    __m512i v = _mm512_loadu_si512((const void*)(CV + c));
    _mm512_storeu_si512((void*)(D4 + c), _mm512_ternarylogic_epi64(v, w0, w1, 0xE0));            // v & (w0 | w1)
    _mm512_storeu_si512((void*)(D4 + Vp + c), _mm512_ternarylogic_epi64(v, w0, w1, 0xB0));       // v & (~w0 | w1)
    _mm512_storeu_si512((void*)(D4 + 2 * Vp + c), _mm512_ternarylogic_epi64(v, w0, w1, 0xD0));   // v & (w0 | ~w1)
    _mm512_storeu_si512((void*)(D4 + 3 * Vp + c), _mm512_ternarylogic_epi64(v, w0, w1, 0x70));   // v & ~(w0 & w1)
  }
  const __m512i one8 = _mm512_set1_epi8(1);
  const unsigned char* xo = s->rl_ord + sb * 64;     // this chunk's rows by list length
  // 2. per block of 64 y: the column side for those y (T: 64 x 64, L1-resident; y in list-length order), its
  //    transpose, the dense part and the row side. Lists are counted in 8-bit lanes (one masked add per entry for
  //    all 64 x or y), widened every 255 entries.
  uint16_t tile[64 * 64] __attribute__((aligned(64)));
  const uint64_t* MJ = s->col + 3 * Vp * NB;
  const __m512i vcap = _mm512_set1_epi16((short)(cap < 65535 ? cap : 65535));
  for (size_t yb = 0; yb < NB; yb++) {
    size_t y0 = yb * 64, ny = N - y0 < 64 ? N - y0 : 64;
    const unsigned char* yo = s->rl_ord + yb * 64;
    for (size_t i = 0; i < 64; i++) {
      size_t yy = yo[i];
      __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
      if (yy < ny) {
        size_t y = y0 + yy;
        const uint32_t* ld = s->rl_d4 + s->rl_off[y];
        size_t nl = s->rl_off[y + 1] - s->rl_off[y];
        for (size_t e0 = 0; e0 < nl; e0 += 255) {
          size_t e1 = nl - e0 < 255 ? nl : e0 + 255;
          __m512i c8 = _mm512_setzero_si512();
          for (size_t e = e0; e < e1; e++) c8 = _mm512_mask_add_epi8(c8, (__mmask64)D4[ld[e]], c8, one8);
          a0 = _mm512_add_epi16(a0, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(c8)));
          a1 = _mm512_add_epi16(a1, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(c8, 1)));
        }
      }
      _mm512_store_si512((void*)(T + yy * 64), a0); _mm512_store_si512((void*)(T + yy * 64 + 32), a1);
    }
    for (size_t xt = 0; xt < 64; xt += 16)
      for (size_t yt = 0; yt < 64; yt += 16) {
        __m256i t[16];
        for (int i = 0; i < 16; i++) t[i] = _mm256_load_si256((const __m256i*)(T + (yt + i) * 64 + xt));
        tr16x16(t);
        for (int i = 0; i < 16; i++) _mm256_store_si256((__m256i*)(tile + (xt + i) * 64 + yt), t[i]);
      }
    if (s->encd) {
      size_t gend = yb * 8 + 8 < s->G ? yb * 8 + 8 : s->G;
      for (size_t j0 = r0; j0 < r1; j0 += KERNEL_ROWS) {
        size_t nk = r1 - j0 < KERNEL_ROWS ? r1 - j0 : KERNEL_ROWS;
        kernel_block(s->encd, s->W64d, 3, 2, gend, N, j0, nk, (size_t)-1, tile, 1, 2, yb * 8, r0, 64, y0);
      }
    }
    const uint64_t* mjb = MJ + yb * Vp;
    __mmask32 k0 = (__mmask32)(ny >= 32 ? 0xFFFFFFFFu : ((1u << ny) - 1));
    __mmask32 k1 = (__mmask32)(ny <= 32 ? 0 : (ny >= 64 ? 0xFFFFFFFFu : ((1u << (ny - 32)) - 1)));
    for (size_t i = 0; i < 64; i++) {
      size_t x = r0 + xo[i];
      if (x >= r1) continue;
      __m512i a0 = _mm512_load_si512((const void*)(tile + (x - r0) * 64));
      __m512i a1 = _mm512_load_si512((const void*)(tile + (x - r0) * 64 + 32));
      const uint32_t* lc = s->rl_col + s->rl_off[x];
      size_t nl = s->rl_off[x + 1] - s->rl_off[x];
      for (size_t e0 = 0; e0 < nl; e0 += 255) {
        size_t e1 = nl - e0 < 255 ? nl : e0 + 255;
        __m512i c8 = _mm512_setzero_si512();
        for (size_t e = e0; e < e1; e++) c8 = _mm512_mask_add_epi8(c8, (__mmask64)mjb[lc[e]], c8, one8);
        a0 = _mm512_add_epi16(a0, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(c8)));
        a1 = _mm512_add_epi16(a1, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(c8, 1)));
      }
      uint16_t* rp = rows + (x - r0) * N + y0;
      _mm512_mask_storeu_epi16(rp, k0, _mm512_min_epu16(a0, vcap));
      _mm512_mask_storeu_epi16(rp + 32, k1, _mm512_min_epu16(a1, vcap));
    }
  }
}
#endif


#if defined(__AVX512BW__)
// ---- gather + encode of one sequence (4-letter modes): context of the three input layouts
typedef struct {
  const rec_t* R; size_t V, W64, L, Wref, RLr, NQr;
  const size_t* offW; const size_t* cols; const uint64_t* infbits; int dense, keepcase;
  size_t pg_n; size_t* pg_k; unsigned char** pg_x;
} genc_t;

// Sequence k's informative columns as plane words: e[w * 24 + 0 / 8 / 16] = bit 1 / bit 2 / valid of columns
// 64w..64w+63 (e points at k's lane in its group of 8). row: V + 64 bytes of scratch (pre-gathered input only).
static void enc_seq(const genc_t* g, size_t k, uint64_t* e, unsigned char* row)
{
  const size_t P = 3, V = g->V, W64 = g->W64;
  if (g->pg_n) {
    // bytes were gathered during the scan: concatenate the record's block slices and encode
    size_t pos = 0;
    for (size_t b = 0; b < g->pg_n; b++)
      if (g->pg_k[b]) { memcpy(row + pos, g->pg_x[b] + k * g->pg_k[b], g->pg_k[b]); pos += g->pg_k[b]; }
    for (size_t w = 0; w < W64; w++) {
      size_t cnt = V - w * 64 < 64 ? V - w * 64 : 64;
      __mmask64 m = cnt == 64 ? ~(__mmask64)0 : (((__mmask64)1 << cnt) - 1);
      uint64_t pl[3];
      encode64_acgt(row + w * 64, m, g->keepcase, pl);
      uint64_t* ew = e + w * P * 8;
      ew[0] = pl[0]; ew[8] = pl[1]; ew[16] = pl[2];
    }
    return;
  }
  const rec_t* r = &g->R[k];
  size_t We = r->W ? r->W : (g->L ? g->L : 1);
  if (g->dense && We == g->Wref) {
    // dense alignments: encode whole 64-byte raw chunks, squeeze out non-informative positions with pext
    uint64_t a0 = 0, a1 = 0, a2 = 0;
    unsigned nb = 0;
    size_t w = 0;
    for (size_t q = 0; q < g->NQr; q++) {
      uint64_t sel = g->infbits[q];
      if (!sel) continue;
      size_t o = q * 64;
      __mmask64 m = g->RLr - o >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << (g->RLr - o)) - 1);
      uint64_t pl[3];
      _mm_prefetch((const char*)(r->seq + o + 2048), _MM_HINT_T0);
      encode64_acgt(r->seq + o, m, g->keepcase, pl);
      unsigned kk = (unsigned)__builtin_popcountll(sel);
      uint64_t b0 = _pext_u64(pl[0], sel), b1 = _pext_u64(pl[1], sel), b2 = _pext_u64(pl[2], sel);
      a0 |= b0 << nb; a1 |= b1 << nb; a2 |= b2 << nb;
      if (nb + kk >= 64) {
        uint64_t* ew = e + w * P * 8;
        ew[0] = a0; ew[8] = a1; ew[16] = a2;
        w++;
        a0 = nb ? b0 >> (64 - nb) : 0; a1 = nb ? b1 >> (64 - nb) : 0; a2 = nb ? b2 >> (64 - nb) : 0;
        nb = nb + kk - 64;
      } else nb += kk;
    }
    if (nb) { uint64_t* ew = e + w * P * 8; ew[0] = a0; ew[8] = a1; ew[16] = a2; }
    return;
  }
  const size_t* off = We == g->Wref ? g->offW : (r->W == 0 ? g->cols : NULL);
  unsigned char tmp[64];
  for (size_t w = 0; w < W64; w++) {
    size_t v0 = w * 64, cnt = V - v0 < 64 ? V - v0 : 64;
    if (off) for (size_t q = 0; q < cnt; q++) tmp[q] = r->seq[off[v0 + q]];
    else for (size_t q = 0; q < cnt; q++) tmp[q] = rec_byte(r, g->cols[v0 + q]);
    __mmask64 m = cnt == 64 ? ~(__mmask64)0 : (((__mmask64)1 << cnt) - 1);
    uint64_t pl[3];
    encode64_acgt(tmp, m, g->keepcase, pl);
    uint64_t* ew = e + w * P * 8;
    ew[0] = pl[0]; ew[8] = pl[1]; ew[16] = pl[2];
  }
}

// Row-streamed split, fused front end: per 64-sequence block, encode the block's sequences into a thread-local
// sequence-major buffer (~190 KB for 8,000 columns, stays in L2) and transpose it into the column-major planes
// col[(p * NB + sb) * Vp + c] (bit1, bit2, valid; the majority plane is filled by split_prepare). Same planes as
// encoding all sequences first and transposing afterwards, without the N x V/8 x 3 array and its page zeroing.
static uint64_t* split_build_fused(const genc_t* g, size_t N)
{
  const size_t P = 3, W64 = g->W64, Vp = W64 * 64, NB = (N + 63) / 64;
  uint64_t* col = big_alloc(4 * Vp * NB * sizeof(uint64_t) + 64);
  if (!col) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
#pragma omp parallel
  {
    uint64_t* eb = xmalloc((8 * W64 * P * 8 + 8) * sizeof(uint64_t));
    unsigned char* row = g->pg_n ? xmalloc(g->V + 64) : NULL;
#pragma omp for schedule(dynamic, 1)
    for (size_t sb = 0; sb < NB; sb++) {
      size_t k0 = sb * 64, k1 = k0 + 64 < N ? k0 + 64 : N;
      if (k1 - k0 < 64) memset(eb, 0, (8 * W64 * P * 8) * sizeof(uint64_t));   // lanes past N read as 0
      for (size_t k = k0; k < k1; k++) enc_seq(g, k, eb + ((k - k0) >> 3) * W64 * P * 8 + (k & 7), row);
      // group i's 8 words of (w, plane) are contiguous in eb: the transpose loads them directly
      for (size_t w = 0; w < W64; w++)
        for (size_t pl = 0; pl < P; pl++)
          transpose64_g(eb + w * P * 8 + pl * 8, W64 * P * 8, col + (pl * NB + sb) * Vp + w * 64);
    }
    free(eb); free(row);
  }
  return col;
}

// Minority lists from the column-major planes: member x of sparse column c (valid, symbol != majority) is a set bit
// of CV & ~MJ in x's 64-sequence block. The per-(column block, sequence) counts come from classify (s->lcnt), so
// each column block has its own write position per sequence and every list comes out in ascending column order.
// rl_sym is not kept (the rows need only rl_col and rl_d4).
static void rs_lists_col(split_t* s)
{
  const size_t N = s->N, V = s->V, Vp = s->Vp, NB = s->NB, CB = s->lcb, ncb = (V + CB - 1) / CB, NP = NB * 64;
  const uint64_t* C0 = s->col; const uint64_t* C1 = s->col + Vp * NB; const uint64_t* CV = s->col + 2 * Vp * NB;
  const uint64_t* MJ = s->col + 3 * Vp * NB;
  const unsigned char* sparse = s->sparse;
  uint32_t* cnt = s->lcnt;              // [cb][x] -> becomes the write position of column block cb for sequence x
  size_t* off = xmalloc((N + 1) * sizeof(size_t));
  size_t acc = 0;
  for (size_t x = 0; x < N; x++) {
    off[x] = acc;
    for (size_t q = 0; q < ncb; q++) { uint32_t v = cnt[q * NP + x]; cnt[q * NP + x] = (uint32_t)acc; acc += v; }
  }
  off[N] = acc;
  uint32_t* rc = xmalloc((acc + 1) * sizeof(uint32_t));
  uint32_t* rd4 = xmalloc((acc + 1) * sizeof(uint32_t));
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t cb0 = 0; cb0 < V; cb0 += CB) {
    size_t cb1 = cb0 + CB < V ? cb0 + CB : V;
    uint32_t* ct = cnt + (cb0 / CB) * NP;
    for (size_t sb = 0; sb < NB; sb++) {
      const uint64_t* cv = CV + sb * Vp; const uint64_t* mj = MJ + sb * Vp;
      const uint64_t* b0 = C0 + sb * Vp; const uint64_t* b1 = C1 + sb * Vp;
      uint32_t* cb = ct + sb * 64;
      for (size_t c = cb0; c < cb1; c++) {
        if (!sparse[c]) continue;
        uint64_t mm = cv[c] & ~mj[c];
        while (mm) {
          unsigned b = (unsigned)__builtin_ctzll(mm); mm &= mm - 1;
          uint32_t e = cb[b]++;
          rc[e] = (uint32_t)c;
          rd4[e] = (uint32_t)((((b0[c] >> b) & 1) | (((b1[c] >> b) & 1) << 1)) * Vp + c);
        }
      }
    }
  }
  free(cnt); s->lcnt = NULL;
  s->rl_off = off; s->rl_col = rc; s->rl_sym = NULL; s->rl_d4 = rd4;
  rl_order(s);
  if (g_prof) fprintf(stderr, "[split] row-streamed (fused): %zu sparse (%zu with >= 2 minority symbols, sum |M| = %zu), %zu dense columns\n",
                      s->nsparse, s->nmulti, off[N], V - s->nsparse);
  prof("sp:lists");
}
#endif

//------------------------------------------------------------------------ main
void show_help(int retcode)
{
  FILE* out = (retcode == EXIT_SUCCESS ? stdout : stderr);
  static const char str[] = {
      "SYNOPSIS\n  Pairwise SNP distance matrix from a FASTA alignment\n"
      "USAGE\n  %s [opts] aligned.fasta[.gz] > matrix.tsv\n"
      "OPTIONS\n"
      "  -h       Show this help\n"
      "  -v       Print version and exit\n"
      "  -j INT   Threads to use [%d]\n"
      "  -q       Quiet mode; no progress messages\n"
      "  -a       Count all differences not just [AGTC]\n"
      "  -k       Keep case, don't uppercase all letters\n"
      "  -m       Output MOLTEN instead of TSV\n"
      "  -L       Ootput lower-trangle only (unique pairs)\n"
      "  -t       Add header to MOLTEN output\n"
      "  -c       Use comma instead of tab in output\n"
      "  -b       Blank top left corner cell\n"
      "  -x INT   Stop counting distance beyond this [99999]\n"
      "URL\n  %s\n"
  };
  fprintf(out, str, EXENAME, cpus, GITHUB_URL);
  exit(retcode);
}

int main(int argc, char* argv[])
{
  int opt, quiet = 0, csv = 0, corner = 1, allchars = 0, keepcase = 0;
  int moltenheader = 0, molten = 0, maxdiff = 9999999;
  int lower = 0;

  while ((opt = getopt(argc, argv, "hj:qcakmtbx:vL")) != -1) {
    switch (opt) {
      case 'h': show_help(EXIT_SUCCESS); break;
      case 'j': cpus = atoi(optarg); break;
      case 'q': quiet = 1; break;
      case 'c': csv = 1; break;
      case 'a': allchars = 1; break;
      case 'k': keepcase = 1; break;
      case 'm': molten = 1; break;
      case 't': moltenheader = 1; break;
      case 'b': corner = 0; break;
      case 'L': lower = 1; break;
      case 'x': maxdiff = atol(optarg); break;
      case 'v': printf("%s %s\n", EXENAME, VERSION); exit(EXIT_SUCCESS);
      default: show_help(EXIT_FAILURE);
    }
  }
  if (optind >= argc) {
    show_help(EXIT_FAILURE);
    return 0;
  }
  const char* fasta = argv[optind];
  g_prof = getenv("SWIFTSNP_PROFILE") != NULL;
  g_t0 = g_tl = now_s();

  if (!quiet) fprintf(stderr, "This is %s %s (swiftsnp)\n", EXENAME, VERSION);
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN); if (ncpu < 1) ncpu = 1;
  if (cpus < 1) cpus = 1;
  if (cpus > ncpu) cpus = (int)ncpu;      // more threads than CPUs only slows down; output is identical
  if (!quiet) fprintf(stderr, "Will use %d threads.\n", cpus);
  omp_set_num_threads(cpus);

  for (int c = 0; c < 256; c++) IS_SPACE[c] = (c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r');
  // transform table (applied to raw bytes of fast-path records)
  unsigned char TR[256];
  for (int c = 0; c < 256; c++) {
    int u = keepcase ? c : ((c >= 'a' && c <= 'z') ? c - 32 : c);
    if (!allchars && u != 'A' && u != 'C' && u != 'G' && u != 'T') u = IGNORE_CHAR;
    TR[c] = (unsigned char)u;
  }

  // ---- 1. load
  inbuf_t in;
  int bail = read_input(fasta, &in);
  prof("read");
  if (!bail) {
#if defined(__AVX512BW__)
    bail = fast_structure(in.p, in.n, !allchars);   // layout prediction is verified by the SIMD scan
#else
    bail = fast_structure(in.p, in.n, 0);
#endif
    // bytes the fast path does not model inside sequence content: NUL (strcpy truncation) and CR (kseq strips
    // a trailing CR from the accumulated sequence). Checked during the scan below.
  }
  prof("structure");
  size_t N, L;
  size_t V = 0;
  size_t* cols = NULL;
  size_t* roff = NULL;     // raw offsets of informative columns in the reference layout (4-letter path)
  size_t pg_n = 0; size_t* pg_k = NULL; unsigned char** pg_x = NULL;   // pre-gathered column bytes per raw block
  size_t ROFF_W = 0;
  int T = cpus;

  // ---- 2. scan for informative columns (retry with the reference loader if the scan finds CR/NUL)
  for (int attempt = 0; attempt < 3; attempt++) {
    if (bail) {
      free_recs();
      load_reference(fasta, keepcase, allchars, quiet);
    }
    N = RN; L = SEQLEN;
#if defined(__AVX512BW__)
    if (!PRE_TRANSFORMED && !allchars) {
      // 4-letter modes: OR a one-hot code (A=1 C=2 G=4 T=8, ignored=0) per column; a column is informative iff
      // two different bits end up set. Records sharing the reference layout (line width Wref) are scanned as
      // one contiguous raw byte range each (newlines included), in 64-byte chunks against a raw-indexed `seen`
      // array; expected newline / line-start bitmaps verify the predicted layout on the fly.
      int badflag = 0, misflag = 0;
      size_t Wref = R[0].W ? R[0].W : (L ? L : 1);
      size_t RL = L ? L + (L - 1) / Wref : 0;
      size_t NQ = (RL + 63) / 64;
      uint64_t* expnl = xcalloc(NQ + 1, sizeof(uint64_t));
      uint64_t* expst = xcalloc(NQ + 1, sizeof(uint64_t));
      for (size_t o = Wref; o < RL; o += Wref + 1) {
        expnl[o >> 6] |= 1ULL << (o & 63);
        if (o + 1 < RL) expst[(o + 1) >> 6] |= 1ULL << ((o + 1) & 63);
      }
      size_t* grpA = xmalloc(N * sizeof(size_t)); size_t nA = 0;
      size_t* grpB = xmalloc(N * sizeof(size_t)); size_t nB = 0;
      for (size_t k = 0; k < N; k++) {
        size_t We = R[k].W ? R[k].W : (L ? L : 1);
        if (We == Wref) grpA[nA++] = k; else grpB[nB++] = k;
      }
      size_t RLp = NQ * 64 + 64;
      // long alignments use the block-parallel scan (threshold lowered by a test knob so the fuzz tests reach it)
      size_t blockmin = 262144;
      { const char* e = getenv("SWIFTSNP_BLOCKMIN"); if (e) blockmin = (size_t)atoll(e); }
      int blockpar = RL >= (size_t)T * blockmin && T > 1;
      unsigned char* seen = NULL;
      if (blockpar) {
        // long alignments: threads own disjoint raw blocks, all records inside each block. When every record
        // shares the layout (nB == 0), a block's informative columns are final as soon as all records have been
        // scanned, so they are listed right there and their bytes gathered while the block's record slices are
        // still cached. Blocks are 64 KB, or smaller (>= 16 KB) when that lets all N slices of a block fit in
        // ~1 MB of L2 (smaller slices than 16 KB cost more in streaming than the gather gains). Measured on the
        // dev sets: 8-16 KB blocks help the gather when N is small, 32-64 KB blocks are best for larger N.
        seen = big_alloc(RLp);
        int pregather = nB == 0;
        size_t BQ = 1024;                                      // chunks of 64 bytes
        if (pregather) {
          size_t bq = ((1u << 20) / (N ? N : 1)) / 64 / 64 * 64;
          if (bq >= 256 && bq < 1024) BQ = bq;
        }
        size_t nblk = (NQ + BQ - 1) / BQ;
        if (pregather) { pg_n = nblk; pg_k = xcalloc(nblk, sizeof(size_t)); pg_x = xcalloc(nblk, sizeof(unsigned char*)); }
#pragma omp parallel reduction(|:badflag, misflag)
        {
          size_t* ob = pregather ? xmalloc(BQ * 64 * sizeof(size_t)) : NULL;
#pragma omp for schedule(dynamic, 1)
          for (size_t blk = 0; blk < nblk; blk++) {
            size_t q0 = blk * BQ, q1 = q0 + BQ < NQ ? q0 + BQ : NQ;
            unsigned fl = 0;
            for (size_t a = 0; a < nA; a++) fl |= scan4_raw(R[grpA[a]].seq, q0, q1, RL, seen, expnl, expst, keepcase);
            badflag |= (fl & 1) != 0; misflag |= (fl & 2) != 0;
            if (!pregather || fl) continue;
            size_t o0 = q0 * 64, o1 = q1 * 64 < RL ? q1 * 64 : RL;
            size_t k = informative_offsets(seen, o0, o1, ob);
            pg_k[blk] = k;
            if (!k) continue;
            unsigned char* X = xmalloc(nA * k);
            for (size_t a = 0; a < nA; a++) {
              const unsigned char* rs = R[grpA[a]].seq; unsigned char* xr = X + a * k;
              for (size_t i = 0; i < k; i++) xr[i] = rs[ob[i]];
            }
            pg_x[blk] = X;
          }
          free(ob);
        }
      } else {
        // short alignments, many records: threads take records, private seen arrays, OR-merged
        seen = big_alloc(RLp * (size_t)T);
#pragma omp parallel num_threads(T) reduction(|:badflag, misflag)
        {
          unsigned char* sn = seen + (size_t)omp_get_thread_num() * RLp;
          unsigned fl = 0;
#pragma omp for schedule(dynamic, 4)
          for (size_t a = 0; a < nA; a++) fl |= scan4_raw(R[grpA[a]].seq, 0, NQ, RL, sn, expnl, expst, keepcase);
          badflag |= (fl & 1) != 0; misflag |= (fl & 2) != 0;
        }
        for (int t = 1; t < T; t++) {
          unsigned char* sn = seen + (size_t)t * RLp;
          for (size_t o = 0; o < RLp; o++) seen[o] |= sn[o];
        }
      }
      // records with a different layout: per-line segments into a column-indexed array
      unsigned char* seenB = NULL;
      if (nB) {
        seenB = xcalloc(L * (size_t)T + 64, 1);
#pragma omp parallel num_threads(T) reduction(|:badflag, misflag)
        {
          unsigned char* sn = seenB + (size_t)omp_get_thread_num() * L;
          unsigned fl = 0; int mis = 0;
#pragma omp for schedule(dynamic, 1)
          for (size_t bb = 0; bb < nB; bb++) {
            const rec_t* r = &R[grpB[bb]];
            size_t W = r->W ? r->W : (L ? L : 1);
            size_t li = 0;
            for (size_t c0 = 0; c0 < L; c0 += W, li++) {
              const unsigned char* sq = r->W ? r->seq + li * (W + 1) : r->seq + c0;
              size_t cnt = L - c0 < W ? L - c0 : W;
              fl |= scan_onehot(sq, cnt, sn + c0, keepcase);
              if (r->W) {
                if (li > 0 && (sq[0] == '>' || sq[0] == '@' || sq[0] == '+')) mis = 1;
                if (c0 + W < L && sq[W] != '\n') mis = 1;
              }
            }
          }
          badflag |= (fl & 1) != 0;
          misflag |= mis | ((fl & 2) != 0);
        }
        for (int t = 1; t < T; t++) for (size_t c = 0; c < L; c++) seenB[c] |= seenB[(size_t)t * L + c];
      }
      free(expnl); free(expst); free(grpA); free(grpB);
      if ((misflag || badflag) && !bail) {
        munmap(seen, hp_round(RLp * (blockpar ? 1 : (size_t)T))); free(seenB);
        for (size_t b = 0; b < pg_n; b++) free(pg_x[b]);
        free(pg_x); free(pg_k); pg_x = NULL; pg_k = NULL; pg_n = 0;
        if (misflag) {
          // a predicted layout was wrong: re-parse the structure line by line and scan again (once; a second
          // inconsistency, which should be impossible, goes to the reference loader)
          free_recs();
          if (attempt == 0) bail = fast_structure(in.p, in.n, 0); else bail = 1;
          prof("reparse");
        } else bail = 1;
        continue;
      }
      V = 0;
      if (pg_n) {
        // listed and gathered per block during the scan
        for (size_t b = 0; b < pg_n; b++) V += pg_k[b];
      } else if (!seenB) {
        // informative columns: raw offsets in the reference layout (newline bytes never have a code), then
        // their column indices
        roff = xmalloc((RL ? RL : 1) * sizeof(size_t));
        V = informative_offsets(seen, 0, RL, roff);
        cols = xmalloc((V ? V : 1) * sizeof(size_t));
        cols_from_roff(roff, V, Wref + 1, cols);
      } else {
        // informative columns, walked line by line in the reference layout
        cols = xmalloc((L ? L : 1) * sizeof(size_t));
        roff = xmalloc((L ? L : 1) * sizeof(size_t));
        for (size_t c0 = 0, o0 = 0; c0 < L; c0 += Wref, o0 += Wref + 1) {
          size_t cnt = L - c0 < Wref ? L - c0 : Wref;
          for (size_t k = 0; k < cnt; k++) {
            unsigned char sv = seen[o0 + k];
            if (seenB) sv |= seenB[c0 + k];
            if (sv & (sv - 1)) { cols[V] = c0 + k; roff[V] = o0 + k; V++; }
          }
        }
      }
      ROFF_W = Wref;
      munmap(seen, hp_round(RLp * (blockpar ? 1 : (size_t)T)));
      free(seenB);
      break;
    }
#endif
    const unsigned char* tr = PRE_TRANSFORMED ? NULL : TR;
    int badflag = 0;
    unsigned char* first = xmalloc(L * (size_t)T);
    unsigned char* var = xcalloc(L * (size_t)T, 1);
    memset(first, IGNORE_CHAR, L * (size_t)T);
#pragma omp parallel num_threads(T) reduction(|:badflag)
    {
      int t = omp_get_thread_num();
      unsigned char* f = first + (size_t)t * L;
      unsigned char* v = var + (size_t)t * L;
#pragma omp for schedule(dynamic, 1)
      for (size_t k = 0; k < N; k++) {
        const rec_t* r = &R[k];
        size_t W = r->W ? r->W : (L ? L : 1);
        for (size_t c0 = 0; c0 < L; c0 += W) {
          const unsigned char* s = r->W ? r->seq + (c0 / W) * (W + 1) : r->seq + c0;
          size_t cnt = L - c0 < W ? L - c0 : W;
          for (size_t i = 0; i < cnt; i++) {
            unsigned char raw = s[i];
            unsigned char b = tr ? tr[raw] : raw;
            if (tr) badflag |= (raw == 0) | (raw == '\r');
            if (b != (unsigned char)IGNORE_CHAR) {
              unsigned char fc = f[c0 + i];
              if (fc == (unsigned char)IGNORE_CHAR) f[c0 + i] = b;
              else v[c0 + i] |= (fc != b);
            }
          }
        }
      }
    }
    if (badflag && !bail) { free(first); free(var); bail = 1; continue; }
    // merge per-thread states
    for (int t = 1; t < T; t++) {
      unsigned char* f = first + (size_t)t * L; unsigned char* v = var + (size_t)t * L;
      for (size_t c = 0; c < L; c++) {
        var[c] |= v[c];
        if (f[c] != (unsigned char)IGNORE_CHAR) {
          if (first[c] == (unsigned char)IGNORE_CHAR) first[c] = f[c];
          else if (first[c] != f[c]) var[c] = 1;
        }
      }
    }
    for (size_t c = 0; c < L; c++) V += var[c];
    cols = xmalloc(V * sizeof(size_t));
    V = 0;
    for (size_t c = 0; c < L; c++) if (var[c]) cols[V++] = c;
    free(first); free(var);
    break;
  }
  if (!quiet && !PRE_TRANSFORMED) fprintf(stderr, "Read %zu sequences of length %zu\n", N, L);
  if (!quiet) fprintf(stderr, "%zu of %zu columns are informative\n", V, L);
  prof("scan");

  // ---- 3+4. gather informative columns and encode them as bitplanes
  size_t W64 = (V + 63) / 64;
  size_t Vp = W64 * 64;
  size_t G = (N + 7) / 8;          // groups of 8 sequences; lane = sequence within group
  int B = 0;
  size_t P = 1;
  uint64_t* enc = NULL;
  size_t enc_words = 0;
#if defined(__AVX512BW__)
  int fast4 = !PRE_TRANSFORMED && !allchars;
#else
  int fast4 = 0;
#endif
  size_t cap = (size_t)maxdiff;   // int -> size_t exactly as the reference passes it to distance()
  // distances never exceed V (informative columns), so 16-bit storage is exact whenever V <= 65535
  int d16 = V <= 65535;
  size_t esz = d16 ? 2 : 4;
  // Very large N: the N x N matrix would not fit comfortably in memory (the reference needs none), so rows are
  // computed per output chunk instead (full rows, no symmetry reuse) and D is never allocated.
  long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE);
  double ram = pages > 0 && psz > 0 ? (double)pages * (double)psz : 8e9;
  double dlimit = ram / 4 < 4e9 ? ram / 4 : 4e9;
  { const char* e = getenv("SWIFTSNP_DLIMIT"); if (e) dlimit = atof(e); }   // test knob
  int stream = (double)N * (double)N * (double)esz > dlimit;
#if defined(__AVX512BW__)
  // the split pays ~1-2 ms for transposes and lists; only worth it when the plain kernel is expensive
  // (estimate: ~7.4 uops per pair and 512-column word; 2e7 uops is ~2 ms on one core)
  double kern_uops = (double)N * (double)N / 2.0 * (double)V / 512.0 * 7.4;
  int split = fast4 && V > 0 && N >= 128 && !stream && kern_uops > 2e7;
  { const char* e = getenv("SWIFTSNP_SPLIT"); if (e) split = atoi(e) ? (fast4 && V > 0 && !stream) : 0; }   // test knob
  // 16-bit distances: the split runs row-streamed (final rows per output chunk, no N x N matrix)
  int rsmode = split && d16;
  { const char* e = getenv("SWIFTSNP_RS"); if (e && !atoi(e)) rsmode = 0; }   // test knob: matrix mode
#else
  int split = 0, rsmode = 0;
#endif
  // The row-streamed split builds its column planes straight from the input (split_build_fused)
  int rs_fused = rsmode;
  { const char* e = getenv("SWIFTSNP_FUSE"); if (e && !atoi(e)) rs_fused = 0; }   // test knob: unfused path
  uint64_t* SP_col = NULL;
#if !defined(__AVX512BW__)
  (void)rs_fused; (void)SP_col;
#endif
#if defined(__AVX512BW__)
  if (fast4) {
    // 4-letter modes: valid = byte is A/C/G/T after the case rule; code planes = bits 1 and 2 of the byte
    // (A 00, C 10, G 11, T 01). Raw bytes of 64 informative columns are gathered into a small buffer through a
    // precomputed offset table (no division per column), then turned into 3 plane words with 64-byte ops.
    B = 2; P = 3;
    enc_words = G * W64 * P * 8 + 8;
    if (!rs_fused) {
      enc = big_alloc(enc_words * sizeof(uint64_t));
      if (!enc) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
    }
    size_t Wref = ROFF_W;
    size_t* offW = roff;
    // Dense alignments (>= 1 informative column per 8 raw bytes, e.g. SNP-only files): encode whole 64-byte raw
    // chunks and squeeze out the non-informative positions with pext, appending to the plane bitstreams.
    size_t RLr = L ? L + (L - 1) / Wref : 0, NQr = (RLr + 63) / 64;
    int dense = V && V * 8 >= RLr && !pg_n;
    uint64_t* infbits = NULL;
    if (dense) {
      infbits = xcalloc(NQr + 1, sizeof(uint64_t));
      for (size_t v = 0; v < V; v++) infbits[offW[v] >> 6] |= 1ULL << (offW[v] & 63);
    }
    genc_t gctx = {R, V, W64, L, Wref, RLr, NQr, offW, cols, infbits, dense, keepcase, pg_n, pg_k, pg_x};
    if (rs_fused) {
      // row-streamed split: the encoding is built per 64-sequence block straight into column-major planes
      // (split_build_fused), so the sequence-major array is never allocated
      enc = NULL; enc_words = 0;
      SP_col = split_build_fused(&gctx, N);
    } else {
#pragma omp parallel
      {
        unsigned char* row = pg_n ? xmalloc(V + 64) : NULL;
#pragma omp for schedule(dynamic, 8)
        for (size_t k = 0; k < N; k++) enc_seq(&gctx, k, enc + (k / 8) * W64 * P * 8 + (k % 8), row);
        free(row);
      }
    }
    // pre-gathered slices are not freed (the process ends soon; 180+ frees cost ~0.7 ms on diverse_full through
    // heap trimming)
    pg_x = NULL; pg_k = NULL;
    free(roff); free(cols); free(infbits);
    prof(rs_fused ? "sp:enc+transp" : "gather+enc");
    // The sequence bytes of a mapped input are not read again (names re-fault on demand). Dropping the mapping's
    // page-table entries now, half per thread, costs ~1.5-2.8 ms on a 200 MB file; leaving them to the exit
    // teardown costs ~3.9 ms on one CPU (measured on outbreak_full's mapping).
    if (in.mapped == 1 && in.n >= (64u << 20)) {
      size_t h = (in.n / 2) & ~(size_t)4095;
#pragma omp parallel num_threads(2)
      {
        if (omp_get_thread_num() == 0) madvise(in.p, h, MADV_DONTNEED);
        else madvise(in.p + h, in.n - h, MADV_DONTNEED);
      }
      prof("unmap-input");
    }
  } else
#endif
  {
    // generic path: transformed bytes, N rows of Vp bytes padded with '.'
    unsigned char* X = big_alloc(N * (Vp ? Vp : 1) + 64);
    if (!X) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
    unsigned char present[256] = {0};
  #pragma omp parallel
    {
      unsigned char pr[256] = {0};
  #pragma omp for schedule(dynamic, 16)
      for (size_t k = 0; k < N; k++) {
        unsigned char* x = X + k * Vp;
        const rec_t* r = &R[k];
        if (PRE_TRANSFORMED) for (size_t v = 0; v < V; v++) x[v] = r->seq[cols[v]];
        else for (size_t v = 0; v < V; v++) x[v] = TR[rec_byte(r, cols[v])];
        memset(x + V, IGNORE_CHAR, Vp - V);
        for (size_t v = 0; v < V; v++) pr[x[v]] = 1;
      }
  #pragma omp critical
      for (int c = 0; c < 256; c++) present[c] |= pr[c];
    }
    free(cols);
    prof("gather");

    // ---- 4. encode. Symbols are compared through a minimal subset of their own bit positions that still tells
    // every present symbol apart (ACGT: bits 1 and 2), so each code plane is one byte-test mask per 64 columns.
    present[(unsigned char)IGNORE_CHAR] = 0;
    int syms[256], K = 0;
    for (int c = 0; c < 256; c++) if (present[c]) syms[K++] = c;
    int bits[8];
    if (K >= 2) {
      int best = -1;
      for (int nb = 1; nb <= 8 && best < 0; nb++)
        for (int m = 1; m < 256 && best < 0; m++) {
          if (__builtin_popcount(m) != nb) continue;
          unsigned char seen[256] = {0}; int ok = 1;
          for (int q = 0; q < K && ok; q++) { int pv = syms[q] & m; if (seen[pv]) ok = 0; seen[pv] = 1; }
          if (ok) best = m;
        }
      for (int bt = 0; bt < 8; bt++) if (best >> bt & 1) bits[B++] = bt;
    }
    P = (size_t)B + 1;
    // layout: enc[((g * W64 + w) * P + p) * 8 + lane]; plane B is the valid plane
    enc_words = G * W64 * P * 8 + 8;
    enc = big_alloc(enc_words * sizeof(uint64_t));
    if (!enc) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
  #pragma omp parallel for schedule(dynamic, 16)
    for (size_t k = 0; k < N; k++) {
      const unsigned char* x = X + k * Vp;
      uint64_t* e = enc + (k / 8) * W64 * P * 8 + (k % 8);
      for (size_t w = 0; w < W64; w++) {
        uint64_t* ew = e + w * P * 8;
  #if defined(__AVX512BW__)
        __m512i xv = _mm512_loadu_si512((const void*)(x + w * 64));
        for (int pl = 0; pl < B; pl++) ew[pl * 8] = _mm512_test_epi8_mask(xv, _mm512_set1_epi8((char)(1 << bits[pl])));
        ew[B * 8] = _mm512_cmpneq_epi8_mask(xv, _mm512_set1_epi8(IGNORE_CHAR));
  #else
        uint64_t pw[9] = {0};
        for (int q = 0; q < 64; q++) {
          unsigned char c = x[w * 64 + q];
          for (int pl = 0; pl < B; pl++) pw[pl] |= (uint64_t)((c >> bits[pl]) & 1) << q;
          pw[B] |= (uint64_t)(c != (unsigned char)IGNORE_CHAR) << q;
        }
        for (int pl = 0; pl <= B; pl++) ew[pl * 8] = pw[pl];
  #endif
      }
    }
    munmap(X, hp_round(N * (Vp ? Vp : 1) + 64));
    prof("encode");
  }

  // ---- 5. pairwise kernel: a block of rows against groups of 8 sequences at a time (one sequence per 64-bit
  // lane), upper triangle only, row-wise stores; then a tiled mirror.
  void* D = (stream || rsmode) ? NULL : big_alloc(N * N * esz + 64);
  if (!D && !stream && !rsmode) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
#if defined(__AVX512BW__)
  split_t SP; memset(&SP, 0, sizeof SP);
#endif
  if (stream) {
    // nothing here: rows are produced in the output loop
  } else if (B == 0) {
    memset(D, 0, N * N * esz);
  } else if (rsmode) {
#if defined(__AVX512BW__)
    split_prepare(&SP, enc, W64, G, N, V, 1, SP_col);
    if (SP_col) rs_lists_col(&SP); else rs_lists(&SP);
    // rows are produced in the output loop
#endif
  } else if (split) {
#if defined(__AVX512BW__)
    split_prepare(&SP, enc, W64, G, N, V, 0, NULL);
    split_rows_D(&SP, d16, D);
    prof("sparse");
    if (SP.encd) {
      const size_t RB = KERNEL_ROWS;
      size_t nrb = (N + RB - 1) / RB;
#pragma omp parallel for schedule(dynamic, 1)
      for (size_t rb = 0; rb < nrb; rb++) {
        size_t j0 = rb * RB;
        size_t nr = N - j0 < RB ? N - j0 : RB;
        kernel_block(SP.encd, SP.W64d, P, B, G, N, j0, nr, (size_t)-1, D, d16, 1, j0 / 8, 0, N, 0);
      }
    }
    split_free(&SP);
    prof("kernel");
    // combine: D[i][j] = D[j][i] = min(D[i][j] + D[j][i], cap) for i < j
    const size_t TB = 64;
    size_t nb = (N + TB - 1) / TB;
#define COMBINE(TYPE) do { TYPE* DD = (TYPE*)D; \
    _Pragma("omp parallel for schedule(dynamic, 1)") \
    for (size_t bi = 0; bi < nb; bi++) { \
      for (size_t bj = 0; bj <= bi; bj++) { \
        size_t i0 = bi * TB, i1 = i0 + TB < N ? i0 + TB : N; \
        size_t j0 = bj * TB, j1 = j0 + TB < N ? j0 + TB : N; \
        for (size_t i = i0; i < i1; i++) { \
          size_t jend = j1 < i ? j1 : i; \
          for (size_t j = j0; j < jend; j++) { \
            size_t v = (size_t)DD[i * N + j] + (size_t)DD[j * N + i]; \
            TYPE t = (TYPE)(v < cap ? v : cap); \
            DD[i * N + j] = t; DD[j * N + i] = t; \
          } \
        } \
      } \
    } } while (0)
    if (d16) combine16((uint16_t*)D, N, cap); else COMBINE(uint32_t);
#undef COMBINE
#endif
  } else {
    const size_t RB = KERNEL_ROWS;
    size_t nrb = (N + RB - 1) / RB;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t rb = 0; rb < nrb; rb++) {
      size_t j0 = rb * RB;
      size_t nr = N - j0 < RB ? N - j0 : RB;
      kernel_block(enc, W64, P, B, G, N, j0, nr, cap, D, d16, 0, j0 / 8, 0, N, 0);
    }
    prof("kernel");
    const size_t TB = 64;
    size_t nb = (N + TB - 1) / TB;
#define MIRROR(TYPE) do { TYPE* DD = (TYPE*)D; \
    _Pragma("omp parallel for schedule(dynamic, 1)") \
    for (size_t bi = 0; bi < nb; bi++) { \
      for (size_t bj = 0; bj <= bi; bj++) { \
        size_t i0 = bi * TB, i1 = i0 + TB < N ? i0 + TB : N; \
        size_t j0 = bj * TB, j1 = j0 + TB < N ? j0 + TB : N; \
        for (size_t i = i0; i < i1; i++) { \
          size_t jend = j1 < i ? j1 : i; \
          for (size_t j = j0; j < jend; j++) DD[i * N + j] = DD[j * N + i]; \
        } \
      } \
    } } while (0)
    if (d16) MIRROR(uint16_t); else MIRROR(uint32_t);
#undef MIRROR
  }
  if (!stream && !rsmode) munmap(enc, hp_round(enc_words * sizeof(uint64_t)));
  prof("mirror");

  // ---- 6. output
  char sep = csv ? ',' : '\t';
  int dig = 1;
  { size_t mx = V < cap ? V : cap;       // an upper bound on every distance (for buffer sizing only)
    while (mx >= 10) { mx /= 10; dig++; } }
  size_t maxname = 0;
  for (size_t k = 0; k < N; k++) if (R[k].namelen > maxname) maxname = R[k].namelen;
  {
    size_t hl = 64;
    for (size_t k = 0; k < N; k++) hl += R[k].namelen + 1;
    char* h = xmalloc(hl); char* o = h;
    if (molten) {
      if (moltenheader) o += sprintf(o, "sequence_1%csequence_2%cdistance\n", sep, sep);
    } else {
      if (corner) o += sprintf(o, "%s %s", EXENAME, VERSION);
      for (size_t k = 0; k < N; k++) { *o++ = sep; memcpy(o, R[k].name, R[k].namelen); o += R[k].namelen; }
      *o++ = '\n';
    }
    write_all(h, (size_t)(o - h));
    free(h);
  }
  // "<sep><digits>" for 0..999 as one 4-byte store + a length
  for (uint32_t v = 0; v < 1000; v++) {
    char t[4] = {sep, 0, 0, 0}; char* e = put_u32(t + 1, v);
    uint32_t w; memcpy(&w, t, 4);
    SEPTAB[v] = (uint64_t)w | (uint64_t)(e - t) << 56;
  }
  for (uint32_t a = 0; a < 100; a++)
    for (uint32_t b = 0; b < 100; b++) {
      uint64_t la = SEPTAB[a] >> 56, lb = SEPTAB[b] >> 56;
      uint64_t ta = SEPTAB[a] & 0xFFFFFFFFull, tb = SEPTAB[b] & 0xFFFFFFFFull;
      PAIRTAB[a << 7 | b] = ta | tb << (8 * la) | (la + lb) << 56;
    }
  fmt_init();
  // exact-enough per-row byte bounds (a single very long name must not inflate every row's buffer)
  size_t sumname = 0;
  for (size_t k = 0; k < N; k++) sumname += R[k].namelen;
#define ROW_BOUND(j) (molten ? N * (R[(j)].namelen + 3 + (size_t)dig) + sumname : R[(j)].namelen + N * (1 + (size_t)dig) + 1)
  size_t rowmax = molten ? N * (maxname + 3 + dig) + sumname : maxname + N * (1 + (size_t)dig) + 1;
  size_t rowtyp = molten ? N * (3 + dig) + 2 * sumname : N * (1 + (size_t)dig) + 1;   // typical row (for chunking)
  size_t rows_per_chunk = (1u << 20) / (rowtyp + 1); if (rows_per_chunk < 1) rows_per_chunk = 1;
  if (rsmode) rows_per_chunk = 64;          // row-streamed split: one 64-sequence block per chunk
  (void)rowmax;
  size_t nchunks = (N + rows_per_chunk - 1) / rows_per_chunk;
  char** tbuf = xcalloc((size_t)T, sizeof(char*));
  size_t* tcap = xcalloc((size_t)T, sizeof(size_t));
  void** sbuf = xcalloc((size_t)T, sizeof(void*));
  uint16_t** tileT = xcalloc((size_t)T, sizeof(uint16_t*));
  uint64_t** d4buf = xcalloc((size_t)T, sizeof(uint64_t*));
#pragma omp parallel for ordered schedule(static, 1) num_threads(T)
  for (size_t ch = 0; ch < nchunks; ch++) {
    int t = omp_get_thread_num();
    size_t r0b = ch * rows_per_chunk, r1b = r0b + rows_per_chunk; if (r1b > N) r1b = N;
    size_t need = 64;
    for (size_t j = r0b; j < r1b; j++) need += ROW_BOUND(j);
    if (tcap[t] < need) {      // per-thread text buffer (huge pages when >= 1 MB: ~1 fault instead of ~300)
      if (tbuf[t]) munmap(tbuf[t], hp_round(tcap[t]));
      size_t nc = need + need / 4;
      tbuf[t] = big_alloc(nc); tcap[t] = nc;
      if (!tbuf[t]) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
    }
    char* o = tbuf[t];
    size_t r0 = ch * rows_per_chunk, r1 = r0 + rows_per_chunk; if (r1 > N) r1 = N;
    if (stream) {
      if (!sbuf[t]) sbuf[t] = xmalloc(rows_per_chunk * N * esz + 64);
      if (B == 0 || V == 0) memset(sbuf[t], 0, (r1 - r0) * N * esz);
      else
        for (size_t j0 = r0; j0 < r1; j0 += KERNEL_ROWS) {
          size_t nr = r1 - j0 < KERNEL_ROWS ? r1 - j0 : KERNEL_ROWS;
          kernel_block(enc, W64, P, B, G, N, j0, nr, cap, sbuf[t], d16, 0, 0, r0, N, 0);
        }
    }
#if defined(__AVX512BW__)
    else if (rsmode) {
      size_t NP = (N + 63) / 64 * 64;
      if (!sbuf[t]) sbuf[t] = big_alloc(64 * N * sizeof(uint16_t) + 64);
      if (!tileT[t]) tileT[t] = big_alloc(64 * 64 * sizeof(uint16_t));      // page-aligned
      if (!d4buf[t]) d4buf[t] = big_alloc(4 * SP.Vp * sizeof(uint64_t) + 64);
      if (!sbuf[t] || !tileT[t]) { fprintf(stderr, "ERROR: out of memory\n"); exit(EXIT_FAILURE); }
      rs_rows(&SP, r0, r1, (uint16_t*)sbuf[t], tileT[t], d4buf[t], cap);
    }
#endif
#define EMIT_ROWS(TYPE) \
    for (size_t j = r0; j < r1; j++) { \
      const TYPE* row = (stream || rsmode) ? (const TYPE*)sbuf[t] + (j - r0) * N : (const TYPE*)D + j * N; \
      if (molten) { \
        size_t start = lower ? j : 0; \
        for (size_t i = start; i < N; i++) { \
          memcpy(o, R[j].name, R[j].namelen); o += R[j].namelen; *o++ = sep; \
          memcpy(o, R[i].name, R[i].namelen); o += R[i].namelen; \
          uint32_t v = row[i]; \
          if (v < 1000) { uint64_t tv = SEPTAB[v]; memcpy(o, &tv, 8); o += tv >> 56; } \
          else { *o++ = sep; o = put_u32(o, v); } \
          *o++ = '\n'; \
        } \
      } else { \
        memcpy(o, R[j].name, R[j].namelen); o += R[j].namelen; \
        size_t end = lower ? j + 1 : N; \
        for (size_t i = 0; i < end; i++) { \
          uint32_t v = row[i]; \
          if (v < 1000) { uint64_t tv = SEPTAB[v]; memcpy(o, &tv, 8); o += tv >> 56; } \
          else { *o++ = sep; o = put_u32(o, v); } \
        } \
        *o++ = '\n'; \
      } \
    }
    if (d16 && !molten) {
      // distances mostly < 100 (large N, SNP-like data): two values per table load and store. The pair path is
      // used for a chunk when its first row has < 1/16 values >= 100 (a failed pair test is a mispredicted
      // branch; core_snps, half 3-digit, was 20% slower with pairs always on).
      int pairs = 0;
      {
        const uint16_t* row0 = (stream || rsmode) ? (const uint16_t*)sbuf[t] : (const uint16_t*)D + r0 * N;
        size_t nbig = 0;
        for (size_t i = 0; i < N; i++) nbig += row0[i] >= 100;
        pairs = nbig * 16 < N;
      }
#define PAIR_AT(q) { \
            uint32_t w; memcpy(&w, row + (q), 4); \
            uint64_t tv = (w & 0xFF80FF80u) ? 0 : PAIRTAB[(w & 0x7Fu) << 7 | (w >> 16)]; \
            if (tv) { memcpy(o, &tv, 8); o += tv >> 56; } \
            else for (int h = 0; h < 2; h++) { \
              uint32_t v = row[(q) + h]; \
              if (v < 1000) { uint64_t sv = SEPTAB[v]; memcpy(o, &sv, 8); o += sv >> 56; } \
              else { *o++ = sep; o = put_u32(o, v); } \
            } }
      for (size_t j = r0; j < r1; j++) {
        const uint16_t* row = (stream || rsmode) ? (const uint16_t*)sbuf[t] + (j - r0) * N : (const uint16_t*)D + j * N;
        memcpy(o, R[j].name, R[j].namelen); o += R[j].namelen;
        size_t end = lower ? j + 1 : N, i = 0;
#if defined(__AVX512BW__) && defined(__BMI2__)
        // 32 values per step. All < 100: PAIRTAB indexes for the 16 pairs in one vector. Otherwise (< 1000): digits
        // with reciprocal multiplies, one "<sep>d2d1d0" dword per value, then per group of 4 values a pshufb that
        // drops leading zeros (table indexed by the >= 10 / >= 100 bits) and one 16-byte store. Blocks with a value
        // >= 1000 use the scalar table for those 32 values.
        for (; i + 32 <= end; i += 32) {
          __m512i v = _mm512_loadu_si512((const void*)(row + i));
          if (_mm512_cmpge_epu16_mask(v, _mm512_set1_epi16(1000))) {
            for (size_t q = i; q < i + 32; q++) {
              uint32_t x = row[q];
              if (x < 1000) { uint64_t sv = SEPTAB[x]; memcpy(o, &sv, 8); o += sv >> 56; }
              else { *o++ = sep; o = put_u32(o, x); }
            }
            continue;
          }
          uint64_t m100 = _mm512_cmpge_epu16_mask(v, _mm512_set1_epi16(100));
          if (!m100) {        // all < 100 (SNP-like large N): PAIRTAB indexes for 16 pairs, one load + store per pair
            __m512i px = _mm512_ternarylogic_epi32(_mm512_slli_epi32(v, 7), _mm512_set1_epi32(0x3F80),
                                                   _mm512_srli_epi32(v, 16), 0xEA);   // (a & b) | c
            uint32_t pidx[16] __attribute__((aligned(64)));
            _mm512_store_si512((void*)pidx, px);
            for (int k = 0; k < 16; k++) { uint64_t tv = PAIRTAB[pidx[k]]; memcpy(o, &tv, 8); o += tv >> 56; }
            continue;
          }
          uint64_t m10 = _mm512_cmpge_epu16_mask(v, _mm512_set1_epi16(10));
          __m512i d2 = _mm512_mulhi_epu16(v, _mm512_set1_epi16(656));
          __m512i r = _mm512_sub_epi16(v, _mm512_mullo_epi16(d2, _mm512_set1_epi16(100)));
          __m512i d1 = _mm512_mulhi_epu16(r, _mm512_set1_epi16(6554));
          __m512i d0 = _mm512_sub_epi16(r, _mm512_mullo_epi16(d1, _mm512_set1_epi16(10)));
          __m512i lo = _mm512_add_epi16(_mm512_slli_epi16(d2, 8), _mm512_set1_epi16((short)(0x3000 | (uint8_t)sep)));
          __m512i hi = _mm512_add_epi16(_mm512_add_epi16(d1, _mm512_slli_epi16(d0, 8)), _mm512_set1_epi16(0x3030));
          __m512i A = _mm512_unpacklo_epi16(lo, hi), Bq = _mm512_unpackhi_epi16(lo, hi);   // lane L: values 8L..8L+3 / 8L+4..8L+7
          uint64_t ix = _pdep_u64(m10, 0x0F0F0F0F0F0F0F0Full) | _pdep_u64(m100, 0xF0F0F0F0F0F0F0F0ull);
#define FMT_GRP(X, g) { unsigned q_ = (unsigned)(ix >> (8 * (g))) & 0xFFu; \
            _mm_storeu_si128((__m128i*)o, _mm_shuffle_epi8((X), _mm_load_si128((const __m128i*)FMT_SHUF[q_]))); \
            o += FMT_LEN[q_]; }
          FMT_GRP(_mm512_castsi512_si128(A), 0) FMT_GRP(_mm512_castsi512_si128(Bq), 1)
          FMT_GRP(_mm512_extracti32x4_epi32(A, 1), 2) FMT_GRP(_mm512_extracti32x4_epi32(Bq, 1), 3)
          FMT_GRP(_mm512_extracti32x4_epi32(A, 2), 4) FMT_GRP(_mm512_extracti32x4_epi32(Bq, 2), 5)
          FMT_GRP(_mm512_extracti32x4_epi32(A, 3), 6) FMT_GRP(_mm512_extracti32x4_epi32(Bq, 3), 7)
#undef FMT_GRP
        }
#endif
        if (pairs)
          for (; i + 2 <= end; i += 2) PAIR_AT(i)
        for (; i < end; i++) {
          uint32_t v = row[i];
          if (v < 1000) { uint64_t sv = SEPTAB[v]; memcpy(o, &sv, 8); o += sv >> 56; }
          else { *o++ = sep; o = put_u32(o, v); }
        }
        *o++ = '\n';
      }
    } else if (d16) { EMIT_ROWS(uint16_t) } else { EMIT_ROWS(uint32_t) }
#undef EMIT_ROWS
#undef PAIR_AT
#pragma omp ordered
    {
      double tw = g_prof ? now_s() : 0;
      write_all(tbuf[t], (size_t)(o - tbuf[t]));
      if (g_prof) g_twrite += now_s() - tw;
    }
  }
  if (g_prof) fprintf(stderr, "[prof] (time inside write(2): %.4f s, %zu chunks)\n", g_twrite, nchunks);
  prof("output");
  // no frees at exit: the process is ending (the reference also leaks nothing that matters here)
  (void)free_input;
  return 0;
}

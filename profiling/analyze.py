#!/usr/bin/env python3
"""Analyze a samply profile of the encode_st bench with inline-frame resolution.

Pipeline:
  1. Load the Firefox-profiler JSON (samply --save-only output, unsymbolicated).
  2. Resolve every unique (library, relative address) with `atos -offset -i`
     against the binary/dSYM: full inline stacks + source file:line.
  3. Demangle Rust v0 symbols with rustfilt, strip generic noise.
  4. Emit:
     - top N functions by weighted self time (innermost inline frame identity)
     - collapsed stacks (.folded) with inline frames expanded, for flamegraphs
     - source-line annotation of the hottest regions
     - a bucketed phase/category breakdown (walker / cache / BPE merge / ...)

Usage:
  python3 analyze.py TRACE.json.gz --bin path/to/encode_st-HASH [-o OUTDIR]
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Symbolication


SENTINEL = 0xFFFFFFFF0  # unmapped offset; atos echoes it back verbatim


def atos_resolve(lib_path, dsym_path, rel_addrs):
    """Resolve relative addresses to inline stacks via atos.

    Returns {addr: [(symbol, file, line), ...]} innermost-first. A sentinel
    (unmapped) address is interleaved between queries; atos echoes it back as
    a bare hex line, giving unambiguous per-address group boundaries.
    """
    if not rel_addrs:
        return {}
    obj = dsym_path if dsym_path and os.path.exists(dsym_path) else lib_path
    out = {}
    addrs = sorted(rel_addrs)
    sent_hex = hex(SENTINEL)
    CHUNK = 500
    for i in range(0, len(addrs), CHUNK):
        chunk = addrs[i : i + CHUNK]
        query = []
        for a in chunk:
            query += [hex(a), sent_hex]
        cmd = [
            "atos", "-o", obj, "-arch", "arm64", "-offset", "-i", "-fullPath",
        ] + query
        res = subprocess.run(cmd, capture_output=True, text=True)
        groups, cur = [], []
        for line in res.stdout.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line == sent_hex:
                groups.append(cur)
                cur = []
            else:
                cur.append(line)
        if len(groups) != len(chunk):
            raise RuntimeError(
                f"atos group mismatch for {lib_path}: {len(groups)} groups, "
                f"{len(chunk)} addrs"
            )
        for a, g in zip(chunk, groups):
            frames = []
            for line in g:
                m = re.match(r"^(.*?) \(in .*?\)(?: \((.*?):(\d+)\))?$", line)
                if m:
                    sym, f, ln = m.group(1), m.group(2), m.group(3)
                    frames.append((sym, f, int(ln) if ln else None))
                else:
                    frames.append((line, None, None))
            out[a] = frames if frames else [(f"0x{a:x}", None, None)]
    return out


def load_sidecar_symbols(trace_path):
    """Load samply's --unstable-presymbolicate sidecar (.syms.json).

    Returns {debug_name: sorted [(rva, size, symbol_name)]} for range lookup.
    """
    sidecar = re.sub(r"\.json(\.gz)?$", ".json.syms.json", trace_path)
    if not os.path.exists(sidecar):
        return {}
    d = json.load(open(sidecar))
    strs = d["string_table"]
    tables = {}
    for lib in d["data"]:
        entries = [
            (e["rva"], e.get("size", 0), strs[e["symbol"]])
            for e in lib["symbol_table"]
        ]
        entries.sort()
        tables[lib["debug_name"]] = entries
    return tables


def sidecar_lookup(table, addr):
    """Find the covering symbol range for addr in a sorted (rva,size,name) list."""
    import bisect

    i = bisect.bisect_right([e[0] for e in table], addr) - 1
    if i >= 0:
        rva, size, name = table[i]
        if addr < rva + max(size, 1) or size == 0:
            return name
    return None


def demangle_all(names):
    """Batch-demangle through rustfilt; returns dict mangled->demangled."""
    uniq = sorted(set(names))
    try:
        res = subprocess.run(
            ["rustfilt"], input="\n".join(uniq), capture_output=True, text=True
        )
        dem = res.stdout.split("\n")
        return dict(zip(uniq, dem))
    except FileNotFoundError:
        return {n: n for n in uniq}


def strip_generics(name):
    """Shorten demangled Rust v0 names.

    `<A as B>::f::<T>` -> `A::f`; generic argument lists are dropped, but a
    leading qualified-self `<Type ...>` is replaced by the type's last path
    segments so trait-impl methods keep their type name.
    """
    # Replace a leading <Self as Trait> / <Self> with Self's short name.
    if name.startswith("<"):
        depth, i = 0, 0
        for i, ch in enumerate(name):
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    break
        inner = name[1:i]
        inner = inner.split(" as ")[0]
        # shorten the self type itself (drop its generics, keep last segment)
        inner = strip_generics(inner) if "<" in inner else inner
        inner = inner.split("::")[-1] if "::" in inner else inner
        name = inner + name[i + 1 :]
    out = []
    depth = 0
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    s = "".join(out)
    while "::::" in s:
        s = s.replace("::::", "::")
    if s.endswith("::"):
        s = s[:-2]
    return s


# --------------------------------------------------------------------------
# Bucketing: map an inline stack (innermost-first) to a category.

CATEGORY_RULES = [
    # (predicate on (symbol, file), category) -- first match on innermost
    # frame wins; some rules look at the whole stack.
    ("pack_pretoken_key", None, "cache: key pack"),
    ("ShortPretokenCache", "pretoken_cache", "cache: probe/insert"),
    ("LongPretokenCache", "pretoken_cache", "cache: probe/insert (long)"),
    (None, "pretoken_cache.rs", "cache: probe/insert"),
    ("byte_pair_merge", None, "bpe: merge (miss fallback)"),
    ("encode_pretoken", None, "bpe: merge (miss fallback)"),
    (None, "pretokenize/fast", "pretokenizer walker"),
    (None, "pretokenize/mod.rs", "pretokenizer walker"),
    (None, "pretokenize/unicode", "pretokenizer walker"),
    ("memoized_encode", "tiktoken.rs", "encode driver (spans/output)"),
    (None, "tiktoken.rs", "encode driver (spans/output)"),
    ("memcpy", None, "libsystem memcpy/memset"),
    ("memset", None, "libsystem memcpy/memset"),
    ("_platform_", None, "libsystem memcpy/memset"),
    ("malloc", None, "malloc/free"),
    ("free", None, "malloc/free"),
    ("nanov2", None, "malloc/free"),
    ("read", None, "kernel: read/syscalls"),
    ("madvise", None, "kernel: read/syscalls"),
    ("mach_", None, "kernel: read/syscalls"),
]


def categorize(pairs_leaf_first):
    """pairs_leaf_first: flattened [(sym, file)] over the whole stack, leaf
    first, with inline frames expanded. Walk outward; the first frame that
    matches any rule decides the category, so inlined std helpers attribute
    to their nearest categorized caller."""
    for sym, f in pairs_leaf_first:
        for pat_sym, pat_file, cat in CATEGORY_RULES:
            if pat_sym and pat_sym not in sym:
                continue
            if pat_file and (not f or pat_file not in f):
                continue
            return cat
        if "load_owt_input" in sym:
            return "corpus read phase"
        if "load_hf_bpe" in sym or "load_tokenizer" in sym:
            return "tokenizer load phase"
    return "other"


PHASE_MARKERS = {
    "memoized_encode": "encode",
    "load_owt_input": "read corpus",
    "load_hf_bpe": "tokenizer load",
}


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--bin", required=True, help="path to profiled encode_st binary")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--hot-regions", type=int, default=3)
    args = ap.parse_args()

    outdir = args.outdir or os.path.splitext(args.trace)[0].replace(
        ".json", ""
    ) + "_analysis"
    os.makedirs(outdir, exist_ok=True)

    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        prof = json.load(f)
    interval_ms = prof["meta"]["interval"]
    threads = [t for t in prof["threads"] if t["samples"]["length"] > 0]
    t = max(threads, key=lambda th: th["samples"]["length"])

    strs = t["stringArray"]
    ft, fut, rt = t["frameTable"], t["funcTable"], t["resourceTable"]
    st = t["stackTable"]
    libs = prof["libs"]

    bin_abs = os.path.abspath(args.bin)
    dsym = f"{bin_abs}.dSYM/Contents/Resources/DWARF/{os.path.basename(bin_abs)}"

    # ---- unique addresses per lib -------------------------------------
    frame_lib = []  # frame index -> lib index (or None)
    frame_addr = ft["address"]
    for fi in range(ft["length"]):
        func = ft["func"][fi]
        res = fut["resource"][func]
        frame_lib.append(rt["lib"][res] if res is not None and res >= 0 else None)

    addrs_by_lib = defaultdict(set)
    for fi in range(ft["length"]):
        li, a = frame_lib[fi], frame_addr[fi]
        if li is not None and a is not None:
            addrs_by_lib[li].add(a)

    def lib_obj_path(li):
        lib = libs[li]
        p = lib.get("debugPath") or lib["path"]
        if not os.path.isabs(p):
            # relative to the recording cwd; try relative to the binary dir
            cand = os.path.normpath(
                os.path.join(os.path.dirname(bin_abs), "..", "..", "..", p)
            )
            p = cand if os.path.exists(cand) else os.path.abspath(p)
        return p

    sidecar = load_sidecar_symbols(args.trace)
    resolved = {}  # (li, addr) -> [(sym, file, line)]
    for li, addrs in addrs_by_lib.items():
        libname = libs[li]["name"]
        if os.path.basename(libs[li]["path"]) == os.path.basename(bin_abs):
            # Main binary: exact per-address inline stacks from DWARF.
            try:
                r = atos_resolve(bin_abs, dsym, addrs)
            except Exception as e:
                print(f"warn: atos failed for {bin_abs}: {e}", file=sys.stderr)
                r = {a: [(f"{libname}+{hex(a)}", None, None)] for a in addrs}
        else:
            # System dylibs (dyld shared cache): symbol names from the
            # samply presymbolicate sidecar; no inline/line info needed.
            table = sidecar.get(libname, [])
            r = {}
            for a in addrs:
                name = sidecar_lookup(table, a) if table else None
                r[a] = [(name or f"{libname}+{hex(a)}", None, None)]
        for a, frames in r.items():
            resolved[(li, a)] = frames

    # ---- demangle ------------------------------------------------------
    all_syms = [s for frames in resolved.values() for s, _, _ in frames]
    dem = demangle_all(all_syms)
    for k, frames in resolved.items():
        resolved[k] = [
            (strip_generics(dem.get(s, s) or s), f, ln) for s, f, ln in frames
        ]

    def frame_inline_stack(fi):
        li, a = frame_lib[fi], frame_addr[fi]
        if li is None or a is None:
            func = ft["func"][fi]
            return [(strs[fut["name"][func]], None, None)]
        return resolved.get((li, a), [("?", None, None)])

    # project root = worktree containing the profiled binary
    # (bin lives at <root>/target/release/deps/<bin>)
    src_root = os.path.normpath(os.path.join(os.path.dirname(bin_abs), "../../.."))

    def is_project(sym, f):
        return "gigatoken" in sym or (f is not None and f.startswith(src_root))

    # ---- walk samples ---------------------------------------------------
    samples = t["samples"]
    n = samples["length"]
    weights = samples["weight"] or [1.0] * n
    cpu_deltas = samples.get("threadCPUDelta")

    # expand each stack index once
    stack_frames_cache = {}

    def stack_frames(si):
        """stack index -> list of frame indices, leaf first."""
        if si in stack_frames_cache:
            return stack_frames_cache[si]
        chain = []
        cur = si
        while cur is not None:
            chain.append(st["frame"][cur])
            cur = st["prefix"][cur]
        stack_frames_cache[si] = chain
        return chain

    self_w = Counter()  # innermost inline identity -> weight
    line_w = Counter()  # (file, line) -> weight
    line_owner = {}
    cat_w = Counter()
    phase_w = Counter()
    folded = Counter()
    total_w = 0.0
    cpu_ms_by_phase = Counter()

    for idx in range(n):
        si = samples["stack"][idx]
        w = weights[idx]
        total_w += w
        if si is None:
            phase_w["<no stack>"] += w
            continue
        chain = stack_frames(si)  # leaf first
        # flattened (sym, file) list for the full stack, leaf first, with
        # inline frames expanded
        pairs_leaf_first = []
        for fi in chain:
            for sym, f, ln in frame_inline_stack(fi):
                pairs_leaf_first.append((sym, f))
        leaf_inl = frame_inline_stack(chain[0])
        sym0, f0, ln0 = leaf_inl[0]
        # context: nearest enclosing project frame, so std one-liners like
        # core::intrinsics::likely are reported with their real home
        ctx = ""
        if not is_project(sym0, f0):
            for sym_c, f_c in pairs_leaf_first:
                if is_project(sym_c, f_c):
                    if sym_c != sym0:
                        ctx = sym_c
                    break
        self_w[(sym0, f0, ctx)] += w
        if f0 and ln0:
            line_w[(f0, ln0)] += w
            line_owner[(f0, ln0)] = sym0
        cat = categorize(pairs_leaf_first)
        cat_w[cat] += w
        phase = "other"
        for marker, ph in PHASE_MARKERS.items():
            if any(marker in nm for nm, _ in pairs_leaf_first):
                phase = ph
                break
        phase_w[phase] += w
        if cpu_deltas and cpu_deltas[idx] is not None:
            cpu_ms_by_phase[phase] += cpu_deltas[idx] / 1000.0  # us -> ms
        folded[";".join(nm for nm, _ in reversed(pairs_leaf_first))] += w

    total_ms = total_w * interval_ms

    # ---- outputs ---------------------------------------------------------
    def pct(w):
        return 100.0 * w / total_w

    with open(os.path.join(outdir, "top_functions.txt"), "w") as f:
        f.write(
            f"trace: {args.trace}\nsamples: {int(total_w)}  "
            f"interval: {interval_ms} ms  total: {total_ms/1000:.2f} s\n"
        )
        f.write("\n== Phases (wall time attributed by stack) ==\n")
        for ph, w in phase_w.most_common():
            cpu = cpu_ms_by_phase.get(ph)
            cpu_s = f"  cpu={cpu/1000:.2f}s" if cpu else ""
            f.write(f"{pct(w):6.2f}%  {w*interval_ms/1000:7.2f}s  {ph}{cpu_s}\n")
        f.write("\n== Category buckets (weighted self time) ==\n")
        for cat, w in cat_w.most_common():
            f.write(f"{pct(w):6.2f}%  {w*interval_ms/1000:7.2f}s  {cat}\n")
        f.write(
            f"\n== Top {args.top} functions by weighted self time "
            "(inline frames resolved) ==\n"
        )
        for (sym, file, ctx), w in self_w.most_common(args.top):
            floc = f"  [{os.path.basename(file) if file else '?'}]"
            ctx_s = f"  <- {ctx}" if ctx else ""
            f.write(
                f"{pct(w):6.2f}%  {w*interval_ms/1000:7.2f}s  {sym}{floc}{ctx_s}\n"
            )

    with open(os.path.join(outdir, "collapsed.folded"), "w") as f:
        for stack, w in folded.most_common():
            f.write(f"{stack} {int(w)}\n")

    # source-line annotation for the hottest project functions
    hot_funcs = []
    for (sym, file, ctx), _w in self_w.most_common(100):
        if file and "/src/" in file and (sym, file) not in hot_funcs:
            hot_funcs.append((sym, file))
        if len(hot_funcs) >= args.hot_regions:
            break
    with open(os.path.join(outdir, "hot_lines.txt"), "w") as f:
        for sym, file in hot_funcs:
            f.write(f"\n===== {sym}  ({file}) =====\n")
            lines = [
                (ln, w)
                for (fl, ln), w in line_w.items()
                if fl == file and line_owner.get((fl, ln)) == sym
            ]
            lines.sort()
            try:
                src = open(file).read().split("\n")
            except OSError:
                src = None
            for ln, w in sorted(lines, key=lambda x: -x[1])[:15]:
                text = src[ln - 1].strip() if src and ln - 1 < len(src) else ""
                f.write(f"{pct(w):6.2f}%  {file.split('/')[-1]}:{ln:<5} {text}\n")

    print(f"total: {total_ms/1000:.2f}s profiled, {int(total_w)} samples")
    print(f"outputs in {outdir}/: top_functions.txt collapsed.folded hot_lines.txt")
    for ph, w in phase_w.most_common():
        print(f"  {pct(w):6.2f}%  {ph}")


if __name__ == "__main__":
    main()

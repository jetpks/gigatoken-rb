#!/usr/bin/env python3
"""Summarize an xctrace 'CPU Counters' (CPU Bottlenecks guided mode) trace.

Exports the MetricTable and RemarksByThread tables and aggregates them.
xctrace XML uses ref-compression: any element with id=N may later be
referenced as <tag ref="N"/>; refs must be resolved. Ratio metrics are
duration-weighted; count metrics are summed.

Usage: python3 pmu_summary.py TRACE.trace [--window START_S END_S] [--pcore-only]
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


def export_table(trace, schema, cache_dir):
    out = os.path.join(cache_dir, f"{os.path.basename(trace)}.{schema}.xml")
    if not os.path.exists(out):
        with open(out, "w") as f:
            subprocess.run(
                [
                    "xcrun", "xctrace", "export", "--input", trace, "--xpath",
                    f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
                ],
                stdout=f,
                check=True,
            )
    return out


class RefResolver:
    """Resolve xctrace export id/ref compression."""

    def __init__(self):
        self.by_id = {}

    def resolve(self, el):
        rid = el.get("ref")
        if rid is not None:
            return self.by_id[rid]
        eid = el.get("id")
        if eid is not None:
            self.by_id[eid] = el
        return el


def parse_metric_table(path, window=None, pcore_only=False):
    rr = RefResolver()
    # (metric_name, is_ratio) -> [sum_weighted_value, sum_weight, sum_value]
    agg = defaultdict(lambda: [0.0, 0.0, 0.0])
    core_types = Counter()
    for _ev, el in ET.iterparse(path, events=("end",)):
        if el.tag != "row":
            continue
        kids = list(el)
        # row layout: start-time, duration, string(pmi-event),
        # string(metric-name), thread, process, fixed-decimal, core,
        # boolean(is-ratio), [markdown-text]
        try:
            start = rr.resolve(kids[0])
            dur = rr.resolve(kids[1])
            pmi = rr.resolve(kids[2])
            name_el = rr.resolve(kids[3])
            val_el = rr.resolve(kids[6])
            core_el = rr.resolve(kids[7])
            ratio_el = rr.resolve(kids[8])
            # register any remaining ids (markdown-text etc.)
            for k in kids[9:]:
                rr.resolve(k)
            for sub in el.iter():
                if sub is not el:
                    rr.resolve(sub)
        except (IndexError, KeyError):
            el.clear()
            continue
        t_ns = int(start.text or start.get("fmt", "0").replace(",", "") or 0)
        d_ns = int(dur.text or 0)
        name = name_el.get("fmt") or (name_el.text or "?")
        val = float(val_el.text or 0.0)
        is_ratio = (ratio_el.text or "0") == "1"
        core_fmt = core_el.get("fmt") or ""
        if window and not (window[0] * 1e9 <= t_ns <= window[1] * 1e9):
            el.clear()
            continue
        if pcore_only and "E Core" in core_fmt:
            el.clear()
            continue
        core_types[core_fmt.split("(")[-1].rstrip(")")] += 1
        a = agg[(name, is_ratio)]
        a[0] += val * d_ns
        a[1] += d_ns
        a[2] += val
        el.clear()
    return agg, core_types


def parse_remarks(path):
    rr = RefResolver()
    remarks = Counter()
    for _ev, el in ET.iterparse(path, events=("end",)):
        if el.tag != "row":
            continue
        name, synopsis = None, None
        for sub in el.iter():
            if sub is el:
                continue
            r = rr.resolve(sub)
            if r.tag == "recount-remark-name" and name is None:
                s = r.find("string")
                name = (s.get("fmt") if s is not None else None) or r.get("fmt")
            # synopsis: a bare string child of row (not inside other elements)
        kids = list(el)
        if len(kids) >= 8:
            syn = rr.resolve(kids[7])
            if syn.tag == "string":
                synopsis = syn.get("fmt") or syn.text
        if name:
            remarks[(name, synopsis or "")] += 1
        el.clear()
    return remarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--window", nargs=2, type=float, default=None,
                    help="restrict to [start end] seconds of trace time")
    ap.add_argument("--pcore-only", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cache = args.cache_dir or os.path.dirname(os.path.abspath(args.trace))
    mt = export_table(args.trace, "MetricTable", cache)
    rm = export_table(args.trace, "RemarksByThread", cache)

    agg, cores = parse_metric_table(mt, args.window, args.pcore_only)
    print(f"== Metric aggregation ({args.trace}) ==")
    if args.window:
        print(f"   window: {args.window[0]}..{args.window[1]} s")
    print(f"   core-type sample counts: {dict(cores)}")
    total_dur = max((a[1] for a in agg.values()), default=0)
    print(f"   covered thread-time: {total_dur/1e9:.2f} s")
    for (name, is_ratio), (wsum, dsum, vsum) in sorted(agg.items()):
        if is_ratio:
            print(f"   {name}: {wsum/dsum:.4f} (duration-weighted mean)")
        elif dsum:
            print(f"   {name}: total {vsum:,.0f}  rate {vsum/(dsum/1e9):,.0f}/s")
        else:
            print(f"   {name}: {vsum:,.0f}")

    remarks = parse_remarks(rm)
    print("\n== Instruments remarks (bottleneck analysis) ==")
    for (name, syn), n in remarks.most_common(15):
        print(f"   [{n:5d}x] {name}: {syn}")


if __name__ == "__main__":
    main()

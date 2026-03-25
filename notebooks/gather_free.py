import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    # Proof-of-concept: build gather-free, permute-style (nibble-trie) tables for Unicode L/N/Z/O classification,
    # then verify they reproduce the reference classification for ALL code points (0..0x10FFFF).
    #
    # The tables are organized into "banks" chosen by high nibbles (e.g., plane for 4-byte cps, high nibble(s) for 3/2-byte).
    # Inside each bank, each trie step is encoded as a 64-entry table mapping (local_shape<<4)|nibble -> code.
    # Codes 0..3 are final classes: O=3, L=0, N=1, Z=2 (see map below). Codes >=4 mean "continue", with next_shape = code-4.
    #
    # We ensure every bank has <= max_shapes_per_step local shapes at every depth by refining banks
    # (splitting by another high nibble) as needed. That keeps each step realizable as a small number of 64B vperm LUTs.

    import unicodedata
    from collections import defaultdict
    from dataclasses import dataclass
    from typing import Dict, List, Tuple
    import math

    # --- L/N/Z/O mapping ---
    L, N, Z, O = 0, 1, 2, 3

    def ref_class(cp: int) -> int:
        if 0xD800 <= cp <= 0xDFFF:  # UTF-16 surrogates -> O
            return O
        cat = unicodedata.category(chr(cp))
        if cat[0] == "L":
            return L
        if cat == "Nd":
            return N
        if cat[0] == "Z":
            return Z
        return O

    def nibble(cp: int, shift: int) -> int:
        return (cp >> shift) & 0xF

    # Initial bank definitions and allowed refinement shifts (in top-down order)
    BANK_KINDS = [
        dict(name="len2", low=0x0080, high=0x0800, refine_shifts=[8, 4]),  # n2, then n1
        dict(name="len3", low=0x0800, high=0x10000, refine_shifts=[12, 8]),  # n3, then n2
        dict(name="plane", low=0x10000, high=0x110000, refine_shifts=[16, 12, 8]),  # plane, then n3, then n2
    ]

    @dataclass
    class Bank:
        kind: str
        key: Tuple[int, ...]
        refine_shifts: Tuple[int, ...]
        low: int
        high: int

    @dataclass
    class StepTables:
        # One step: G groups (each up to 4 shapes). Each group has a 64-entry LUT.
        group_luts: List[List[int]]
        num_shapes: int  # total shapes at this step

    @dataclass
    class BankTables:
        steps: List[StepTables]  # top -> bottom
        root_shape: int
        depth: int

    return (
        BANK_KINDS,
        Bank,
        BankTables,
        Dict,
        L,
        List,
        N,
        O,
        StepTables,
        Tuple,
        Z,
        math,
        nibble,
        ref_class,
    )


@app.cell
def _(
    BANK_KINDS,
    Bank,
    BankTables,
    List,
    O,
    StepTables,
    Tuple,
    math,
    ref_class,
):
    def bank_step_shifts(bank: Bank) -> List[int]:
        if bank.kind == "len2":
            full = [8, 4, 0]
        elif bank.kind == "len3":
            full = [12, 8, 4, 0]
        elif bank.kind == "plane":
            full = [12, 8, 4, 0]  # within a plane
        else:
            raise ValueError
        fixed = set(bank.refine_shifts)
        return [s for s in full if s not in fixed]

    def initial_banks():
        return [Bank(bk["name"], tuple(), tuple(), bk["low"], bk["high"]) for bk in BANK_KINDS]

    def iter_prefixes(bank: Bank, used_shifts: List[int]):
        # Generate all combinations for the specified shifts, honoring fixed bank nibble values
        fixed_map = {s: v for s, v in zip(bank.refine_shifts, bank.key)}

        def rec(i, acc):
            if i == len(used_shifts):
                yield tuple(acc)
                return
            s = used_shifts[i]
            if s in fixed_map:
                acc.append(fixed_map[s])
                yield from rec(i + 1, acc)
                acc.pop()
            else:
                for v in range(16):
                    acc.append(v)
                    yield from rec(i + 1, acc)
                    acc.pop()

        yield from rec(0, [])

    def compose_cp_from_prefix(bank: Bank, used_shifts: List[int], prefix: Tuple[int, ...], last_shift: int, last_val: int) -> int:
        cp = 0
        # Apply *all* fixed nibbles (including plane at shift 16 if present)
        for s, v in zip(bank.refine_shifts, bank.key):
            cp |= v << s
        # Apply chosen higher-nibble prefix
        for s, v in zip(used_shifts, prefix):
            cp |= v << s
        # Apply the row's nibble
        cp |= last_val << last_shift
        return cp

    # --- Core table builder for a bank (refines if any step has too many shapes) ---

    def build_tables_for_bank_with_plane_split(bank: Bank, max_shapes_per_step=12):
        shifts = bank_step_shifts(bank)
        depth = len(shifts)
        used = shifts[:-1]
        last = shifts[-1] if depth > 0 else 0

        # Bottom rows (over the last nibble)
        bottom_rows = {}
        for pref in iter_prefixes(bank, used):
            row = []
            for n in range(16):
                cp = compose_cp_from_prefix(bank, used, pref, last, n) if depth > 0 else 0
                if not (bank.low <= cp < bank.high):
                    cls = O
                else:
                    cls = ref_class(cp)
                row.append(cls)
            bottom_rows[pref] = tuple(row)

        # Dedup to shapes (level = 1 from bottom)
        shapes_levels = []  # list of dict: sid -> 16-tuple codes; bottom .. top
        sid_of_prefix = {}
        row_to_sid = {}
        for pref, row in bottom_rows.items():
            rid = row_to_sid.get(row)
            if rid is None:
                rid = len(row_to_sid)
                row_to_sid[row] = rid
            sid_of_prefix[pref] = rid
        shapes_levels.append({sid: row for row, sid in row_to_sid.items()})

        # Build upper levels
        for lvl in range(2, depth + 1):
            used_up = shifts[:-(lvl)]
            # For each parent prefix (higher nibbles), build a row where entries are either a leaf class 0..3
            # (if the child row is uniform), or 4+child_shape_id
            row_map = {}
            for pp in iter_prefixes(bank, used_up):
                codes = []
                for n in range(16):
                    sid = sid_of_prefix[pp + (n,)]
                    child_row = shapes_levels[-1][sid]
                    first = child_row[0]
                    if all(x == first for x in child_row):
                        codes.append(first)
                    else:
                        codes.append(4 + sid)
                row_map[pp] = tuple(codes)
            new_row_to_sid, new_shapes, new_sid_of_prefix = {}, {}, {}
            for pp, row in row_map.items():
                rid = new_row_to_sid.get(row)
                if rid is None:
                    rid = len(new_row_to_sid)
                    new_row_to_sid[row] = rid
                    new_shapes[rid] = row
                new_sid_of_prefix[pp] = rid
            shapes_levels.append(new_shapes)
            sid_of_prefix = new_sid_of_prefix

        root_sid = sid_of_prefix.get((), 0)

        # Check per-step counts (top..bottom). If any exceeds threshold, ask to refine.
        counts = [len(d) for d in shapes_levels[::-1]]
        allowed = next(bk for bk in BANK_KINDS if bk["name"] == bank.kind)["refine_shifts"]
        # Consider any allowed shift not already fixed (including 16 for plane)
        to_use = [s for s in allowed if s not in bank.refine_shifts]
        if any(c > max_shapes_per_step for c in counts) and to_use:
            split_shift = to_use[0]
            children = [Bank(bank.kind, bank.key + (val,), bank.refine_shifts + (split_shift,), bank.low, bank.high) for val in range(16)]
            return None, children

        # Emit per-step grouped 64-entry LUTs (≤4 shapes/group → 64 entries: (local_shape<<4)|nibble)
        steps = []
        for shape_rows in shapes_levels[::-1]:
            S = len(shape_rows)
            groups = math.ceil(S / 4)
            group_luts = [[O] * 64 for _ in range(groups)]
            for sid, row in shape_rows.items():
                g, loc = divmod(sid, 4)
                base = loc << 4
                lut = group_luts[g]
                for n in range(16):
                    lut[base | n] = row[n]
            steps.append(StepTables(group_luts=group_luts, num_shapes=S))
        return BankTables(steps=steps, root_shape=root_sid, depth=depth), []

    def build_tables_for_bank(bank: Bank, max_shapes_per_step=12):
        # Always split 'plane' first by plane id if it's not already fixed (prevents bogus "all O" rows)
        if bank.kind == "plane" and 16 not in bank.refine_shifts:
            children = [Bank(bank.kind, bank.key + (val,), bank.refine_shifts + (16,), bank.low, bank.high) for val in range(16)]
            return None, children
        return build_tables_for_bank_with_plane_split(bank, max_shapes_per_step)

    def build_all_banks(max_shapes_per_step=12):
        pending = initial_banks()
        done = {}
        while pending:
            bank = pending.pop()
            tables, children = build_tables_for_bank(bank, max_shapes_per_step=max_shapes_per_step)
            if children:
                pending.extend(children)
            else:
                key = (bank.kind, bank.refine_shifts, bank.key, bank.low, bank.high)
                done[key] = (bank, tables)
        return done

    return bank_step_shifts, build_all_banks


@app.cell
def _(
    Bank,
    BankTables,
    Dict,
    L,
    N,
    O,
    Tuple,
    Z,
    bank_step_shifts,
    nibble,
    ref_class,
):
    # --- Runtime simulator using the LUTs (simulates vperm + 1–4 tables per step with mask blend) ---

    def classify_with_tables(cp: int, banks: Dict[Tuple, Tuple[Bank, BankTables]]) -> int:
        # ASCII fast path (computed, no tables)
        if cp < 0x80:
            c = chr(cp)
            if "A" <= c <= "Z" or "a" <= c <= "z":
                return L
            if "0" <= c <= "9":
                return N
            if cp == 0x20:
                return Z
            return O

        # Pick bank (linear scan is fine for PoC; production would index)
        for (k_kind, refine_shifts, key, low, high), (bank, bt) in banks.items():
            if not (low <= cp < high):
                continue
            ok = True
            for s, v in zip(refine_shifts, key):
                if nibble(cp, s) != v:
                    ok = False
                    break
            if not ok:
                continue

            # Walk nibble steps (top → bottom)
            shape = bt.root_shape
            shifts = bank_step_shifts(bank)
            for step_idx, step in enumerate(bt.steps):
                n = nibble(cp, shifts[step_idx])
                g, loc = divmod(shape, 4)
                code = step.group_luts[g][(loc << 4) | n]
                if code < 4:
                    return code
                shape = code - 4
            # Should always return inside loop
            return O

        # Fallback (shouldn't happen)
        return ref_class(cp)

    return (classify_with_tables,)


@app.cell
def _(build_all_banks, classify_with_tables, ref_class):
    # --- Build & verify ---
    banks = build_all_banks(max_shapes_per_step=12)

    # Stats
    num_banks = len(banks)
    total_steps = sum(len(bt.steps) for _, bt in banks.values())
    total_luts = sum(len(step.group_luts) for _, bt in banks.values() for step in bt.steps)
    lut_bytes = total_luts * 64
    max_groups_per_step = max(len(step.group_luts) for _, bt in banks.values() for step in bt.steps)
    max_shapes_per_step = max(step.num_shapes for _, bt in banks.values() for step in bt.steps)

    print(
        f"Banks: {num_banks}, Steps: {total_steps}, LUTs: {total_luts}, ~{lut_bytes} bytes of tables, "
        f"max groups/step: {max_groups_per_step}, max shapes/step: {max_shapes_per_step}"
    )

    # Verify equality for every code point
    mismatches = 0
    for cp in range(0x110000):
        r = ref_class(cp)
        g = classify_with_tables(cp, banks)
        if r != g:
            mismatches += 1
            if mismatches <= 5:
                print(f"Mismatch at U+{cp:04X}: ref={r}, got={g}")
    print("Mismatches:", mismatches)

    return


app._unparsable_cell(
    r"""
    # Build ranges
    for cp in range(0x110000):
        r = ref_class(cp)
        if
    """,
    name="_",
)


if __name__ == "__main__":
    app.run()

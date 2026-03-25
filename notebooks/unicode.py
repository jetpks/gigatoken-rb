import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    ord("ø")
    return


@app.cell
def _(category):
    category("Ø")
    return


@app.cell
def _():
    from unicodedata import category

    codepoints = []
    invalid = 0
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:  # skip surrogate range
            continue
        ch = chr(cp)
        cat = category(ch)
        if cat == "Cn":
            invalid += 1
            continue
        category_group = cat[0]
        if category_group not in ("L", "N", "Z"):
            category_group = "O"
        codepoints.append((ch, category_group))
    return category, codepoints, invalid


@app.cell
def _(codepoints):
    codepoints[:300]
    return


app._unparsable_cell(
    r"""
    category_group_runs =
    """,
    name="_",
)


@app.cell
def _(invalid):
    invalid
    return


@app.cell
def _(codepoints):
    len(codepoints)
    return


@app.cell
def _(codepoints):
    # Calculate how many runs of contiguous values there are (we can simplify this information into ranges if there are few enough)
    prev_g = None
    ranges = 0
    for c, g in codepoints:
        if g != prev_g:
            ranges += 1
            prev_g = g
    ranges
    return


@app.cell
def _():
    from enum import Enum, auto

    class Group(Enum):
        LETTER = auto()
        NUMBER = auto()
        SEPARATOR = auto()
        OTHER = auto()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from dataclasses import dataclass
    from enum import IntEnum
    from unicodedata import category

    class CLASS(IntEnum):
        """2 bit enum representing the character class of a code point"""

        L = 0
        N = 1
        Z = 2
        O = 3

        def __repr__(self):
            return self.name

        def as_bits(self):
            return tuple([int(b) for b in f"{self.value:02b}"])

    def cp_class(cp: int) -> int:
        if 0xD800 <= cp <= 0xDFFF:  # surrogates
            return CLASS.O
        c0 = category(chr(cp))[0]
        if c0 == "L":
            return CLASS.L
        if c0 == "N":
            return CLASS.N
        if c0 == "Z":
            return CLASS.Z
        return CLASS.O

    def get_bits(b):
        return [int(i) for i in f"{b:08b}"]

    @dataclass(slots=True)
    class Codepoint:
        cp: int
        bytes: tuple[int]
        cls: CLASS

        @classmethod
        def from_utf32(cls, cp: int) -> Codepoint:
            bytes = tuple(chr(cp).encode("utf-8", "surrogatepass"))
            clss = cp_class(cp)
            return cls(cp, bytes, clss)

        def as_sequence(self) -> tuple[int | CLASS]:
            return (*self.bytes, cls)

        # @lru_cache(maxsize=None)
        def as_bit_sequence(self, binary: CLASS | None = None, reverse: bool = False) -> tuple[int | CLASS]:
            cls = self.cls
            if binary is not None:
                if cls != binary:
                    cls = CLASS.O
            if reverse:
                return (*list(bit for b in self.bytes for bit in reversed(get_bits(b))), cls)
            return (*(bit for b in self.bytes for bit in get_bits(b)), cls)

        def __repr__(self):
            bytes_str = " ".join(f"{b:02X}" for b in self.bytes)
            return f"{chr(self.cp)}[U+{self.cp:X}, {bytes_str}, {self.cls.name}]"

        def __str__(self):
            return repr(self)

        def char(self) -> str:
            return chr(self.cp)

        def __eq__(self, other):
            return self.cp == other.cp

        def __hash__(self):
            return hash(self.cp)

    def _codepoints():
        res = []
        for cp in range(0x110000):
            res.append(Codepoint.from_utf32(cp))

        by_length: list[list[Codepoint]] = [[] for _ in range(5)]
        for cp in res:
            by_length[len(cp.bytes)].append(cp)

        return res, by_length

    codepoints, cp_by_length = _codepoints()
    return Codepoint, category


@app.cell
def _(Codepoint, category):
    from emoji import EMOJI_DATA

    print(EMOJI_DATA.keys())

    crying_laughing = Codepoint.from_utf32(ord("😂"))
    category("😂")
    return (EMOJI_DATA,)


@app.cell
def _(EMOJI_DATA, category):
    # Which emojis are not length 4, and which categories are they in?
    from collections import Counter, defaultdict

    by_cat = defaultdict(list)
    for f in EMOJI_DATA:
        if len(f[0].encode("utf-8")) < 4:
            continue
        cat = category(f[0])
        by_cat[cat].append(f)
    by_cat
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

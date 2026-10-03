"""Whole numbers as English words, for read-backs: 43 -> "forty-three",
1500 -> "one thousand five hundred". No "and", no commas, so it reads aloud cleanly."""

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
]  # fmt: skip
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")]


def number_words(n: int) -> str:
    if n < 0:
        return f"minus {number_words(-n)}"
    if n < 1000:
        return _below_1000(n)
    parts = []
    for value, name in _SCALES:
        count, n = divmod(n, value)
        if count:
            parts.append(f"{number_words(count)} {name}")
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if not hundreds:
        return _below_100(rest)
    return f"{_ONES[hundreds]} hundred" + (f" {_below_100(rest)}" if rest else "")


def _below_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")

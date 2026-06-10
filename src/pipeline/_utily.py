import re

from datasketch import MinHash


def normalize_indent(code: str, per_indent: int = 4) -> str:
    """Normalise tabs to spaces and round indentation to multiples.

    Converts leading tabs to spaces, then rounds the indent level
    to the nearest multiple of per_indent. Blank lines are preserved.

    Args:
        code: Source code string.
        per_indent: Spaces per indentation level.

    Returns:
        Code with consistent, rounded indentation.
    """
    lines = []
    for line in code.split("\n"):
        if not line.strip():
            lines.append("")
            continue
        stripped = line.lstrip()
        leading = line[: len(line) - len(stripped)]
        leading = leading.expandtabs(per_indent)
        level = round(len(leading) / per_indent)
        lines.append(" " * (level * per_indent) + stripped)
    return "\n".join(lines)


def shingle(code: str, n: int = 5) -> set:
    """Build token n-gram shingles for MinHash similarity.

    Splits on non-alphanumeric characters and forms n-grams
    of consecutive tokens.

    Args:
        code: Source code string.
        n: Shingle size in tokens.

    Returns:
        Set of shingle strings.
    """
    tokens = re.split(r"[^a-zA-Z0-9_]", code)
    tokens = [t for t in tokens if t]
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def minhash(shingles: set, num_perm: int = 128) -> MinHash:
    """Build a MinHash digest from a shingle set.

    Args:
        shingles: Set of shingle strings.
        num_perm: Number of permutation hash functions.

    Returns:
        MinHash instance usable with MinHashLSH.
    """
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m

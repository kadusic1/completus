"""Phase 2: EDA and data cleaning. Produces diagnostic figures."""

import collections
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from datasets import load_dataset  # type: ignore[import-untyped]

RAW_COLOR = "#B4B2A9"
CLEAN_COLOR = "#5DCAA5"
LANGUAGES = ["python", "go"]
HIST_RANGE = (0, 3000)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def valid(x: dict[str, Any]) -> bool:
    """Return True if a CodeSearchNet sample passes quality filters.

    Args:
        x: A dataset row with func_documentation_string and
            func_code_string fields.

    Returns:
        True when the sample meets all quality thresholds.
    """
    doc = x["func_documentation_string"]
    code = x["func_code_string"]
    return (
        len(doc) > 20
        and len(code) > 50
        and len(code) // 4 <= 2048
        and "TODO" not in code
        and code.count("\n") > 2
    )


def _token_lengths(dataset: Any) -> list[int]:
    """Compute approximate token lengths using char_len // 4 proxy.

    Args:
        dataset: HuggingFace dataset with func_code_string column.

    Returns:
        List of approximate token counts, one per sample.
    """
    return [len(x["func_code_string"]) // 4 for x in dataset]


_IMPORT_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"^import\s+([\w]+)",
        r"^from\s+([\w]+)",
    ],
    "go": [
        r'^import\s+"([\w./]+)"',
        r'^\s+"([\w./]+)"',
    ],
}

_GO_STDLIB = {
    "fmt", "os", "io", "strings", "strconv", "errors", "bytes",
    "bufio", "math", "sort", "sync", "time", "log", "net", "http",
    "path", "filepath", "context", "reflect", "regexp", "unicode",
    "encoding", "json", "xml", "csv", "flag", "testing", "runtime",
}


def _top_imports(
    dataset: Any, lang: str, n: int = 15
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Extract the top-n import package names from a dataset split.

    Go func_code_string rarely contains import blocks (imports live at
    file scope), so Go results are filtered to known stdlib names or
    paths containing '/' to avoid matching unrelated string literals.

    Args:
        dataset: HuggingFace dataset with func_code_string column.
        lang: Language key used to select the correct import patterns.
        n: Number of top imports to return.

    Returns:
        A (labels, counts) tuple of length <= n.
    """
    counter: collections.Counter[str] = collections.Counter()
    patterns = _IMPORT_PATTERNS.get(lang, _IMPORT_PATTERNS["python"])
    for x in dataset:
        code = x["func_code_string"]
        for pattern in patterns:
            for match in re.findall(pattern, code, re.MULTILINE):
                if lang == "go" and "/" not in match:
                    if match not in _GO_STDLIB:
                        continue
                counter[match] += 1
    top = counter.most_common(n)
    if not top:
        return ("(none found)",), (0,)  # type: ignore[return-value]
    labels, counts = zip(*top)
    return labels, counts  # type: ignore[return-value]


def _plot_and_save(
    lang: str,
    raw: Any,
    clean: Any,
    lengths_raw: list[int],
    lengths_clean: list[int],
) -> None:
    """Render the 3-subplot diagnostic figure and save as PDF.

    Args:
        lang: Language name used in titles and filename.
        raw: Full (unfiltered) dataset split.
        clean: Filtered dataset split.
        lengths_raw: Approximate token lengths for raw split.
        lengths_clean: Approximate token lengths for clean split.
    """
    labels, counts = _top_imports(clean, lang)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(
        lengths_raw, bins=50, range=HIST_RANGE, color=RAW_COLOR, label="raw"
    )
    axes[0].hist(
        lengths_clean, bins=50, range=HIST_RANGE, color=CLEAN_COLOR,
        label="clean", alpha=0.7,
    )
    axes[0].axvline(
        2048, color="red", linestyle="--", label="2048 token cutoff"
    )
    axes[0].set_title(f"{lang} - token lengths")
    axes[0].legend()

    axes[1].bar(labels, counts, color=CLEAN_COLOR)
    axes[1].set_title(f"{lang} - top imports")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(
        ["raw", "clean"],
        [len(raw), len(clean)],
        color=[RAW_COLOR, CLEAN_COLOR],
    )
    axes[2].set_title(f"{lang} - sample count before/after")

    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / f"eda_{lang}.pdf", dpi=150)
    plt.close(fig)
    print(f"Saved results/eda_{lang}.pdf")


def _print_table(rows: list[tuple[str, int, int]]) -> None:
    """Print a formatted before/after retention table.

    Args:
        rows: List of (language, raw_count, clean_count) tuples.
    """
    header = f"{'Language':<10} {'Raw':>8} {'Clean':>8} {'Retention':>10}"
    print(header)
    print("-" * len(header))
    for lang, raw_n, clean_n in rows:
        pct = 100 * clean_n / raw_n if raw_n else 0.0
        print(f"{lang:<10} {raw_n:>8} {clean_n:>8} {pct:>9.1f}%")


def main() -> None:
    """Run EDA pipeline for all configured languages."""
    rows: list[tuple[str, int, int]] = []

    for lang in LANGUAGES:
        raw = load_dataset(
            "claudios/code_search_net", lang, split="train"
        )
        clean = raw.filter(valid)

        rows.append((lang, len(raw), len(clean)))

        lengths_raw = _token_lengths(raw)
        lengths_clean = _token_lengths(clean)

        _plot_and_save(lang, raw, clean, lengths_raw, lengths_clean)

    _print_table(rows)


if __name__ == "__main__":
    main()

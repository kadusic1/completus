import ast
import random

from datasketch import MinHashLSH
from transformers import PreTrainedTokenizerBase

from data_pipeline._base import PipelineStage
from data_pipeline._constants import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
from data_pipeline._types import CodeSample, PipelineConfig
from data_pipeline._utily import minhash, normalize_indent, shingle


class NearDedupStage(PipelineStage):
    """Removes near-duplicate code samples via MinHash LSH.

    Uses 5-gram token shingling and 128-perm MinHash at a
    configurable Jaccard threshold (default 0.8).
    """

    @property
    def name(self) -> str:
        return "dedup"

    def run(
        self, samples: list[CodeSample], config: PipelineConfig
    ) -> list[CodeSample]:
        """Filter out near-duplicate samples.

        Args:
            samples: Input samples with func_code_string.
            config: Pipeline configuration (dedup_threshold, shingle_size).

        Returns:
            Deduplicated sample list.
        """
        lsh = MinHashLSH(threshold=config.dedup_threshold, num_perm=128)
        deduped: list[CodeSample] = []
        for i, sample in enumerate(samples):
            s = shingle(sample["func_code_string"], config.shingle_size)
            mh = minhash(s)
            key = f"doc_{i}"
            if not lsh.query(mh):
                lsh.insert(key, mh)
                deduped.append(sample)
        return deduped


class CodeSmellFilter(PipelineStage):
    """Applies heuristic code-quality filters to each sample.

    Drops samples with lines exceeding 120 characters, token counts
    below the minimum, or no-op function bodies. Normalises
    indentation on kept samples.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Initialise with a tokenizer for token-count checks.

        Args:
            tokenizer: HuggingFace tokenizer for length estimation.
        """
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        return "code_smell"

    def run(
        self, samples: list[CodeSample], config: PipelineConfig
    ) -> list[CodeSample]:
        """Filter and normalise samples based on code quality.

        Args:
            samples: Input samples with func_code_string.
            config: Pipeline configuration (max_line_length, min_tokens).

        Returns:
            Samples that pass all quality checks.
        """
        result: list[CodeSample] = []
        for s in samples:
            code = s["func_code_string"]
            if not self._line_length_ok(code, config.max_line_length):
                continue
            if len(self._tokenizer.encode(code)) < config.min_tokens:
                continue
            if self._has_noop(code):
                continue
            s["func_code_string"] = normalize_indent(code)
            result.append(s)
        return result

    @staticmethod
    def _line_length_ok(code: str, max_line: int) -> bool:
        """Check no line in code exceeds max_line characters.

        Args:
            code: Source code string.
            max_line: Maximum allowed line length.

        Returns:
            True if all lines are within the limit.
        """
        return all(len(line) <= max_line for line in code.split("\n"))

    @staticmethod
    def _has_noop(code: str) -> bool:
        """Detect functions consisting only of pass, ellipsis, or bare return.

        Args:
            code: Source code string (Python).

        Returns:
            True if a no-op function body is found.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.body:
                return True
            for stmt in node.body:
                if isinstance(stmt, (ast.Pass, ast.Ellipsis)):
                    continue
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    continue
                if isinstance(stmt, ast.Return) and stmt.value is None:
                    continue
                break
            else:
                return True
        return False


class DocstringFilter(PipelineStage):
    """Removes samples with low-quality docstrings.

    Heuristic filters: minimum token length, no placeholder
    content (TODO/FIXME/XXX), minimum word count, no exact
    match with the function name.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Initialise with a tokenizer for token-count checks.

        Args:
            tokenizer: HuggingFace tokenizer for length estimation.
        """
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        return "docstring"

    def run(
        self, samples: list[CodeSample], config: PipelineConfig
    ) -> list[CodeSample]:
        """Filter samples by docstring quality heuristics.

        Args:
            samples: Input samples with func_documentation_string.
            config: Pipeline configuration (docstring_min_tokens).

        Returns:
            Samples with acceptable docstrings.
        """
        return [s for s in samples if self._is_ok(s, config)]

    def _is_ok(self, sample: CodeSample, config: PipelineConfig) -> bool:
        """Evaluate a single docstring against all quality rules.

        Args:
            sample: A single CodeSearchNet sample.
            config: Pipeline configuration.

        Returns:
            True if the docstring passes all checks.
        """
        doc = sample.get("func_documentation_string", "")
        if not doc or not doc.strip():
            return False
        if len(self._tokenizer.encode(doc)) < config.docstring_min_tokens:
            return False
        upper = doc.upper()
        if any(p in upper for p in ["TODO", "FIXME", "XXX"]):
            return False
        stripped = doc.strip()
        if stripped.lower() == "pass":
            return False
        if len(stripped.split()) < 2:
            return False
        if stripped.lower() == sample.get("func_name", "").lower():
            return False
        return True


class FIMTransform(PipelineStage):
    """Applies Fill-in-the-Middle transform with SPM ordering.

    80% of samples receive FIM (split across three granularity
    types: single-token, single-line, multi-line). The remaining
    20% stay as standard causal samples.
    """

    def __init__(self) -> None:
        self._rng = random.Random(42)

    @property
    def name(self) -> str:
        return "fim"

    def run(
        self, samples: list[CodeSample], config: PipelineConfig
    ) -> list[CodeSample]:
        """Apply FIM transform to each sample.

        Args:
            samples: Input samples with func_code_string.
            config: Pipeline configuration (fim_rate).

        Returns:
            Samples with a ``text`` field containing FIM or raw code.
        """
        result: list[CodeSample] = []
        for s in samples:
            text = self._transform(s["func_code_string"], config.fim_rate)
            if text is not None:
                s["text"] = text
                result.append(s)
        return result

    def _transform(self, code: str, fim_rate: float) -> str | None:
        """Apply FIM to a single code string.

        Splits the code into prefix / middle / suffix using
        character-level positions.

        Granularity is diversified: 20% single-token, 40%
        single-line, 40% multi-line or random character split.

        Args:
            code: Source code string.
            fim_rate: Probability of applying FIM.

        Returns:
            FIM-formatted string, raw code, or None.
        """
        if self._rng.random() >= fim_rate:
            return code

        lines = code.split("\n")
        strat = self._rng.random()

        prefix: str
        middle: str
        suffix: str

        if strat < 0.2 and len(code) > 20:
            pos = self._rng.randint(1, len(code) - 1)
            span = self._rng.randint(1, min(5, len(code) - pos))
            prefix = code[:pos]
            middle = code[pos : pos + span]
            suffix = code[pos + span :]
        elif strat < 0.6 and len(lines) >= 3:
            mid_line = self._rng.randint(1, len(lines) - 2)
            prefix = "\n".join(lines[:mid_line])
            middle = lines[mid_line]
            suffix = "\n".join(lines[mid_line + 1 :])
        elif len(lines) >= 6:
            start = self._rng.randint(1, len(lines) - 4)
            end = self._rng.randint(start + 2, len(lines) - 2)
            prefix = "\n".join(lines[:start])
            middle = "\n".join(lines[start:end])
            suffix = "\n".join(lines[end:])
        else:
            boundaries = sorted(self._rng.sample(range(len(code) + 1), 2))
            p_start, p_end = boundaries[0], boundaries[1]
            prefix = code[:p_start]
            middle = code[p_start:p_end]
            suffix = code[p_end:]

        return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"


class LengthManagement(PipelineStage):
    """Enforces token-length bounds and context-to-completion ratio.

    Drops samples below min_tokens. For FIM samples, truncates
    the completion to maintain a 3:1 context ratio. When exceeding
    max_tokens, truncates from the head of the prefix to preserve
    suffix and middle context.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Precompute FIM special token IDs for fast lookups.

        Args:
            tokenizer: HuggingFace tokenizer.
        """
        self._tokenizer = tokenizer
        self._fim_prefix = tokenizer.encode(FIM_PREFIX, add_special_tokens=False)[0]
        self._fim_suffix = tokenizer.encode(FIM_SUFFIX, add_special_tokens=False)[0]
        self._fim_middle = tokenizer.encode(FIM_MIDDLE, add_special_tokens=False)[0]

    @property
    def name(self) -> str:
        return "length_mgmt"

    def run(
        self, samples: list[CodeSample], config: PipelineConfig
    ) -> list[CodeSample]:
        """Apply token-length management to each sample.

        Args:
            samples: Input samples with ``text`` field.
            config: Pipeline configuration (min/max tokens, ratio).

        Returns:
            Samples that fit within all length constraints.
        """
        result: list[CodeSample] = []
        for s in samples:
            text = self._manage(s["text"], config)
            if text is not None:
                s["text"] = text
                result.append(s)
        return result

    def _manage(self, text: str, config: PipelineConfig) -> str | None:
        """Adjust a single text to fit token limits.

        Steps:
            1. Drop if below min_tokens.
            2. For FIM samples, cap the completion to satisfy
               the context-to-completion ratio.
            3. If still above max_tokens, truncate from the
               head of the prefix (or from the left for raw text).
            4. Re-check min_tokens after decoding.

        Args:
            text: FIM-formatted or raw code string.
            config: Pipeline configuration.

        Returns:
            Length-adjusted text, or None if below minimum.
        """
        tokens = self._tokenizer.encode(text)
        if len(tokens) < config.min_tokens:
            return None

        fim_ids = {self._fim_prefix, self._fim_suffix, self._fim_middle}
        has_fim = fim_ids.issubset(tokens)

        if has_fim:
            mid_pos = tokens.index(self._fim_middle)
            context_len = mid_pos + 1
            completion_len = len(tokens) - mid_pos - 1
            max_completion = int(context_len // config.context_ratio)
            if completion_len > max_completion:
                tokens = (
                    tokens[: mid_pos + 1]
                    + tokens[mid_pos + 1 : mid_pos + 1 + max_completion]
                )

        if len(tokens) > config.max_tokens and has_fim:
            prefix_pos = tokens.index(self._fim_prefix)
            suffix_pos = tokens.index(self._fim_suffix)
            overflow = len(tokens) - config.max_tokens
            prefix_tokens = tokens[prefix_pos + 1 : suffix_pos]
            keep = max(0, len(prefix_tokens) - overflow)
            truncated = [] if keep == 0 else prefix_tokens[-keep:]
            tokens = [self._fim_prefix] + truncated + tokens[suffix_pos:]
        elif len(tokens) > config.max_tokens:
            tokens = tokens[-config.max_tokens :]

        final: str = self._tokenizer.decode(tokens)  # type: ignore[assignment]
        if len(self._tokenizer.encode(final)) < config.min_tokens:
            return None
        return final

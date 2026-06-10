from pipeline._base import PipelineStage
from pipeline._types import CodeSample, PipelineConfig


class Pipeline:
    """Orchestrates a sequence of PipelineStage instances.

    Stages are injected via the constructor (not hardcoded),
    matching the sklearn/datatrove composability pattern.
    """

    def __init__(self, stages: list[PipelineStage]) -> None:
        """Initialise with an ordered list of pipeline stages.

        Args:
            stages: PipelineStage instances to run sequentially.
        """
        self._stages = stages

    def run(
        self, raw: list[CodeSample], config: PipelineConfig
    ) -> tuple[list[CodeSample]]:
        """Run raw samples through every stage sequentially.

        Args:
            raw: Raw samples loaded from the dataset.
            config: Pipeline configuration.

        Returns:
            Tuple of (processed samples).
        """
        samples = raw
        for stage in self._stages:
            samples = stage.run(samples, config)
        return samples

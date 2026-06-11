from abc import ABC, abstractmethod

from data_pipeline._types import CodeSample, PipelineConfig


class PipelineStage(ABC):
    """Base class for a single stage in the data pipeline.

    Each stage receives a list of samples and a config, and returns
    the filtered or transformed list. Stages are composable and
    independently testable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name for logging."""

    @abstractmethod
    def run(
        self, samples: list[CodeSample], config: PipelineConfig
    ) -> list[CodeSample]:
        """Execute this stage's transformation.

        Args:
            samples: Input samples to process.
            config: Global pipeline configuration.

        Returns:
            Processed samples.
        """

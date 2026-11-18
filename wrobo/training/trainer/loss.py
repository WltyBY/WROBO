"""Mean loss components consumed by the shared optimizer update."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class LossTerm:
    """A differentiable local mean and its averaging unit.

    ``valid_steps`` excludes padded action timesteps; ``samples`` is useful for
    sample-wise objectives such as ACT's KL divergence. Action width is fixed
    within a run, so weighting action means by valid steps also weights by valid
    action elements. ``weight`` affects the objective, not the reported component.
    """

    value: Tensor
    normalization: str = "valid_steps"
    weight: float = 1.0

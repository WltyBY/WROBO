"""Resolve ACT state/action widths without changing legacy configurations."""


def resolve_act_dimensions(proprio_dim, action_dim=None):
    """Default the action width to the state width for existing ACT configs."""
    if action_dim is None:
        action_dim = proprio_dim
    for name, width in (("proprio_dim", proprio_dim), ("action_dim", action_dim)):
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError(f"ACT Policy.{name} must be a positive integer")
    return proprio_dim, action_dim

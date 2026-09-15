"""Target lookup shared by normal and capsule model execution."""


def select_cta_target(targets, scene_index=None, message_order=None, trial=None):
    """Select one CTA target by stream identity or paired trial."""
    matches = [
        target for target in targets
        if (
            target.get("trial") == trial
            if trial is not None else
            target.get("scene_index") == scene_index
            and target.get("message_order") == message_order
        )
    ]
    if len(matches) > 1:
        raise ValueError("CTA target identity is ambiguous")
    return matches[0] if matches else None

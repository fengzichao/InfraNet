"""Fail-fast stream-order assertions for paired InfraNet datasets.

The historical `train_rgb`/`val_rgb` keys mean *secondary stream*, not RGB.
InfraNet-IR C explicitly fixes which modality is in each stream.
"""

from typing import Mapping

EXPECTED_ORDERS = {
    "c": ("ir", "rgb"),
}


def validate_modality_order(data: Mapping, variant: str, *, require: bool = False) -> bool:
    """Reject known wrong stream order; return False for undeclared legacy data.

    Passing an undeclared legacy dataset is permitted only when require=False.
    This does *not* prove that image contents match the declared modalities.
    """
    if variant not in EXPECTED_ORDERS:
        raise ValueError(f"Unsupported InfraNet variant: {variant!r}")
    declared = data.get("modality_order")
    if declared is None:
        if require:
            raise ValueError(
                "Dataset YAML is missing modality_order. "
                f"Declare {list(EXPECTED_ORDERS[variant])!r} for variant {variant.upper()} "
                "to prevent silent RGB/IR stream swaps."
            )
        return False
    actual = tuple(str(modality).strip().lower() for modality in declared)
    expected = EXPECTED_ORDERS[variant]
    if actual != expected:
        raise ValueError(
            f"Incorrect {variant.upper()} modality_order: {list(actual)!r}. "
            f"Expected {list(expected)!r} (first stream, second stream). "
            "Note: train_rgb/val_rgb name the SECOND stream, not necessarily RGB."
        )
    return True

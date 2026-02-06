"""Selection validation implementation."""

from dataclasses import dataclass


@dataclass
class ValidationError:
    """Represents a validation error."""

    dataset_key: str
    message: str
    error_type: str  # "cardinality", "missing_key", etc.


def validate_cardinality(
    dataset_key: str,
    cardinality: str,
    selected_count: int,
) -> ValidationError | None:
    """Validate selection against cardinality constraint.

    Args:
        dataset_key: The dataset key.
        cardinality: The cardinality constraint ("one", "many", "zeroOrMany").
        selected_count: Number of selected items.

    Returns:
        ValidationError if invalid, None if valid.
    """
    if cardinality == "one":
        if selected_count != 1:
            return ValidationError(
                dataset_key=dataset_key,
                message=f"Cardinality 'one' requires exactly 1 item, got {selected_count}",
                error_type="cardinality",
            )
    elif cardinality == "many":
        if selected_count < 1:
            return ValidationError(
                dataset_key=dataset_key,
                message=f"Cardinality 'many' requires at least 1 item, got {selected_count}",
                error_type="cardinality",
            )
    elif cardinality == "zeroOrMany":
        # Always valid
        pass
    else:
        return ValidationError(
            dataset_key=dataset_key,
            message=f"Unknown cardinality: {cardinality}",
            error_type="unknown_cardinality",
        )

    return None

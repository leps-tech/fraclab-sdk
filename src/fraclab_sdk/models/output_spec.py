"""OutputSpec helper aliases for algorithm workspace schemas."""

from fraclab_sdk.models.output_contract import (
    BlobOutputSchema,
    Cardinality,
    DatasetKind,
    DatasetRole,
    FrameOutputSchema,
    ObjectOutputSchema,
    OutputContract,
    OutputDatasetContract,
    OwnerType,
    ScalarOutputSchema,
)

BlobSchema = BlobOutputSchema
ObjectSchema = ObjectOutputSchema
ScalarSchema = ScalarOutputSchema
FrameSchema = FrameOutputSchema

__all__ = [
    "BlobSchema",
    "ObjectSchema",
    "ScalarSchema",
    "FrameSchema",
    "OutputContract",
    "OutputDatasetContract",
    "OwnerType",
    "DatasetKind",
    "DatasetRole",
    "Cardinality",
]

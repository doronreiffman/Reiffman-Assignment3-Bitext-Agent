"""Pydantic input schemas for dataset tools."""

from pydantic import BaseModel, Field


class ListIntentsInput(BaseModel):
    """Optional category to list intents for."""

    category: str | None = Field(
        default=None,
        description=(
            "High-level category name (e.g. ACCOUNT, REFUND). "
            "If omitted, returns all intents in the dataset."
        ),
    )


class FilterByCategoryInput(BaseModel):
    category: str = Field(
        description="Category name exactly as in the dataset (e.g. SHIPPING, ACCOUNT, FEEDBACK)."
    )


class FilterByIntentInput(BaseModel):
    intent: str = Field(
        description=(
            "Intent string exactly as in the dataset (e.g. get_refund, track_refund, complaint). "
            "Call list_intents first if you are unsure of the exact name."
        )
    )


class SampleExamplesInput(BaseModel):
    n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of random examples to return from the current filtered view (max 20).",
    )

"""Pydantic models for API requests/responses and the structured Gemini output."""

from pydantic import BaseModel, Field


class Mistake(BaseModel):
    original: str = Field(description="The exact mistaken phrase or sentence from the user's text.")
    corrected: str = Field(description="The corrected version of that phrase or sentence.")
    category: str = Field(
        description="One of: article, tense, preposition, subject-verb-agreement, "
        "word-order, vocabulary, spelling, punctuation, other."
    )
    explanation: str = Field(description="A short, friendly explanation of why this is a mistake.")
    severity: str = Field(description="One of: minor, moderate, major.")


class FeedbackResult(BaseModel):
    """The structured object Gemini is asked to return."""

    corrected_text: str = Field(description="The user's full text, corrected.")
    mistakes: list[Mistake]
    summary: str = Field(description="A short (1-2 sentence) encouraging overall summary.")


class FeedbackRequest(BaseModel):
    text: str
    user_id: str = "default"


class FeedbackResponse(FeedbackResult):
    submission_id: int

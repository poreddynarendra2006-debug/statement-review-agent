"""FinSight Review Agent package.

Synthesizes, explains, and narratively structures financial audit findings
from the Evidence Agent for human reviewers.
"""

from review_agent.models import ReviewResult
from review_agent.agent import ReviewAgent, generate_review
from review_agent.prompt import (
    prepare_review_context,
    build_system_prompt,
    build_user_prompt,
)

__all__ = [
    "ReviewAgent",
    "generate_review",
    "ReviewResult",
    "prepare_review_context",
    "build_system_prompt",
    "build_user_prompt",
]

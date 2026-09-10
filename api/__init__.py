"""HTTP interface to the review engine.

The workspace is one consumer of the review engine; this is another. Keeping
the engine behind an interface rather than inside the UI is what makes it
reusable - it can be called from another system, tested without a browser, and
scaled independently.
"""

from .models import ReviewRequest, ReviewResponse

__all__ = ["ReviewRequest", "ReviewResponse"]

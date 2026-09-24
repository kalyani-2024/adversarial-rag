"""Domain exceptions.

Each error carries an HTTP status and a stable machine-readable `code`, so the
API layer can translate them uniformly without business logic knowing about
HTTP.
"""

from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- Documents ---------------------------------------------------------------
class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class DocumentParseError(AppError):
    status_code = 422
    code = "document_parse_error"


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"


# --- Models ------------------------------------------------------------------
class EmbeddingError(AppError):
    status_code = 503
    code = "embedding_unavailable"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class LLMTimeoutError(LLMError):
    status_code = 504
    code = "llm_timeout"


class LLMRateLimitError(LLMError):
    status_code = 429
    code = "llm_rate_limited"


class LLMConfigError(LLMError):
    status_code = 503
    code = "llm_not_configured"


class LLMOutputError(LLMError):
    """The model answered, but not in the structure we asked for."""

    code = "llm_bad_output"

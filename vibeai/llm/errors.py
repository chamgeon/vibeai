class InsufficientQuotaError(RuntimeError):
    """Raised when the OpenAI account has no credit left. Retrying will not help."""


class RefusalError(RuntimeError):
    """Raised when Claude declines a request (stop_reason == "refusal").
    Not retryable: the same input will be declined again."""

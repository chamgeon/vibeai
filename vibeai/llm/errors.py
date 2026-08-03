class InsufficientQuotaError(RuntimeError):
    """Raised when the OpenAI account has no credit left. Retrying will not help."""

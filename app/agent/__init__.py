"""Dog-focused veterinary assistant agent (interpretation, triage, safety, RAG-backed responses)."""

from app.agent.rag_client import retrieve_context

__all__ = ["retrieve_context"]

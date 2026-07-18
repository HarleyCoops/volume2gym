"""Adapters for external artifact and training ecosystems."""

from volume2gym.integrations.huggingface import HuggingFaceFixture, load_huggingface_fixture

__all__ = ["HuggingFaceFixture", "load_huggingface_fixture"]

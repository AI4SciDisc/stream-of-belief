"""Stream of Belief: collective reasoning through latent belief dynamics (arXiv:2611.21250)."""
from .belief import belief_token, latent_words, nearest_tokens
from .dynamics import BeliefConfig, Cycle, Result, StreamOfBelief, neighbours
from .model import FrozenLM
from . import tasks

__version__ = "0.1.0"
__all__ = ["StreamOfBelief", "BeliefConfig", "FrozenLM", "Result", "Cycle", "belief_token", "latent_words",
           "nearest_tokens", "neighbours", "tasks"]

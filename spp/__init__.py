"""Self-Play Pretraining with Zero Data (arXiv:2609.30063) -- reference implementation.

A generator LM writes Brainf*ck-like programs, a universal machine executes them into
byte sequences, and a learner LM is pretrained on those bytes. The generator is trained
by policy gradient with a learning-progress reward (preconditioned gradient alignment).
"""
__version__ = "0.1.0"

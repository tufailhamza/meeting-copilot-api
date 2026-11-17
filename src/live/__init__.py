"""QLU Live Agent.

Isolated feature module for the real-time meeting-supervisor agent. Kept fully separate
from the existing transcription/notes code; the only touch-point in the core app is a single
``include_router`` line in ``main.py``.
"""

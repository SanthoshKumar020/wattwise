"""Vercel serverless entrypoint - exposes the Flask app in ../app.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import app  # noqa: E402,F401  (Vercel looks for `app` here)

# src/web/__init__.py
"""
Web module - Flask server và management API.
"""
from .server import run_server, app
from .management import management_bp, init_management

__all__ = [
    'run_server',
    'app',
    'management_bp',
    'init_management',
]

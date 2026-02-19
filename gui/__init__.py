# gui/__init__.py
# Makes 'gui' a package and exposes the public API.
from .app import BilliardsApp, draw_overlay

__all__ = ['BilliardsApp', 'draw_overlay']

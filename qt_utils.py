"""Qt utility helpers."""
from __future__ import annotations


def get_qt_widget(obj, name: str):
    """Return a Qt widget class from the same binding as ``obj``.

    Parameters
    ----------
    obj: Any Qt object
        The instance to inspect for its Qt binding (``PyQt5`` or ``PySide6``).
    name: str
        The name of the widget class to retrieve.

    Returns
    -------
    type | None
        The requested widget class or ``None`` if it cannot be found.
    """
    pkg = obj.__class__.__module__.split(".")[0]
    try:
        mod = __import__(f"{pkg}.QtWidgets", fromlist=[name])
        return getattr(mod, name)
    except Exception:
        pass
    for pkg in ("PyQt5", "PySide6"):
        try:
            mod = __import__(f"{pkg}.QtWidgets", fromlist=[name])
            return getattr(mod, name)
        except Exception:
            continue
    return None

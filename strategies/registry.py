import importlib
import pkgutil
from typing import Dict, Type

from strategies.base import BaseStrategy


def discover_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Auto-discover all BaseStrategy subclasses in the strategies package."""
    import strategies

    registry = {}
    for importer, modname, ispkg in pkgutil.iter_modules(strategies.__path__):
        if modname in ("base", "registry", "__init__"):
            continue
        module = importlib.import_module(f"strategies.{modname}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseStrategy)
                and attr is not BaseStrategy
            ):
                registry[attr_name] = attr

    return registry

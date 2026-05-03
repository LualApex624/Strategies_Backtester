from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, params: Dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series of signals: 1 (long), -1 (short), 0 (flat)."""
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return the strategy's current parameter dictionary."""
        ...

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params.update(params)

    @property
    def name(self) -> str:
        return self.__class__.__name__

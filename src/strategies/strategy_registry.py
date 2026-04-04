from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

from .base_strategy import BaseStrategy


@dataclass
class StrategySpec:
    name: str
    version: str
    family: str
    strategy_cls: Type[BaseStrategy]
    default_params: Dict = field(default_factory=dict)

    def build(self, params: Optional[Dict] = None) -> BaseStrategy:
        merged = {**self.default_params, **(params or {})}
        return self.strategy_cls(params=merged)


class StrategyRegistry:
    def __init__(self):
        self._items: Dict[str, StrategySpec] = {}

    def register(self, spec: StrategySpec) -> None:
        key = f"{spec.name}:{spec.version}"
        if key in self._items:
            raise ValueError(f"Strategy already registered: {key}")
        self._items[key] = spec

    def list(self) -> List[StrategySpec]:
        return list(self._items.values())

    def list_names(self) -> List[str]:
        return [s.name for s in self._items.values()]

    def get(self, name: str, version: Optional[str] = None) -> StrategySpec:
        if version is not None:
            key = f"{name}:{version}"
            if key not in self._items:
                raise KeyError(f"Unknown strategy: {key}")
            return self._items[key]

        candidates = [s for s in self._items.values() if s.name == name]
        if not candidates:
            raise KeyError(f"Unknown strategy name: {name}")
        # version lexical max as default fallback
        return sorted(candidates, key=lambda x: x.version)[-1]

    def build(self, name: str, version: Optional[str] = None, params: Optional[Dict] = None) -> BaseStrategy:
        return self.get(name=name, version=version).build(params=params)


registry = StrategyRegistry()

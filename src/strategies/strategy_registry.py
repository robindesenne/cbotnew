from __future__ import annotations

import ast
import inspect
import os
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from .base_strategy import BaseStrategy

# Colonnes interdites pour la génération de signaux (futures/labels)
FORBIDDEN_SIGNAL_COLUMNS = {
    "label",
    "label_tb",
    "label_fh",
    "event_end_idx",
}


def _literal_str(node) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Index):  # py<3.9 compat
        return _literal_str(node.value)
    return None


def detect_forbidden_column_usage(strategy_cls: Type[BaseStrategy]) -> List[str]:
    """
    Détection statique conservative dans generate_signals:
    - x["col"]
    - x.get("col", ...)
    """
    try:
        src = inspect.getsource(strategy_cls.generate_signals)
    except Exception:
        return []

    tree = ast.parse(textwrap.dedent(src))
    hits = set()

    class Visitor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript):
            k = _literal_str(node.slice)
            if k in FORBIDDEN_SIGNAL_COLUMNS:
                hits.add(k)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
                k = _literal_str(node.args[0])
                if k in FORBIDDEN_SIGNAL_COLUMNS:
                    hits.add(k)
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(hits)


@dataclass
class StrategySpec:
    name: str
    version: str
    family: str
    strategy_cls: Type[BaseStrategy]
    default_params: Dict = field(default_factory=dict)
    forbidden_refs: List[str] = field(default_factory=list)

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

        leakage_mode = os.getenv("CRYPTOBOT_LEAKAGE_MODE", "error").strip().lower() or "error"
        forbidden = detect_forbidden_column_usage(spec.strategy_cls)
        spec.forbidden_refs = forbidden

        if forbidden:
            msg = (
                f"Leakage guard: strategy {key} references forbidden columns in generate_signals: "
                f"{', '.join(forbidden)}"
            )
            if leakage_mode in {"warn", "audit"}:
                warnings.warn(msg)
            else:
                raise ValueError(msg)

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

"""ID <-> name registry for blocks and items (single shared vocabulary)."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


@dataclass
class Registry:
    _name_to_id: dict[str, int]
    _id_to_name: dict[int, str]

    @classmethod
    def load(cls, path: pathlib.Path) -> "Registry":
        if not path.exists():
            raise FileNotFoundError(f"registry not found: {path}")
        doc = json.loads(path.read_text())
        merged: dict[str, int] = {}
        merged.update(doc.get("blocks", {}))
        merged.update(doc.get("items", {}))
        id_to_name = {v: k for k, v in merged.items()}
        return cls(merged, id_to_name)

    def id_of(self, name: str) -> int:
        return self._name_to_id[name]

    def name_of(self, ident: int) -> str:
        return self._id_to_name[ident]

    @property
    def size(self) -> int:
        return len(self._name_to_id)

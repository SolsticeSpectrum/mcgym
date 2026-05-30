"""ID <-> name registry for blocks and items.

Blocks and items live in SEPARATE id spaces on the Java side
(BuiltInRegistries.BLOCK vs BuiltInRegistries.ITEM): the same name can map to a
different integer in each table (e.g. ``minecraft:oak_log`` is block id 49 but
item id 134). The voxel grid and ``target_block`` carry BLOCK ids; the inventory
``inv_item_id`` carries ITEM ids. We therefore keep both maps and expose them
separately via ``block_id_of``/``item_id_of`` (and the legacy ``id_of`` which
resolves against an items-over-blocks merge for backwards compatibility).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


@dataclass
class Registry:
    _block_to_id: dict[str, int]
    _item_to_id: dict[str, int]
    # Merged (items override blocks); kept so legacy id_of/name_of/size still work.
    _name_to_id: dict[str, int]
    _id_to_name: dict[int, str]

    @classmethod
    def load(cls, path: pathlib.Path) -> "Registry":
        if not path.exists():
            raise FileNotFoundError(f"registry not found: {path}")
        doc = json.loads(path.read_text())
        blocks: dict[str, int] = dict(doc.get("blocks", {}))
        items: dict[str, int] = dict(doc.get("items", {}))
        merged: dict[str, int] = {}
        merged.update(blocks)
        merged.update(items)
        id_to_name = {v: k for k, v in merged.items()}
        return cls(blocks, items, merged, id_to_name)

    # --- separate id spaces -------------------------------------------------
    def block_id_of(self, name: str) -> int:
        return self._block_to_id[name]

    def item_id_of(self, name: str) -> int:
        return self._item_to_id[name]

    def block_ids(self) -> dict[str, int]:
        return dict(self._block_to_id)

    def item_ids(self) -> dict[str, int]:
        return dict(self._item_to_id)

    # --- legacy merged view -------------------------------------------------
    def id_of(self, name: str) -> int:
        return self._name_to_id[name]

    def name_of(self, ident: int) -> str:
        return self._id_to_name[ident]

    @property
    def size(self) -> int:
        return len(self._name_to_id)

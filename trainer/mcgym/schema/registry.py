"""id to name registry for blocks and items"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


# blocks and items are separate id spaces on the java side, same name can map to
# different ints (oak_log is block 49 but item 134). voxels and target_block carry
# block ids, inventory carries item ids
@dataclass
class Registry:
    _block_to_id: dict[str, int]
    _item_to_id:  dict[str, int]
    # merged items over blocks, kept for legacy id_of/name_of/size
    _name_to_id:  dict[str, int]
    _id_to_name:  dict[int, str]

    @classmethod
    def load(cls, path: pathlib.Path) -> "Registry":
        if not path.exists():
            raise FileNotFoundError(f"registry not found: {path}")
        doc = json.loads(path.read_text())

        blocks: dict[str, int] = dict(doc.get("blocks", {}))
        items:  dict[str, int] = dict(doc.get("items", {}))
        merged: dict[str, int] = {}
        merged.update(blocks)
        merged.update(items)

        id_to_name = {v: k for k, v in merged.items()}
        return cls(blocks, items, merged, id_to_name)

    def block_id_of(self, name: str) -> int:
        return self._block_to_id[name]

    def item_id_of(self, name: str) -> int:
        return self._item_to_id[name]

    def block_ids(self) -> dict[str, int]:
        return dict(self._block_to_id)

    def item_ids(self) -> dict[str, int]:
        return dict(self._item_to_id)

    # legacy merged view
    def id_of(self, name: str) -> int:
        return self._name_to_id[name]

    def name_of(self, ident: int) -> str:
        return self._id_to_name[ident]

    @property
    def size(self) -> int:
        return len(self._name_to_id)

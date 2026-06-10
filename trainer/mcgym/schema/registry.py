"""id to name registry for blocks and items"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


# blocks and items are separate id spaces on the gym side, same name can map to
# different ints (oak_log is block 49 but item 134). voxels and target_block carry
# block ids, inventory carries item ids
@dataclass
class Registry:
    _block_to_id: dict[str, int]
    _item_to_id:  dict[str, int]

    @classmethod
    def load(cls, path: pathlib.Path) -> "Registry":
        if not path.exists():
            raise FileNotFoundError(f"registry not found: {path}")
        doc = json.loads(path.read_text())

        blocks: dict[str, int] = dict(doc.get("blocks", {}))
        items:  dict[str, int] = dict(doc.get("items", {}))

        return cls(blocks, items)

    def block_id_of(self, name: str) -> int:
        return self._block_to_id[name]

    def item_id_of(self, name: str) -> int:
        return self._item_to_id[name]

    def block_ids(self) -> dict[str, int]:
        return dict(self._block_to_id)

    def item_ids(self) -> dict[str, int]:
        return dict(self._item_to_id)


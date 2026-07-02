"""Compact per-expert model store for mobile/offline sync."""

import json
import os
import re
import time

import numpy as np


GB = 1024 ** 3


def _safe_part(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return text.strip("-") or "model"


def folder_size(path):
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def memory_status(path, warning_step_bytes=None):
    step = int(warning_step_bytes or int(2.5 * GB))
    size = folder_size(path)
    if step <= 0:
        level = 0
        next_level = 0
    else:
        level = (size // step) * step
        next_level = level + step
    return {
        "path": path,
        "exists": os.path.isdir(path),
        "size_bytes": int(size),
        "size_gb": round(size / GB, 4),
        "warning_step_bytes": int(step),
        "warning_step_gb": round(step / GB, 4) if step else 0,
        "warning_level_bytes": int(level),
        "warning_level_gb": round(level / GB, 4) if level else 0,
        "next_warning_bytes": int(next_level),
        "next_warning_gb": round(next_level / GB, 4) if next_level else 0,
        "warning": bool(level),
    }


class MobileModelStore:
    """Save each DAS expert as a small standalone artifact and track folder size."""

    def __init__(self, path, warning_step_bytes=None):
        self.path = path
        self.warning_step_bytes = int(warning_step_bytes or int(2.5 * GB))

    @property
    def manifest_path(self):
        return os.path.join(self.path, "manifest.json")

    def status(self):
        return memory_status(self.path, self.warning_step_bytes)

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return {"version": 1, "models": []}
        with open(self.manifest_path) as fh:
            return json.load(fh)

    def _save_manifest(self, manifest):
        os.makedirs(self.path, exist_ok=True)
        with open(self.manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

    def manifest(self):
        return self._load_manifest()

    def export_expert(self, control_plane, eid):
        idx, rec = control_plane._find(int(eid))
        leaf = control_plane.forest.leaves[idx]
        os.makedirs(self.path, exist_ok=True)
        manifest = self._load_manifest()
        for old in manifest.get("models", []):
            if old.get("eid") == rec["eid"]:
                old_path = os.path.join(self.path, old.get("file", ""))
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass
        filename = (
            f"eid{rec['eid']}__{_safe_part(rec['tenant'])}__"
            f"{_safe_part(rec['name'])}__{leaf.weight_hash()}.npz"
        )
        rel = filename
        dest = os.path.join(self.path, rel)
        arrays = {f"W{i}": w for i, w in enumerate(leaf.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(leaf.b)})
        np.savez_compressed(
            dest,
            dims=np.asarray(leaf.dims, dtype=int),
            frozen=np.asarray([int(bool(leaf.frozen))], dtype=int),
            eid=np.asarray([int(rec["eid"])], dtype=int),
            **arrays,
        )
        row = {
            "eid": int(rec["eid"]),
            "tenant": rec["tenant"],
            "name": rec["name"],
            "specialty": rec.get("specialty"),
            "parent": rec.get("parent"),
            "hash": leaf.weight_hash(),
            "file": rel,
            "bytes": os.path.getsize(dest),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest["models"] = [m for m in manifest.get("models", []) if m.get("eid") != rec["eid"]]
        manifest["models"].append(row)
        manifest["models"].sort(key=lambda m: (str(m.get("tenant")), str(m.get("name")), int(m.get("eid", 0))))
        manifest["updated_at"] = row["saved_at"]
        self._save_manifest(manifest)
        return row

    def export_all(self, control_plane, prune_missing=True):
        if prune_missing:
            keep = {int(rec["eid"]) for rec in control_plane.experts}
            manifest = self._load_manifest()
            for old in manifest.get("models", []):
                if int(old.get("eid", -1)) not in keep:
                    old_path = os.path.join(self.path, old.get("file", ""))
                    if os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except OSError:
                            pass
            manifest["models"] = [m for m in manifest.get("models", []) if int(m.get("eid", -1)) in keep]
            self._save_manifest(manifest)
        rows = [self.export_expert(control_plane, rec["eid"]) for rec in control_plane.experts]
        return {"saved": rows, "status": self.status()}

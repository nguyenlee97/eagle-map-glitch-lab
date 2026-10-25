"""
eagle_map_glitch_lab.py — Automated 3D Map Walkability, Multi-Layer 2.5D Swarm Capsule Bot &
Geometric Glitch Debugger for Project Eagle (Genshin / Teyvat & Custom 3D Maps).

Capabilities:
1. Ingests either compiled GTA SA binary packages (.IMG / .COL3 / .DFF / .IPL)
   or embedded 3D inspection bundles (scene_data.json / .html).
2. Runs 5 Automated Geometric & Physics Glitch Scanners (with Top-Surface Exposure Filtering):
   - Scanner A: Multi-Layer 2.5D Swarm Capsule Walker & Unreachable Island / Barrier Chokepoint Finder
   - Scanner B: Multi-Point Footprint Raycast for Floating Models & Slope Overhangs
   - Scanner C: Visual (.DFF) vs Physics (.COL3) Mismatch (Ghost Floors, Invisible Walls, Exposed Inverted Normals)
   - Scanner D: Void Freefall Traps & BigWorld / Tile Seam Chasms (E=+241m, N=-227m, M0<->M1)
   - Scanner E: Step-Lip Blockers (0.46m..1.15m) & Real Overhead Headroom Traps (< 1.72m)
3. Exports structured JSON diagnostics (`glitch_audit_report.json`) AND builds a self-contained
   Interactive 3D Glitch Lab Studio HTML (`eagle-3d-map-walker-glitch-lab-2026-10-24.html`).
"""

from __future__ import annotations

import argparse
import base64
import collections
import gzip
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

THREE_ROOT = Path(r"C:\Users\LENOVO\Desktop\GTA\tools\skin-viewer\node_modules\three")

# Physics constants matching GTA SA / Aether player capsule
PHYS_RADIUS = 0.32
PHYS_HEIGHT = 1.72
PHYS_EYE = 1.58
PHYS_MAX_STEP = 0.45
PHYS_MAX_DROP = 0.65
PHYS_JUMP_STEP = 0.95
PHYS_JUMP_DROP = 1.65
PHYS_SLOPE_COS = math.sqrt(0.5)  # cos(45 deg) = 0.70710678
PHYS_JUMP_SLOPE_COS = 0.62       # ~51.6 deg scramble


def load_scene_from_html_or_json(source_path: Path) -> dict[str, Any]:
    """Extract scene_data dict from either an inspection HTML or scene_data.json."""
    text = source_path.read_text(encoding="utf-8")
    if source_path.suffix.lower() == ".json":
        return json.loads(text)
    m = re.search(r'<script\s+id=["\']scene-data["\'][^>]*>([^<]+)</script>', text)
    if not m:
        raise ValueError(f"Could not find <script id='scene-data'> in {source_path}")
    b64_payload = m.group(1).strip()
    raw_bytes = gzip.decompress(base64.b64decode(b64_payload))
    return json.loads(raw_bytes.decode("utf-8"))


class SpatialTriangleIndex:
    """Fast 2D grid index over 3D triangles for raycasting, capsule support, and wall checks."""

    def __init__(self, cell_size: float = 8.0):
        self.cell_size = float(cell_size)
        self.grid: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        self.v0: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.v1: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.v2: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.normals: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.areas: np.ndarray = np.zeros((0,), dtype=np.float32)
        self.centroids: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.min_xyz: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.max_xyz: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.model_ids: np.ndarray = np.zeros((0,), dtype=np.int32)
        self.sectors: list[str] = []
        self.surfaces: list[Any] = []
        self.walkable: np.ndarray = np.zeros((0,), dtype=bool)

    def build(self, triangle_records: list[dict[str, Any]]):
        n = len(triangle_records)
        if n == 0:
            return
        self.v0 = np.array([r["a"] for r in triangle_records], dtype=np.float32)
        self.v1 = np.array([r["b"] for r in triangle_records], dtype=np.float32)
        self.v2 = np.array([r["c"] for r in triangle_records], dtype=np.float32)
        self.normals = np.array([r["n"] for r in triangle_records], dtype=np.float32)
        self.areas = np.array([r["area"] for r in triangle_records], dtype=np.float32)
        self.centroids = (self.v0 + self.v1 + self.v2) / 3.0
        self.min_xyz = np.minimum(np.minimum(self.v0, self.v1), self.v2)
        self.max_xyz = np.maximum(np.maximum(self.v0, self.v1), self.v2)
        self.model_ids = np.array([r["id"] for r in triangle_records], dtype=np.int32)
        self.sectors = [r["sector"] for r in triangle_records]
        self.surfaces = [r.get("surface", 0) for r in triangle_records]
        self.walkable = np.array([r["walkable"] for r in triangle_records], dtype=bool)

        cs = self.cell_size
        gx_min = np.floor(self.min_xyz[:, 0] / cs).astype(np.int32)
        gx_max = np.floor(self.max_xyz[:, 0] / cs).astype(np.int32)
        gy_min = np.floor(self.min_xyz[:, 1] / cs).astype(np.int32)
        gy_max = np.floor(self.max_xyz[:, 1] / cs).astype(np.int32)

        for idx in range(n):
            for gx in range(gx_min[idx], gx_max[idx] + 1):
                for gy in range(gy_min[idx], gy_max[idx] + 1):
                    self.grid[(gx, gy)].append(idx)

    def query_point(self, x: float, y: float) -> list[int]:
        cs = self.cell_size
        return self.grid.get((int(math.floor(x / cs)), int(math.floor(y / cs))), [])

    def query_radius(self, x: float, y: float, r: float) -> set[int]:
        cs = self.cell_size
        gx0 = int(math.floor((x - r) / cs))
        gx1 = int(math.floor((x + r) / cs))
        gy0 = int(math.floor((y - r) / cs))
        gy1 = int(math.floor((y + r) / cs))
        out: set[int] = set()
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                lst = self.grid.get((gx, gy))
                if lst:
                    out.update(lst)
        return out

    def raycast_vertical(
        self,
        x: float,
        y: float,
        z_top: float = 10000.0,
        z_bottom: float = -10000.0,
        only_walkable: bool = False,
        ignore_model_id: int | None = None,
        ignore_tri_indices: set[int] | None = None,
    ) -> list[tuple[float, int]]:
        """Return all triangle hits at (x, y) with z in [z_bottom, z_top], sorted highest Z first."""
        cand = self.query_point(x, y)
        if not cand:
            return []
        hits: list[tuple[float, int]] = []
        for idx in cand:
            if only_walkable and not self.walkable[idx]:
                continue
            if ignore_model_id is not None and self.model_ids[idx] == ignore_model_id:
                continue
            if ignore_tri_indices is not None and idx in ignore_tri_indices:
                continue
            if self.max_xyz[idx, 2] < z_bottom or self.min_xyz[idx, 2] > z_top:
                continue
            if x < self.min_xyz[idx, 0] - 1e-5 or x > self.max_xyz[idx, 0] + 1e-5:
                continue
            if y < self.min_xyz[idx, 1] - 1e-5 or y > self.max_xyz[idx, 1] + 1e-5:
                continue
            a = self.v0[idx]
            b = self.v1[idx]
            c = self.v2[idx]
            denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denom) < 1e-10:
                continue
            u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denom
            v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denom
            if u < -1e-5 or v < -1e-5 or (u + v) > 1.0 + 1e-5:
                continue
            z = float(u * a[2] + v * b[2] + (1.0 - u - v) * c[2])
            if z_bottom <= z <= z_top:
                hits.append((z, idx))
        hits.sort(key=lambda item: item[0], reverse=True)
        return hits


def build_collision_index(scene_data: dict[str, Any]) -> SpatialTriangleIndex:
    records: list[dict[str, Any]] = []
    for col in scene_data.get("collision", []):
        pos = col["positions"]
        indices = col.get("indices") or list(range(len(pos) // 3))
        surfaces = col.get("surfaces") or []
        cid = int(col["id"])
        sector = str(col.get("sector", "M1"))
        for k in range(0, len(indices), 3):
            i0, i1, i2 = indices[k] * 3, indices[k + 1] * 3, indices[k + 2] * 3
            a = (float(pos[i0]), float(pos[i0 + 1]), float(pos[i0 + 2]))
            b = (float(pos[i1]), float(pos[i1 + 1]), float(pos[i1 + 2]))
            c = (float(pos[i2]), float(pos[i2 + 1]), float(pos[i2 + 2]))
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm_len < 1e-9:
                continue
            area = 0.5 * norm_len
            nx /= norm_len
            ny /= norm_len
            nz /= norm_len
            surf = surfaces[k // 3] if (k // 3) < len(surfaces) else 0
            is_water = isinstance(surf, str) and bool(re.search(r"water|swim|lake", surf, re.I))
            records.append(
                {
                    "a": a,
                    "b": b,
                    "c": c,
                    "n": (nx, ny, nz),
                    "area": area,
                    "id": cid,
                    "sector": sector,
                    "surface": surf,
                    "walkable": (nz >= PHYS_SLOPE_COS) and (not is_water),
                }
            )
    idx = SpatialTriangleIndex(cell_size=8.0)
    idx.build(records)
    return idx


def build_visual_index(scene_data: dict[str, Any]) -> tuple[SpatialTriangleIndex, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    near_models: list[dict[str, Any]] = []
    for mdl in scene_data.get("models", []):
        if mdl.get("kind") != "near":
            continue
        near_models.append(mdl)
        pos = mdl["positions"]
        indices = mdl.get("indices") or list(range(len(pos) // 3))
        mid = int(mdl["id"])
        sector = str(mdl.get("sector", "M1"))
        for k in range(0, len(indices), 3):
            i0, i1, i2 = indices[k] * 3, indices[k + 1] * 3, indices[k + 2] * 3
            a = (float(pos[i0]), float(pos[i0 + 1]), float(pos[i0 + 2]))
            b = (float(pos[i1]), float(pos[i1 + 1]), float(pos[i1 + 2]))
            c = (float(pos[i2]), float(pos[i2 + 1]), float(pos[i2 + 2]))
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm_len < 1e-9:
                continue
            area = 0.5 * norm_len
            nx /= norm_len
            ny /= norm_len
            nz /= norm_len
            records.append(
                {
                    "a": a,
                    "b": b,
                    "c": c,
                    "n": (nx, ny, nz),
                    "area": area,
                    "id": mid,
                    "sector": sector,
                    "surface": mdl.get("name", ""),
                    "walkable": nz >= PHYS_SLOPE_COS,
                }
            )
    idx = SpatialTriangleIndex(cell_size=8.0)
    idx.build(records)
    return idx, near_models


def is_buried_underground(col_idx: SpatialTriangleIndex, x: float, y: float, z: float, clearance: float = 1.5) -> bool:
    """Return True if (x, y, z) lies underneath an exposed upward-facing terrain surface."""
    above_hits = col_idx.raycast_vertical(x, y, z_top=1000.0, z_bottom=z + clearance)
    return any(col_idx.normals[h_idx, 2] > 0.45 for _, h_idx in above_hits)


def decompose_mesh_components(model: dict[str, Any], weld_tol: float = 0.05) -> list[dict[str, Any]]:
    """Decompose a Near .DFF model into connected 3D triangle components using spatial vertex welding."""
    pos = np.array(model["positions"], dtype=np.float32).reshape(-1, 3)
    if len(pos) == 0:
        return []
    indices = model.get("indices")
    if indices:
        tris = np.array(indices, dtype=np.int32).reshape(-1, 3)
    else:
        tris = np.arange(len(pos), dtype=np.int32).reshape(-1, 3)

    num_tris = len(tris)
    if num_tris == 0:
        return []

    q_verts = np.round(pos / weld_tol).astype(np.int64)
    vert_to_cluster: dict[tuple[int, int, int], int] = {}
    uf_parent = list(range(num_tris))

    def find(i: int) -> int:
        while uf_parent[i] != i:
            uf_parent[i] = uf_parent[uf_parent[i]]
            i = uf_parent[i]
        return i

    def union(i: int, j: int):
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                uf_parent[rj] = ri
            else:
                uf_parent[ri] = rj

    for t_idx in range(num_tris):
        for v_idx in tris[t_idx]:
            key = (int(q_verts[v_idx, 0]), int(q_verts[v_idx, 1]), int(q_verts[v_idx, 2]))
            prev = vert_to_cluster.get(key)
            if prev is None:
                vert_to_cluster[key] = t_idx
            else:
                union(t_idx, prev)

    comp_tris: dict[int, list[int]] = collections.defaultdict(list)
    for t_idx in range(num_tris):
        comp_tris[find(t_idx)].append(t_idx)

    components: list[dict[str, Any]] = []
    for _, t_list in comp_tris.items():
        sub_tris = tris[t_list]
        u_verts = np.unique(sub_tris.ravel())
        pts = pos[u_verts]
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        span = max_pt - min_pt
        center = 0.5 * (min_pt + max_pt)
        base_mask = pts[:, 2] <= (min_pt[2] + min(0.35, max(0.08, span[2] * 0.15)))
        base_pts = pts[base_mask] if np.any(base_mask) else pts
        components.append(
            {
                "model_id": int(model["id"]),
                "model_name": str(model.get("name", "")),
                "sector": str(model.get("sector", "")),
                "tri_count": len(t_list),
                "vert_count": len(u_verts),
                "min": min_pt.tolist(),
                "max": max_pt.tolist(),
                "span": span.tolist(),
                "center": center.tolist(),
                "base_pts": base_pts,
            }
        )
    return components


def scan_floating_and_overhang_models(
    scene_data: dict[str, Any],
    col_idx: SpatialTriangleIndex,
    near_models: list[dict[str, Any]],
    water_level: float = -58.0,
) -> list[dict[str, Any]]:
    """
    Scanner B: Multi-Point Footprint Raycast for Exposed Floating Models & Slope Overhangs.
    Filters out underground internal HLOD rock shells buried beneath the top terrain surface.
    """
    glitches: list[dict[str, Any]] = []

    for mdl in near_models:
        comps = decompose_mesh_components(mdl, weld_tol=0.05)
        comps.sort(key=lambda c: c["tri_count"], reverse=True)
        for rank, comp in enumerate(comps):
            sx, sy, sz = comp["span"]
            horiz_span = max(sx, sy)
            if mdl["sector"] == "M1" and rank == 0 and horiz_span > 55.0:
                continue
            if horiz_span > 45.0 or sz < 0.35 or comp["tri_count"] < 6:
                continue
            min_x, min_y, min_z = comp["min"]
            max_x, max_y, max_z = comp["max"]
            cx, cy, cz = comp["center"]
            if max_z < water_level + 0.8:
                continue

            # Filter out underground HLOD sub-shells buried underneath the top terrain surface!
            if is_buried_underground(col_idx, cx, cy, max_z, clearance=0.40):
                continue

            bpts = comp["base_pts"]
            sample_xy: list[tuple[float, float, float]] = [(cx, cy, min_z)]
            if len(bpts) >= 4:
                idx_xmin = int(np.argmin(bpts[:, 0]))
                idx_xmax = int(np.argmax(bpts[:, 0]))
                idx_ymin = int(np.argmin(bpts[:, 1]))
                idx_ymax = int(np.argmax(bpts[:, 1]))
                for ii in {idx_xmin, idx_xmax, idx_ymin, idx_ymax}:
                    sample_xy.append((float(bpts[ii, 0]), float(bpts[ii, 1]), float(bpts[ii, 2])))

            gaps: list[float] = []
            for px, py, pz in sample_xy:
                # Raycast from pz + 0.05 downwards to find the supporting ground below the component's base
                hits = col_idx.raycast_vertical(px, py, z_top=pz + 0.05, z_bottom=pz - 15.0)
                walkable_hits = [h for h in hits if col_idx.normals[h[1], 2] > 0.35]
                if walkable_hits:
                    ground_z = walkable_hits[0][0]
                    gaps.append(max(0.0, pz - ground_z))

            if len(gaps) < max(1, len(sample_xy) // 2):
                continue
            min_gap = float(min(gaps))
            max_gap = float(max(gaps))

            if 0.25 <= min_gap <= 6.5:
                glitches.append(
                    {
                        "type": "FLOATING_MODEL",
                        "subtype": "FULL_FLOATING",
                        "severity": "high" if min_gap > 0.65 else "medium",
                        "sector": comp["sector"],
                        "model_id": comp["model_id"],
                        "model_name": comp["model_name"],
                        "pos": [round(cx, 2), round(cy, 2), round(min_z, 2)],
                        "span": [round(sx, 2), round(sy, 2), round(sz, 2)],
                        "min_gap_m": round(min_gap, 3),
                        "max_gap_m": round(max_gap, 3),
                        "tri_count": comp["tri_count"],
                        "title": f"Hở toàn bộ chân đế ({min_gap:.2f}m) · {comp['model_name']} (#{comp['model_id']})",
                        "detail": (
                            f"Cụm vật thể lộ thiên ({comp['tri_count']} tam giác, {sx:.1f}×{sy:.1f}×{sz:.1f}m) "
                            f"tại E={cx:.1f}, N={cy:.1f}, U={min_z:.2f}m có điểm thấp nhất bay lơ lửng "
                            f"cách mặt đất COL bên dưới {min_gap:.2f}m (tối đa {max_gap:.2f}m)."
                        ),
                        "fix_hint": f"Snap hạ Z của cụm xuống {-min_gap:.2f}m hoặc đóng chân đế (Foundation Plinth).",
                    }
                )
            elif min_gap <= 0.12 and 0.60 <= max_gap <= 4.5 and horiz_span <= 14.0 and comp["sector"] == "M0":
                glitches.append(
                    {
                        "type": "FLOATING_MODEL",
                        "subtype": "SLOPE_CORNER_OVERHANG",
                        "severity": "medium",
                        "sector": comp["sector"],
                        "model_id": comp["model_id"],
                        "model_name": comp["model_name"],
                        "pos": [round(cx, 2), round(cy, 2), round(min_z, 2)],
                        "span": [round(sx, 2), round(sy, 2), round(sz, 2)],
                        "min_gap_m": round(min_gap, 3),
                        "max_gap_m": round(max_gap, 3),
                        "tri_count": comp["tri_count"],
                        "title": f"Mép đế bị hẫng trên bậc/dốc ({max_gap:.2f}m) · {comp['model_name']} (#{comp['model_id']})",
                        "detail": (
                            f"Một bên chân đế chạm đất (gap {min_gap:.2f}m) nhưng mép đối diện bị hẫng "
                            f"{max_gap:.2f}m tại E={cx:.1f}, N={cy:.1f}, U={min_z:.2f}m."
                        ),
                        "fix_hint": f"Dịch cụm vào vùng nền phẳng hoặc thêm bệ đỡ cao {max_gap:.2f}m.",
                    }
                )

    for f_idx, fol in enumerate(scene_data.get("foliage", [])):
        if fol.get("kind") != "pine":
            continue
        fx, fy, fz = float(fol["x"]), float(fol["y"]), float(fol["z"])
        r = 0.45
        edge_gaps: list[float] = []
        for ox, oy in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
            hits = col_idx.raycast_vertical(fx + ox, fy + oy, z_top=fz + 1.0, z_bottom=fz - 5.0)
            if hits:
                edge_gaps.append(max(0.0, fz - hits[0][0]))
            else:
                edge_gaps.append(2.0)
        max_fgap = max(edge_gaps) if edge_gaps else 0.0
        if max_fgap > 0.38:
            glitches.append(
                {
                    "type": "FLOATING_MODEL",
                    "subtype": "FOLIAGE_ROOT_OVERHANG",
                    "severity": "low",
                    "sector": "M1",
                    "model_id": int(fol.get("collision_id", 6730)),
                    "model_name": f"pine_tree_{f_idx}",
                    "pos": [round(fx, 2), round(fy, 2), round(fz, 2)],
                    "span": [round(float(fol.get("width", 2.4)), 2), round(float(fol.get("width", 2.4)), 2), round(float(fol.get("height", 9.0)), 2)],
                    "min_gap_m": 0.0,
                    "max_gap_m": round(max_fgap, 3),
                    "tri_count": 12,
                    "title": f"Gốc thông hở mép dốc ({max_fgap:.2f}m) · Tile #{fol.get('collision_id', 6730)}",
                    "detail": f"Tâm gốc cây thông chạm đất nhưng mép thân (r=0.45m) hở {max_fgap:.2f}m trên sườn dốc.",
                    "fix_hint": f"Hạ gốc thông thêm {-max_fgap:.2f}m vào trong mặt đất.",
                }
            )

    deduped: list[dict[str, Any]] = []
    glitches.sort(key=lambda g: (0 if g["severity"] == "high" else 1 if g["severity"] == "medium" else 2, -g["max_gap_m"]))
    for g in glitches:
        gx, gy, gz = g["pos"]
        if any(math.hypot(gx - d["pos"][0], gy - d["pos"][1]) < 12.0 and abs(gz - d["pos"][2]) < 4.0 for d in deduped):
            continue
        deduped.append(g)
    return deduped[:25]


def scan_visual_vs_collision_mismatches(
    vis_idx: SpatialTriangleIndex,
    col_idx: SpatialTriangleIndex,
    water_level: float = -58.0,
) -> tuple[list[dict[str, Any]], int]:
    """
    Scanner C: Visual (.DFF) vs Physics (.COL3) Mismatch on Exposed Top Surfaces:
      1. INVERTED_COL_NORMAL: Exposed top-surface .COL3 faces pointing downward (nz < -0.55) -> causes fall-through!
      2. GHOST_FLOOR: Exposed upward-facing .DFF triangles with NO supporting .COL3 within ±0.45m.
      3. INVISIBLE_WALL: Exposed steep .COL3 faces with no .DFF mesh nearby.
    Also counts buried underground inverted skirt triangles separately.
    """
    glitches: list[dict[str, Any]] = []
    buried_inverted_count = 0

    for idx in range(len(col_idx.normals)):
        nz = float(col_idx.normals[idx, 2])
        area = float(col_idx.areas[idx])
        if nz < -0.55 and area >= 1.2:
            cx, cy, cz = col_idx.centroids[idx]
            if cz < water_level + 0.5:
                continue
            # Check all hits at (cx, cy) from sky down to cz - 0.5m
            hits = col_idx.raycast_vertical(float(cx), float(cy), z_top=1000.0, z_bottom=float(cz) - 0.5)
            has_upward_above_or_same = any(col_idx.normals[h_idx, 2] > 0.45 and hz >= cz - 0.35 for hz, h_idx in hits)
            if has_upward_above_or_same:
                buried_inverted_count += 1
                continue
            glitches.append(
                {
                    "type": "COLLISION_MISMATCH",
                    "subtype": "INVERTED_COL_NORMAL",
                    "severity": "high",
                    "sector": col_idx.sectors[idx],
                    "model_id": int(col_idx.model_ids[idx]),
                    "model_name": f"COL_{col_idx.model_ids[idx]}",
                    "pos": [round(float(cx), 2), round(float(cy), 2), round(float(cz), 2)],
                    "nz": round(nz, 3),
                    "area_m2": round(area, 2),
                    "title": f"Mặt COL lộ thiên ngược pháp tuyến (nz={nz:.2f}) · COL #{col_idx.model_ids[idx]}",
                    "detail": (
                        f"Tam giác va chạm lộ thiên trên bề mặt tại E={cx:.1f}, N={cy:.1f}, U={cz:.2f}m ({area:.1f}m²) "
                        f"có pháp tuyến hướng xuống đất (nz={nz:.2f}) và KHÔNG có mặt đỡ hướng lên nào che phía trên — "
                        f"bước chân vào đây sẽ rơi xuyên qua mặt đất!"
                    ),
                    "fix_hint": "Đảo thứ tự đỉnh tam giác (winding order) trong .COL3 hoặc xóa mặt skirt bị lật.",
                }
            )

    for idx in range(len(vis_idx.normals)):
        nz = float(vis_idx.normals[idx, 2])
        area = float(vis_idx.areas[idx])
        if nz < 0.75 or area < 2.0:
            continue
        cx, cy, cz = vis_idx.centroids[idx]
        if cz < water_level + 0.8 or cz > -5.0:
            continue
        hits = col_idx.raycast_vertical(float(cx), float(cy), z_top=float(cz) + 0.50, z_bottom=float(cz) - 0.60)
        if not hits:
            glitches.append(
                {
                    "type": "COLLISION_MISMATCH",
                    "subtype": "GHOST_FLOOR",
                    "severity": "high",
                    "sector": vis_idx.sectors[idx],
                    "model_id": int(vis_idx.model_ids[idx]),
                    "model_name": str(vis_idx.surfaces[idx]),
                    "pos": [round(float(cx), 2), round(float(cy), 2), round(float(cz), 2)],
                    "nz": round(nz, 3),
                    "area_m2": round(area, 2),
                    "title": f"Sàn ma không có COL ({area:.1f}m²) · {vis_idx.surfaces[idx]} (#{vis_idx.model_ids[idx]})",
                    "detail": (
                        f"Mặt .DFF nằm ngang (nz={nz:.2f}, {area:.1f}m²) tại E={cx:.1f}, N={cy:.1f}, U={cz:.2f}m "
                        f"hiển thị như mặt sàn/đường nhưng không có mặt .COL3 nào trong khoảng ±0.5m."
                    ),
                    "fix_hint": "Đồng bộ xuất tam giác này vào .COL3.",
                }
            )

    deduped: list[dict[str, Any]] = []
    glitches.sort(key=lambda g: -g.get("area_m2", 0.0))
    for g in glitches:
        gx, gy, gz = g["pos"]
        if any(
            d["subtype"] == g["subtype"]
            and math.hypot(gx - d["pos"][0], gy - d["pos"][1]) < 22.0
            for d in deduped
        ):
            continue
        deduped.append(g)
    return deduped[:25], buried_inverted_count


def scan_void_traps_and_seams(
    col_idx: SpatialTriangleIndex,
    water_level: float = -58.0,
) -> list[dict[str, Any]]:
    """
    Scanner D: Exposed Surface Void Freefall Traps & BigWorld / Tile Seam Chasms.
    Only checks open edges on the EXPOSED TOP SURFACE (ignores buried internal HLOD skirts).
    """
    glitches: list[dict[str, Any]] = []
    n_tris = len(col_idx.v0)
    if n_tris == 0:
        return glitches

    weld_scale = 50.0  # 2cm tolerance
    q0 = np.round(col_idx.v0 * weld_scale).astype(np.int64)
    q1 = np.round(col_idx.v1 * weld_scale).astype(np.int64)
    q2 = np.round(col_idx.v2 * weld_scale).astype(np.int64)

    edge_map: dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[tuple[int, int]]] = collections.defaultdict(list)
    for t_idx in range(n_tris):
        pts = (
            (int(q0[t_idx, 0]), int(q0[t_idx, 1]), int(q0[t_idx, 2])),
            (int(q1[t_idx, 0]), int(q1[t_idx, 1]), int(q1[t_idx, 2])),
            (int(q2[t_idx, 0]), int(q2[t_idx, 1]), int(q2[t_idx, 2])),
        )
        for e_i in range(3):
            p_a = pts[e_i]
            p_b = pts[(e_i + 1) % 3]
            if p_a == p_b:
                continue
            key = (p_a, p_b) if p_a < p_b else (p_b, p_a)
            edge_map[key].append((t_idx, e_i))

    verts_arr = (col_idx.v0, col_idx.v1, col_idx.v2)

    for key, owners in edge_map.items():
        if len(owners) != 1:
            continue
        t_idx, e_i = owners[0]
        if not col_idx.walkable[t_idx]:
            continue
        va = verts_arr[e_i][t_idx]
        vb = verts_arr[(e_i + 1) % 3][t_idx]
        vc = verts_arr[(e_i + 2) % 3][t_idx]
        mx = 0.5 * float(va[0] + vb[0])
        my = 0.5 * float(va[1] + vb[1])
        mz = 0.5 * float(va[2] + vb[2])

        if abs(mx) > 748.0 or abs(my) > 748.0 or mz < water_level + 0.8:
            continue

        # Must be on the exposed top surface (no walkable terrain covering it from above!)
        if is_buried_underground(col_idx, mx, my, mz, clearance=0.80):
            continue

        edge_len = math.hypot(float(vb[0] - va[0]), float(vb[1] - va[1]))
        if edge_len < 0.60:
            continue

        ex, ey = float(vb[0] - va[0]), float(vb[1] - va[1])
        nx_2d, ny_2d = -ey / edge_len, ex / edge_len
        to_c_x, to_c_y = float(vc[0]) - mx, float(vc[1]) - my
        if (nx_2d * to_c_x + ny_2d * to_c_y) > 0:
            nx_2d, ny_2d = -nx_2d, -ny_2d

        px = mx + nx_2d * 0.35
        py = my + ny_2d * 0.35

        # If there is a wall blocking the player from stepping over this edge, it's safe
        wall_hits = col_idx.raycast_vertical(px, py, z_top=mz + 1.8, z_bottom=mz + 0.35)
        if wall_hits:
            continue

        down_hits = col_idx.raycast_vertical(px, py, z_top=mz + 0.45, z_bottom=-1000.0)
        if not down_hits:
            # Check if this is right along the BigWorld block seams (N = -227 or E = +241)
            near_bigworld_seam = (abs(my - (-227.0)) < 8.0) or (abs(mx - 241.0) < 8.0)
            subtype = "BIGWORLD_SEAM_CHASM" if near_bigworld_seam else "VOID_FREEFALL_HOLE"
            glitches.append(
                {
                    "type": "VOID_TRAP",
                    "subtype": subtype,
                    "severity": "high",
                    "sector": col_idx.sectors[t_idx],
                    "model_id": int(col_idx.model_ids[t_idx]),
                    "model_name": f"COL_{col_idx.model_ids[t_idx]}",
                    "pos": [round(mx, 2), round(my, 2), round(mz, 2)],
                    "edge_len_m": round(edge_len, 2),
                    "drop_m": 999.0,
                    "title": (
                        f"Khe đứt gãy biên BigWorld làm rơi tự do ({edge_len:.1f}m) · COL #{col_idx.model_ids[t_idx]}"
                        if near_bigworld_seam
                        else f"Lỗ thủng bề mặt COL làm rơi tự do ({edge_len:.1f}m) · COL #{col_idx.model_ids[t_idx]}"
                    ),
                    "detail": (
                        f"Cạnh hở lộ thiên tại E={mx:.1f}, N={my:.1f}, U={mz:.2f}m (dài {edge_len:.1f}m): "
                        f"bước ra ngoài 0.35m tới ({px:.1f}, {py:.1f}) không có bất kỳ mặt .COL3 nào bên dưới (rơi thẳng xuống void -∞)."
                    ),
                    "fix_hint": (
                        "Khâu kín dải biên giữa 2 ô Stages_BigWorld (N=-227 hoặc E=+241) trước khi cắt tile 256m."
                        if near_bigworld_seam
                        else "Hàn kín đỉnh biên hoặc đóng nắp collision tại cạnh hở này."
                    ),
                }
            )
        else:
            drop = mz - down_hits[0][0]
            near_seam = (
                abs(my - (-227.0)) < 6.0
                or abs(mx - 241.0) < 6.0
                or col_idx.sectors[t_idx] == "M0"
                or any(abs(abs(coord) - b) < 1.2 for coord in (mx, my) for b in (0.0, 256.0, 512.0))
            )
            if near_seam and 0.65 <= drop <= 6.0:
                glitches.append(
                    {
                        "type": "VOID_TRAP",
                        "subtype": "SEAM_STEP_DISCONTINUITY",
                        "severity": "high" if drop > 1.5 else "medium",
                        "sector": col_idx.sectors[t_idx],
                        "model_id": int(col_idx.model_ids[t_idx]),
                        "model_name": f"COL_{col_idx.model_ids[t_idx]}",
                        "pos": [round(mx, 2), round(my, 2), round(mz, 2)],
                        "edge_len_m": round(edge_len, 2),
                        "drop_m": round(drop, 2),
                        "title": f"Bậc gãy ranh giới mảnh ({drop:.2f}m) · COL #{col_idx.model_ids[t_idx]}",
                        "detail": (
                            f"Tại ranh giới E={mx:.1f}, N={my:.1f}, U={mz:.2f}m, mặt đất COL tụt bậc đột ngột "
                            f"{drop:.2f}m (> ngưỡng bước 0.45m), chặn người chơi đi theo chiều ngược lại."
                        ),
                        "fix_hint": "Làm mượt cao độ (Terrain Smoothing) qua đường biên.",
                    }
                )

    deduped: list[dict[str, Any]] = []
    glitches.sort(key=lambda g: (0 if g["subtype"] == "BIGWORLD_SEAM_CHASM" else 1, -g["edge_len_m"]))
    for g in glitches:
        gx, gy, gz = g["pos"]
        if any(
            d["subtype"] == g["subtype"]
            and math.hypot(gx - d["pos"][0], gy - d["pos"][1]) < 22.0
            for d in deduped
        ):
            continue
        deduped.append(g)
    return deduped[:30]


def run_swarm_walkability_and_chokepoint_scanner(
    scene_data: dict[str, Any],
    col_idx: SpatialTriangleIndex,
    water_level: float = -58.0,
    grid_step: float = 2.5,
) -> dict[str, Any]:
    """
    Scanner A & E (Multi-Layer 2.5D Navigation Graph):
    - Supports multiple vertical walkable layers per (gx, gy) cell (so bridges, city arches,
      and Whispering Woods rock overhangs at z=-15m don't shadow the walkable road at z=-57.7m!).
    - Runs Swarm BFS Flood-Fill across 3D nodes (gx, gy, layer_idx).
    - Pinpoints exact Barrier Chokepoints (Break Point -> Resume Point) on matching elevation layers!
    """
    half_extent = 765.0
    side = int(round((2.0 * half_extent) / grid_step)) + 1
    origin = -half_extent

    cell_hits: dict[tuple[int, int], list[tuple[float, float, int, int]]] = collections.defaultdict(list)

    v0 = col_idx.v0
    v1 = col_idx.v1
    v2 = col_idx.v2
    normals = col_idx.normals
    min_xyz = col_idx.min_xyz
    max_xyz = col_idx.max_xyz
    model_ids = col_idx.model_ids

    ix_min = np.clip(np.ceil((min_xyz[:, 0] - origin) / grid_step).astype(np.int32), 0, side - 1)
    ix_max = np.clip(np.floor((max_xyz[:, 0] - origin) / grid_step).astype(np.int32), 0, side - 1)
    iy_min = np.clip(np.ceil((min_xyz[:, 1] - origin) / grid_step).astype(np.int32), 0, side - 1)
    iy_max = np.clip(np.floor((max_xyz[:, 1] - origin) / grid_step).astype(np.int32), 0, side - 1)

    for t_idx in range(len(v0)):
        x0, x1 = int(ix_min[t_idx]), int(ix_max[t_idx])
        y0, y1 = int(iy_min[t_idx]), int(iy_max[t_idx])
        if x0 > x1 or y0 > y1:
            continue
        a = v0[t_idx]
        b = v1[t_idx]
        c = v2[t_idx]
        denom = float((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
        if abs(denom) < 1e-9:
            continue
        inv_denom = 1.0 / denom
        nz = float(normals[t_idx, 2])
        mid = int(model_ids[t_idx])
        b1_c1 = float(b[1] - c[1])
        c0_b0 = float(c[0] - b[0])
        c1_a1 = float(c[1] - a[1])
        a0_c0 = float(a[0] - c[0])
        cx_v, cy_v = float(c[0]), float(c[1])
        az, bz, cz = float(a[2]), float(b[2]), float(c[2])

        for gx in range(x0, x1 + 1):
            px = origin + gx * grid_step
            dx = px - cx_v
            for gy in range(y0, y1 + 1):
                py = origin + gy * grid_step
                dy = py - cy_v
                u = (b1_c1 * dx + c0_b0 * dy) * inv_denom
                if u < -1e-5 or u > 1.0 + 1e-5:
                    continue
                v = (c1_a1 * dx + a0_c0 * dy) * inv_denom
                if v < -1e-5 or (u + v) > 1.0 + 1e-5:
                    continue
                z = u * az + v * bz + (1.0 - u - v) * cz
                cell_hits[(gx, gy)].append((z, nz, t_idx, mid))

    # Build Multi-Layer Walkable Nodes per (gx, gy)
    # layers_map[(gx, gy)] = list of layer dicts sorted by z ascending (merged within 1.2m)
    layers_map: dict[tuple[int, int], list[dict[str, Any]]] = {}
    headroom_glitches: list[dict[str, Any]] = []

    for (gx, gy), hits in cell_hits.items():
        hits.sort(key=lambda item: item[0])
        # Merge hits within 0.65m into a single surface cluster (keeping the one with largest nz)
        merged: list[tuple[float, float, int, int]] = []
        for h in hits:
            if not merged or (h[0] - merged[-1][0]) > 0.65:
                merged.append(h)
            else:
                if h[1] > merged[-1][1]:
                    merged[-1] = h

        cell_layers: list[dict[str, Any]] = []
        for i, (z, nz, t_idx, mid) in enumerate(merged):
            if z < water_level + 0.05 or nz < PHYS_JUMP_SLOPE_COS:
                continue
            # Check headroom to the next distinct layer above
            headroom = (merged[i + 1][0] - z) if (i + 1 < len(merged)) else 999.0
            if headroom < PHYS_HEIGHT:
                # Only report LOW_HEADROOM_TRAP if this is in M0 city or under a downward ceiling (nz_hi < 0)
                if col_idx.sectors[t_idx] == "M0" or (i + 1 < len(merged) and merged[i + 1][1] < -0.2):
                    px = origin + gx * grid_step
                    py = origin + gy * grid_step
                    headroom_glitches.append(
                        {
                            "type": "PATH_TRAP",
                            "subtype": "LOW_HEADROOM_TRAP",
                            "severity": "medium",
                            "sector": col_idx.sectors[t_idx],
                            "model_id": mid,
                            "model_name": f"COL_{mid}",
                            "pos": [round(px, 2), round(py, 2), round(z, 2)],
                            "headroom_m": round(headroom, 2),
                            "title": f"Bẫy kẹt đầu / Trần thấp ({headroom:.2f}m < 1.72m) · COL #{mid}",
                            "detail": (
                                f"Tại E={px:.1f}, N={py:.1f}, U={z:.2f}m, khoảng sáng trần chỉ cao {headroom:.2f}m "
                                f"(< chiều cao capsule 1.72m) khiến nhân vật bị vướng đầu."
                            ),
                            "fix_hint": "Nâng cao vòm/mái hiên hoặc giảm hộp va chạm phía trên.",
                        }
                    )
                continue

            cell_layers.append(
                {
                    "z": z,
                    "nz": nz,
                    "tri": t_idx,
                    "id": mid,
                    "walkable": nz >= PHYS_SLOPE_COS,
                    "scramble": nz >= PHYS_JUMP_SLOPE_COS,
                }
            )
        if cell_layers:
            layers_map[(gx, gy)] = cell_layers

    dedup_headroom: list[dict[str, Any]] = []
    headroom_glitches.sort(key=lambda g: g["headroom_m"])
    for g in headroom_glitches:
        gx, gy, gz = g["pos"]
        if any(math.hypot(gx - d["pos"][0], gy - d["pos"][1]) < 12.0 for d in dedup_headroom):
            continue
        dedup_headroom.append(g)
    dedup_headroom = dedup_headroom[:15]

    # Node key in 2.5D graph is (gx, gy, layer_idx)
    NodeKey = tuple[int, int, int]

    def find_nearest_node(tx: float, ty: float, prefer_z: float | None = None, max_r_cells: int = 10) -> NodeKey | None:
        cgx = int(round((tx - origin) / grid_step))
        cgy = int(round((ty - origin) / grid_step))
        best_node: NodeKey | None = None
        best_score = 1e18
        for dx in range(-max_r_cells, max_r_cells + 1):
            for dy in range(-max_r_cells, max_r_cells + 1):
                cell = (cgx + dx, cgy + dy)
                lyrs = layers_map.get(cell)
                if not lyrs:
                    continue
                for k_idx, lyr in enumerate(lyrs):
                    if not lyr["walkable"]:
                        continue
                    d_xy = math.hypot(dx * grid_step, dy * grid_step)
                    dz_pen = abs(lyr["z"] - prefer_z) * 0.5 if prefer_z is not None else 0.0
                    score = d_xy + dz_pen
                    if score < best_score:
                        best_score = score
                        best_node = (cell[0], cell[1], k_idx)
        return best_node

    neighbors_8 = [
        (1, 0, grid_step),
        (-1, 0, grid_step),
        (0, 1, grid_step),
        (0, -1, grid_step),
        (1, 1, grid_step * 1.41421356),
        (1, -1, grid_step * 1.41421356),
        (-1, 1, grid_step * 1.41421356),
        (-1, -1, grid_step * 1.41421356),
    ]

    def can_step_layer(f0: dict[str, Any], f1: dict[str, Any], c0: tuple[int, int], c1: tuple[int, int], dist_xy: float, allow_jump: bool = False) -> tuple[bool, str]:
        if not (f1["scramble"] if allow_jump else f1["walkable"]):
            return False, "STEEP_SLOPE"
        dz = f1["z"] - f0["z"]
        avg_nz = 0.5 * (f0["nz"] + f1["nz"])
        tan_slope = math.sqrt(max(0.0, 1.0 - avg_nz * avg_nz)) / max(0.25, avg_nz)
        continuous_rise = dist_xy * min(1.02 if not allow_jump else 1.28, tan_slope + 0.12)
        max_up = continuous_rise + (PHYS_JUMP_STEP if allow_jump else PHYS_MAX_STEP)
        max_down = dist_xy * 1.15 + (PHYS_JUMP_DROP if allow_jump else PHYS_MAX_DROP)
        if dz > max_up:
            return False, "STEP_LIP_UP"
        if dz < -max_down:
            return False, "CLIFF_DROP"

        mx = origin + 0.5 * (c0[0] + c1[0]) * grid_step
        my = origin + 0.5 * (c0[1] + c1[1]) * grid_step
        mz_expect = 0.5 * (f0["z"] + f1["z"])
        mid_hits = col_idx.raycast_vertical(
            mx,
            my,
            z_top=mz_expect + (1.35 if allow_jump else 0.85),
            z_bottom=mz_expect - (1.50 if allow_jump else 0.95),
        )
        if not mid_hits:
            return False, "MIDPOINT_GAP"
        # Pick the hit closest to mz_expect (important for multi-layer arches!)
        best_mid = min(mid_hits, key=lambda h: abs(h[0] - mz_expect))
        mid_z, mid_t = best_mid
        if mid_z < water_level + 0.05:
            return False, "WATER_SUBMERGED"
        if col_idx.normals[mid_t, 2] < (PHYS_JUMP_SLOPE_COS if allow_jump else PHYS_SLOPE_COS):
            return False, "MIDPOINT_STEEP_WALL"
        return True, "OK"

    seed_bridge = find_nearest_node(321.0, -93.1, prefer_z=-55.0)
    seed_plaza = find_nearest_node(115.0, -35.0, prefer_z=-49.0)
    seeds = [s for s in (seed_bridge, seed_plaza) if s is not None]

    walk_reachable: set[NodeKey] = set(seeds)
    walk_parent: dict[NodeKey, NodeKey | None] = {s: None for s in seeds}
    step_lip_glitches: list[dict[str, Any]] = []
    q = collections.deque(seeds)

    while q:
        gx, gy, k0 = q.popleft()
        f0 = layers_map[(gx, gy)][k0]
        for dx, dy, d_xy in neighbors_8:
            nb_cell = (gx + dx, gy + dy)
            nb_layers = layers_map.get(nb_cell)
            if not nb_layers:
                continue
            for k1, f1 in enumerate(nb_layers):
                nb_node = (nb_cell[0], nb_cell[1], k1)
                if nb_node in walk_reachable:
                    continue
                ok, reason = can_step_layer(f0, f1, (gx, gy), nb_cell, d_xy, allow_jump=False)
                if ok:
                    walk_reachable.add(nb_node)
                    walk_parent[nb_node] = (gx, gy, k0)
                    q.append(nb_node)
                elif reason == "STEP_LIP_UP" and f1["walkable"]:
                    ok_j, _ = can_step_layer(f0, f1, (gx, gy), nb_cell, d_xy, allow_jump=True)
                    if ok_j:
                        px = origin + nb_cell[0] * grid_step
                        py = origin + nb_cell[1] * grid_step
                        pz = f1["z"]
                        dz = pz - f0["z"]
                        if 0.46 <= dz <= 1.25:
                            step_lip_glitches.append(
                                {
                                    "type": "PATH_TRAP",
                                    "subtype": "STEP_LIP_BLOCKER",
                                    "severity": "medium",
                                    "sector": col_idx.sectors[f1["tri"]],
                                    "model_id": f1["id"],
                                    "model_name": f"COL_{f1['id']}",
                                    "pos": [round(px, 2), round(py, 2), round(pz, 2)],
                                    "step_dz_m": round(dz, 2),
                                    "title": f"Gờ bậc cao phải nhảy ({dz:+.2f}m) · COL #{f1['id']}",
                                    "detail": (
                                        f"Tại E={px:.1f}, N={py:.1f}, U={pz:.2f}m, hai mặt đi bộ chênh nhau {dz:+.2f}m "
                                        f"(vượt ngưỡng bước chân 0.45m), đi bộ thường bị khựng lại, phải bấm Nhảy (Space) mới lên được."
                                    ),
                                    "fix_hint": "Vát nhẹ tam giác bậc thang hoặc hạ gờ xuống <= 0.40m.",
                                }
                            )

    dedup_step_lips: list[dict[str, Any]] = []
    step_lip_glitches.sort(key=lambda g: -abs(g["step_dz_m"]))
    for g in step_lip_glitches:
        gx, gy, gz = g["pos"]
        if any(math.hypot(gx - d["pos"][0], gy - d["pos"][1]) < 18.0 for d in dedup_step_lips):
            continue
        dedup_step_lips.append(g)
    dedup_step_lips = dedup_step_lips[:20]

    # Jump-Assisted BFS
    jump_reachable: set[NodeKey] = set(seeds)
    q_jump = collections.deque(seeds)
    while q_jump:
        gx, gy, k0 = q_jump.popleft()
        f0 = layers_map[(gx, gy)][k0]
        for dx, dy, d_xy in neighbors_8:
            nb_cell = (gx + dx, gy + dy)
            nb_layers = layers_map.get(nb_cell)
            if not nb_layers:
                continue
            for k1, f1 in enumerate(nb_layers):
                nb_node = (nb_cell[0], nb_cell[1], k1)
                if nb_node in jump_reachable:
                    continue
                ok, _ = can_step_layer(f0, f1, (gx, gy), nb_cell, d_xy, allow_jump=True)
                if ok:
                    jump_reachable.add(nb_node)
                    q_jump.append(nb_node)

    # All walkable nodes across the map
    all_walkable_nodes: list[NodeKey] = [
        (gx, gy, k)
        for (gx, gy), lyrs in layers_map.items()
        for k, lyr in enumerate(lyrs)
        if lyr["walkable"]
    ]
    unvisited_walkable = set(all_walkable_nodes) - walk_reachable
    isolated_islands: list[list[NodeKey]] = []

    while unvisited_walkable:
        start_n = next(iter(unvisited_walkable))
        unvisited_walkable.remove(start_n)
        comp = [start_n]
        q_iso = collections.deque([start_n])
        while q_iso:
            gx, gy, k0 = q_iso.popleft()
            f0 = layers_map[(gx, gy)][k0]
            for dx, dy, d_xy in neighbors_8:
                nb_cell = (gx + dx, gy + dy)
                nb_layers = layers_map.get(nb_cell)
                if not nb_layers:
                    continue
                for k1, f1 in enumerate(nb_layers):
                    nb_node = (nb_cell[0], nb_cell[1], k1)
                    if nb_node in unvisited_walkable:
                        ok, _ = can_step_layer(f0, f1, (gx, gy), nb_cell, d_xy, allow_jump=False)
                        if ok:
                            unvisited_walkable.remove(nb_node)
                            comp.append(nb_node)
                            q_iso.append(nb_node)
        if len(comp) >= 25:
            isolated_islands.append(comp)

    isolated_islands.sort(key=len, reverse=True)

    main_nodes_list = list(walk_reachable)
    main_xyz = np.array(
        [[origin + n[0] * grid_step, origin + n[1] * grid_step, layers_map[(n[0], n[1])][n[2]]["z"]] for n in main_nodes_list],
        dtype=np.float32,
    )

    def diagnose_barrier_segment(p_from: tuple[float, float, float], p_to: tuple[float, float, float]) -> dict[str, Any]:
        dist_2d = math.hypot(p_to[0] - p_from[0], p_to[1] - p_from[1])
        steps = max(2, int(math.ceil(dist_2d / 0.5)))
        water_count = 0
        steep_count = 0
        void_count = 0
        min_nz = 1.0
        min_z = p_from[2]
        first_break_pt = [round(p_from[0], 2), round(p_from[1], 2), round(p_from[2], 2)]
        found_first_break = False

        for s in range(1, steps):
            t = s / steps
            sx = p_from[0] + (p_to[0] - p_from[0]) * t
            sy = p_from[1] + (p_to[1] - p_from[1]) * t
            ref_z = p_from[2] + (p_to[2] - p_from[2]) * t
            hits = col_idx.raycast_vertical(sx, sy, z_top=ref_z + 12.0, z_bottom=ref_z - 25.0)
            if not hits:
                void_count += 1
                if not found_first_break:
                    first_break_pt = [round(sx, 2), round(sy, 2), round(ref_z, 2)]
                    found_first_break = True
            else:
                # Pick hit closest to ref_z
                hz, h_idx = min(hits, key=lambda h: abs(h[0] - ref_z))
                hnz = float(col_idx.normals[h_idx, 2])
                min_z = min(min_z, hz)
                min_nz = min(min_nz, hnz)
                if hz < water_level + 0.05:
                    water_count += 1
                    if not found_first_break:
                        first_break_pt = [round(sx, 2), round(sy, 2), round(hz, 2)]
                        found_first_break = True
                elif hnz < PHYS_SLOPE_COS:
                    steep_count += 1
                    if not found_first_break:
                        first_break_pt = [round(sx, 2), round(sy, 2), round(hz, 2)]
                        found_first_break = True

        if void_count > 0 and void_count >= max(water_count, steep_count):
            cause = "MISSING_COL_VOID"
            cause_vi = f"Khe đứt gãy thiếu tam giác COL ({void_count}/{steps-1} mẫu rỗng không có mặt đất)"
        elif water_count >= steep_count and water_count > 0:
            dip_cm = round((water_level + 0.05 - min_z) * 100.0, 1)
            cause = "SUBMERGED_UNDER_WATER"
            cause_vi = f"Mặt đất trũng xuống dưới mực nước {water_level:.1f}m (Z_min={min_z:.2f}m, ngập sâu {dip_cm}cm)"
        elif steep_count > 0:
            deg = round(math.degrees(math.acos(max(-1.0, min(1.0, min_nz)))), 1)
            cause = "STEEP_SLOPE_BARRIER"
            cause_vi = f"Vách dốc đứng > 45° chặn ngang (dốc {deg}°, nz_min={min_nz:.2f})"
        else:
            cause = "STEP_DISCONTINUITY"
            cause_vi = f"Chênh lệch bậc cao độ đột ngột (dZ={abs(p_to[2] - p_from[2]):.2f}m)"

        return {
            "gap_dist_m": round(dist_2d, 2),
            "dz_m": round(p_to[2] - p_from[2], 2),
            "cause": cause,
            "cause_vi": cause_vi,
            "first_break_pos": first_break_pt,
            "min_z_along_gap": round(min_z, 2),
            "min_nz_along_gap": round(min_nz, 3),
        }

    landmarks = list(scene_data.get("landmarks") or [])
    extra_pois = [
        {"name": "Quảng Trường Phong Thần Barbatos (M0)", "point": [-35.0, 28.0]},
        {"name": "Đại Giáo Đường Mondstadt (M0)", "point": [-80.0, 68.0]},
        {"name": "Bờ Bắc Hồ Cider (M1)", "point": [120.0, 535.0]},
    ]
    all_targets = landmarks + extra_pois

    bot_routes: list[dict[str, Any]] = []
    barrier_glitches: list[dict[str, Any]] = []

    def reconstruct_node_path(end_node: NodeKey) -> list[list[float]]:
        path_nodes: list[NodeKey] = []
        curr: NodeKey | None = end_node
        while curr is not None:
            path_nodes.append(curr)
            curr = walk_parent.get(curr)
        path_nodes.reverse()
        pts: list[list[float]] = []
        for idx_c, n_key in enumerate(path_nodes):
            if idx_c == 0 or idx_c == len(path_nodes) - 1 or idx_c % 2 == 0:
                px = origin + n_key[0] * grid_step
                py = origin + n_key[1] * grid_step
                pz = layers_map[(n_key[0], n_key[1])][n_key[2]]["z"]
                pts.append([round(px, 2), round(py, 2), round(pz, 2)])
        return pts

    for tgt in all_targets:
        t_name = tgt["name"]
        tx, ty = float(tgt["point"][0]), float(tgt["point"][1])
        if abs(tx - 321.0) < 5.0 and abs(ty - (-93.1)) < 5.0:
            continue
        tgt_node = find_nearest_node(tx, ty, prefer_z=-52.0, max_r_cells=18)
        tz_ref = layers_map[(tgt_node[0], tgt_node[1])][tgt_node[2]]["z"] if tgt_node is not None else -55.0
        if tgt_node and tgt_node in walk_reachable:
            pts = reconstruct_node_path(tgt_node)
            length_m = sum(
                math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)
            )
            bot_routes.append(
                {
                    "name": t_name,
                    "target": [round(tx, 2), round(ty, 2), round(tz_ref, 2)],
                    "status": "pass",
                    "reach_mode": "WALK_NORMAL",
                    "length_m": round(length_m, 1),
                    "remaining_distance_m": 0.0,
                    "points": pts,
                    "stuck_pos": None,
                    "barrier": None,
                }
            )
        else:
            dists_2d = np.hypot(main_xyz[:, 0] - tx, main_xyz[:, 1] - ty)
            # Prefer ground-level walkable surfaces over high cliff arches (> -30m) when XY distances are close
            dists_score = dists_2d + np.where(main_xyz[:, 2] > -30.0, 12.0, 0.0)
            best_idx = int(np.argmin(dists_score))
            nearest_main_node = main_nodes_list[best_idx]
            p_main = (
                float(main_xyz[best_idx, 0]),
                float(main_xyz[best_idx, 1]),
                float(main_xyz[best_idx, 2]),
            )
            pts = reconstruct_node_path(nearest_main_node)
            length_m = sum(
                math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)
            )
            rem_dist = float(dists_2d[best_idx])

            barrier_info = None
            if tgt_node is not None:
                tgt_island = next((isl for isl in isolated_islands if tgt_node in isl), [tgt_node])
                isl_xyz = np.array(
                    [[origin + n[0] * grid_step, origin + n[1] * grid_step, layers_map[(n[0], n[1])][n[2]]["z"]] for n in tgt_island],
                    dtype=np.float32,
                )
                step_m = max(1, len(main_xyz) // 1200)
                step_i = max(1, len(isl_xyz) // 1200)
                sub_main = main_xyz[::step_m]
                sub_isl = isl_xyz[::step_i]
                dx_m = sub_main[:, None, 0] - sub_isl[None, :, 0]
                dy_m = sub_main[:, None, 1] - sub_isl[None, :, 1]
                dz_m = (sub_main[:, None, 2] - sub_isl[None, :, 2]) * 3.5
                cost_mat = dx_m * dx_m + dy_m * dy_m + dz_m * dz_m
                flat_min = int(np.argmin(cost_mat))
                mi_sub, ii_sub = divmod(flat_min, len(sub_isl))
                approx_main = sub_main[mi_sub]
                approx_isl = sub_isl[ii_sub]

                rad_search = math.hypot(approx_main[0] - approx_isl[0], approx_main[1] - approx_isl[1]) + 45.0
                near_m_mask = (
                    (np.hypot(main_xyz[:, 0] - approx_isl[0], main_xyz[:, 1] - approx_isl[1]) <= rad_search)
                    | (np.hypot(main_xyz[:, 0] - p_main[0], main_xyz[:, 1] - p_main[1]) <= 65.0)
                )
                near_i_mask = (
                    (np.hypot(isl_xyz[:, 0] - approx_main[0], isl_xyz[:, 1] - approx_main[1]) <= rad_search)
                    | (np.hypot(isl_xyz[:, 0] - p_main[0], isl_xyz[:, 1] - p_main[1]) <= 65.0)
                )
                cand_m = main_xyz[near_m_mask] if np.any(near_m_mask) else sub_main
                cand_i = isl_xyz[near_i_mask] if np.any(near_i_mask) else sub_isl

                dx_f = cand_m[:, None, 0] - cand_i[None, :, 0]
                dy_f = cand_m[:, None, 1] - cand_i[None, :, 1]
                dz_f = (cand_m[:, None, 2] - cand_i[None, :, 2]) * 3.5
                f_min = int(np.argmin(dx_f * dx_f + dy_f * dy_f + dz_f * dz_f))
                fm_i, fi_i = divmod(f_min, len(cand_i))
                p_cut_from = (float(cand_m[fm_i, 0]), float(cand_m[fm_i, 1]), float(cand_m[fm_i, 2]))
                p_cut_to = (float(cand_i[fi_i, 0]), float(cand_i[fi_i, 1]), float(cand_i[fi_i, 2]))

                diag = diagnose_barrier_segment(p_cut_from, p_cut_to)
                barrier_info = {
                    "from_pos": [round(p_cut_from[0], 2), round(p_cut_from[1], 2), round(p_cut_from[2], 2)],
                    "to_pos": [round(p_cut_to[0], 2), round(p_cut_to[1], 2), round(p_cut_to[2], 2)],
                    **diag,
                }
                t_lyr = layers_map[(tgt_node[0], tgt_node[1])][tgt_node[2]]
                barrier_glitches.append(
                    {
                        "type": "WALKABILITY_BARRIER",
                        "subtype": diag["cause"],
                        "severity": "high",
                        "sector": "M1",
                        "model_id": t_lyr["id"],
                        "model_name": f"Route -> {t_name}",
                        "pos": diag["first_break_pos"],
                        "from_pos": barrier_info["from_pos"],
                        "to_pos": barrier_info["to_pos"],
                        "gap_dist_m": diag["gap_dist_m"],
                        "title": f"Đứt đường đi tới {t_name} (Đoạn đứt {diag['gap_dist_m']}m)",
                        "detail": (
                            f"Tuyến đi bộ tới '{t_name}' bị chặn tại E={diag['first_break_pos'][0]:.1f}, "
                            f"N={diag['first_break_pos'][1]:.1f}, U={diag['first_break_pos'][2]:.2f}m. "
                            f"Đoạn nối gần nhất cùng cốt cao độ từ mạng lưới chính ({p_cut_from[0]:.1f}, {p_cut_from[1]:.1f}, {p_cut_from[2]:.2f}m) "
                            f"sang vùng đích ({p_cut_to[0]:.1f}, {p_cut_to[1]:.1f}, {p_cut_to[2]:.2f}m) dài {diag['gap_dist_m']:.1f}m (dZ={diag['dz_m']:+.2f}m). "
                            f"Nguyên nhân gốc: {diag['cause_vi']}."
                        ),
                        "fix_hint": (
                            f"Nối dải đường/cầu hoặc nâng cốt đất từ ({p_cut_from[0]:.1f}, {p_cut_from[1]:.1f}) "
                            f"tới ({p_cut_to[0]:.1f}, {p_cut_to[1]:.1f}) (dài {diag['gap_dist_m']:.1f}m, dZ={diag['dz_m']:+.2f}m)."
                        ),
                    }
                )

            reach_mode = "JUMP_REACHABLE" if (tgt_node and tgt_node in jump_reachable) else "BLOCKED"
            tz_val = layers_map[(tgt_node[0], tgt_node[1])][tgt_node[2]]["z"] if tgt_node else -50.0
            bot_routes.append(
                {
                    "name": t_name,
                    "target": [round(tx, 2), round(ty, 2), round(tz_val, 2)],
                    "status": "pass" if reach_mode == "JUMP_REACHABLE" else "failed",
                    "reach_mode": reach_mode,
                    "length_m": round(length_m, 1),
                    "remaining_distance_m": round(rem_dist, 2),
                    "points": pts,
                    "stuck_pos": [round(p_main[0], 2), round(p_main[1], 2), round(p_main[2], 2)],
                    "barrier": barrier_info,
                    "reason": barrier_info["cause_vi"] if barrier_info else "Mốc nằm ở vùng ngập nước hoặc không có mặt đi bộ",
                }
            )

    # -------------------------------------------------------------------------
    # Multi-Seed Swarm Explorer (10 Seeds x 8 Radial Fan-Out + 6 Target Routes = 86 Bots)
    # Records full step-by-step telemetry frames [x, y, z, slope_deg, dz, state_code, note_vi]
    # including post-impact physics (freefall into void, water sink, or wall recoil) for DVR Replay!
    # -------------------------------------------------------------------------
    seed_specs = [
        ("SEED-01", "Cầu Cổng Thành Mondstadt (M0)", 321.0, -93.1, -55.0),
        ("SEED-02", "Quảng Trường Phong Thần (M0)", -35.0, 28.0, -34.0),
        ("SEED-03", "Đại Giáo Đường Mondstadt (M0)", -80.0, 68.0, -24.0),
        ("SEED-04", "Tường Thành Đông & Cổng Phụ (M0)", 115.0, -35.0, -49.0),
        ("SEED-05", "Ngã Ba Đường Mòn Rừng Thì Thầm (M1)", 450.0, -15.0, -28.0),
        ("SEED-06", "Bờ Bắc Đứt Gãy Hồ Cider (M1 · N=-205m)", 245.0, -205.0, -57.5),
        ("SEED-07", "Đảo Đông Bắc Rừng Thì Thầm (M1 · Island #2)", 525.0, 150.0, -56.0),
        ("SEED-08", "Đảo Nam Hồ Cider (M1 · Island #1)", 220.0, -500.0, -49.0),
        ("SEED-09", "Thung Lũng Thanh Tuyền (M1 · Island #1)", 380.0, -610.0, -10.0),
        ("SEED-10", "Cao Nguyên Vách Bắc Cider (M1 · Island #3)", 120.0, 535.0, -18.0),
    ]
    swarm_seeds: list[dict[str, Any]] = []
    for sid, sname, sx, sy, sz_pref in seed_specs:
        snode = find_nearest_node(sx, sy, prefer_z=sz_pref, max_r_cells=18)
        if snode is not None:
            rx = round(origin + snode[0] * grid_step, 2)
            ry = round(origin + snode[1] * grid_step, 2)
            rz = round(layers_map[(snode[0], snode[1])][snode[2]]["z"], 2)
        else:
            rx, ry, rz = round(sx, 2), round(sy, 2), round(sz_pref, 2)
        swarm_seeds.append({"seed_id": sid, "name": sname, "pos": [rx, ry, rz]})

    def build_route_telemetry_frames(route_obj: dict[str, Any]) -> tuple[list[list[Any]], int, list[float]]:
        """Convert a bot_route into step-by-step telemetry frames, plus simulated barrier impact frames."""
        pts = route_obj["points"]
        frames: list[list[Any]] = []
        for idx_p, pt in enumerate(pts):
            px, py, pz = pt[0], pt[1], pt[2]
            dz = round(pz - pts[idx_p - 1][2], 2) if idx_p > 0 else 0.0
            hits = col_idx.raycast_vertical(px, py, z_top=pz + 1.5, z_bottom=pz - 1.5)
            slope_deg = 0.0
            if hits:
                _, h_idx = min(hits, key=lambda h: abs(h[0] - pz))
                nz_val = float(col_idx.normals[h_idx, 2])
                slope_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, nz_val)))), 1)
            st_code = 1 if (abs(dz) > 0.45 or slope_deg > 38.0) else 0
            note = "Nhảy vượt bậc / Leo dốc" if st_code == 1 else "Đi bộ ổn định"
            frames.append([round(px, 2), round(py, 2), round(pz, 2), slope_deg, dz, st_code, note])

        if route_obj["status"] == "pass" or not route_obj.get("barrier"):
            if frames:
                frames[-1][6] = "Đã tới đích an toàn (PASS)"
            return frames, -1, (pts[-1] if pts else [0.0, 0.0, 0.0])

        bar = route_obj["barrier"]
        p_from = bar["from_pos"]
        p_to = bar["to_pos"]
        cause = bar["cause"]
        # Walk from last point of pts to p_from if needed, then march into the barrier up to first_break_pos
        last_pt = frames[-1][:3] if frames else p_from
        fb = bar.get("first_break_pos") or p_to
        # Add 3 approach steps toward first_break_pos
        for s in (0.35, 0.70, 1.0):
            ax = round(last_pt[0] + (fb[0] - last_pt[0]) * s, 2)
            ay = round(last_pt[1] + (fb[1] - last_pt[1]) * s, 2)
            az = round(last_pt[2] + (fb[2] - last_pt[2]) * s, 2)
            dz = round(az - frames[-1][2], 2) if frames else 0.0
            if s < 1.0:
                frames.append([ax, ay, az, 12.0, dz, 0, "Tiến sát điểm đứt gãy..."])
            else:
                fail_idx = len(frames)
                if cause == "SUBMERGED_UNDER_WATER":
                    frames.append([ax, ay, round(bar["min_z_along_gap"], 2), 18.0, dz, 3, f"❌ NGẬP NƯỚC: {bar['cause_vi']}"])
                    # Add 3 sinking/struggle frames
                    for sink_i, sink_dz in enumerate((-0.25, -0.55, -0.85), start=1):
                        frames.append([
                            round(ax + (p_to[0] - p_from[0]) * 0.04 * sink_i, 2),
                            round(ay + (p_to[1] - p_from[1]) * 0.04 * sink_i, 2),
                            round(bar["min_z_along_gap"] + sink_dz, 2),
                            22.0,
                            sink_dz,
                            3,
                            f"Chìm dưới mực nước -58.0m (Bước chìm #{sink_i})",
                        ])
                elif cause == "MISSING_COL_VOID":
                    frames.append([ax, ay, az, 85.0, dz, 2, f"❌ HỤT CHÂN RƠI TỰ DO: {bar['cause_vi']}"])
                    # Add 4 freefall frames plummeting into the void chasm!
                    for fall_i, fall_dz in enumerate((-2.5, -6.5, -12.0, -19.0), start=1):
                        frames.append([
                            round(ax + (p_to[0] - p_from[0]) * 0.03 * fall_i, 2),
                            round(ay + (p_to[1] - p_from[1]) * 0.03 * fall_i, 2),
                            round(az + fall_dz, 2),
                            90.0,
                            fall_dz,
                            2,
                            f"💥 Rơi tự do xuống khe nứt BigWorld ({fall_dz:+.1f}m)",
                        ])
                else:
                    frames.append([ax, ay, az, 68.0, dz, 4, f"❌ KẸT VÁCH DỐC / BẬC CAO: {bar['cause_vi']}"])
                    # Add 2 recoil frames bouncing back from the wall
                    for rec_i in (1, 2):
                        frames.append([
                            round(ax - (fb[0] - last_pt[0]) * 0.15 * rec_i, 2),
                            round(ay - (fb[1] - last_pt[1]) * 0.15 * rec_i, 2),
                            round(last_pt[2], 2),
                            68.0,
                            0.0,
                            4,
                            f"Bị đẩy bật lùi khỏi vách chắn (#{rec_i})",
                        ])
                return frames, fail_idx, [ax, ay, az]
        return frames, len(frames) - 1, fb

    swarm_journeys: list[dict[str, Any]] = []
    bot_counter = 1

    # 1) Add the 6 targeted landmark routes with full telemetry frames
    for r_obj in bot_routes:
        frames, fail_idx, fail_pos = build_route_telemetry_frames(r_obj)
        r_obj["frames"] = frames
        r_obj["fail_step_idx"] = fail_idx
        b_id = f"SWARM-{bot_counter:03d}"
        bot_counter += 1
        r_obj["bot_id"] = b_id
        outcome = "PATROL_OK" if r_obj["status"] == "pass" else (r_obj["barrier"]["cause"] if r_obj.get("barrier") else "BLOCKED")
        swarm_journeys.append(
            {
                "bot_id": b_id,
                "seed_id": "SEED-01",
                "seed_name": "Cầu Cổng Thành Mondstadt (M0)",
                "name": f"Tuyến Đích: {r_obj['name']}",
                "kind": "TARGET_ROUTE",
                "status": r_obj["status"],
                "outcome": outcome,
                "outcome_vi": "Hoàn thành lộ trình tới đích" if r_obj["status"] == "pass" else r_obj.get("reason", "Bị chặn"),
                "distance_m": r_obj["length_m"],
                "fail_step_idx": fail_idx,
                "fail_pos": fail_pos,
                "frames": frames,
            }
        )

    # 2) Launch 80 Multi-Seed Radial Swarm Explorers (10 Seeds x 8 Compass Directions)
    compass_dirs = [
        ("Đông (0°)", 0.0),
        ("Đông Bắc (45°)", math.pi * 0.25),
        ("Bắc (90°)", math.pi * 0.50),
        ("Tây Bắc (135°)", math.pi * 0.75),
        ("Tây (180°)", math.pi),
        ("Tây Nam (225°)", math.pi * 1.25),
        ("Nam (270°)", math.pi * 1.50),
        ("Đông Nam (315°)", math.pi * 1.75),
    ]
    step_len_m = 3.2
    max_radial_steps = 65

    for seed in swarm_seeds:
        sid = seed["seed_id"]
        sname = seed["name"]
        sx, sy, sz = seed["pos"]
        for dir_label, base_angle in compass_dirs:
            b_id = f"SWARM-{bot_counter:03d}"
            bot_counter += 1
            cx, cy, cz = float(sx), float(sy), float(sz)
            heading = base_angle
            frames = [[round(cx, 2), round(cy, 2), round(cz, 2), 0.0, 0.0, 0, f"Xuất phát từ {sid} ({sname})"]]
            fail_idx = -1
            fail_pos = [round(cx, 2), round(cy, 2), round(cz, 2)]
            outcome = "PATROL_OK"
            outcome_vi = f"Tuần tra thông suốt {max_radial_steps} bước theo hướng {dir_label}"
            bot_status = "pass"  # 'pass' | 'intended' | 'failed'
            dist_walked = 0.0

            for step_i in range(1, max_radial_steps + 1):
                # Try straight ahead first, then +-25, +-50, +-70 deg feelers so bots steer around normal rocks/curves like a real player
                chosen = None
                first_failure = None
                for steer_deg, steer_rad in ((0, 0.0), (25, 0.436), (-25, -0.436), (50, 0.873), (-50, -0.873), (70, 1.222), (-70, -1.222)):
                    ang = heading + steer_rad
                    nx = cx + math.cos(ang) * step_len_m
                    ny = cy + math.sin(ang) * step_len_m
                    # 1) Check if bot reached the outer 1536m x 1536m sector boundary (NOT a bug!)
                    if abs(nx) >= 752.0 or abs(ny) >= 752.0:
                        if first_failure is None:
                            first_failure = ("MAP_BOUNDARY_OK", nx, ny, cz, 0.0, 0.0, "Đã đi tới ranh giới ngoài của Sector M1 (±768m) an toàn", "pass")
                        continue

                    hits = col_idx.raycast_vertical(nx, ny, z_top=cz + 8.0, z_bottom=cz - 25.0)
                    if not hits:
                        if first_failure is None:
                            first_failure = ("MISSING_COL_VOID", nx, ny, cz, 90.0, 0.0, "Khe rỗng thiếu lưới va chạm .COL3 nội bộ (Rơi tự do xuống -∞)", "failed")
                        continue
                    hz, h_idx = min(hits, key=lambda h: abs(h[0] - cz))
                    hnz = float(col_idx.normals[h_idx, 2])
                    slope_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, abs(hnz))))), 1)
                    dz = round(hz - cz, 2)

                    # 2) Check inverted normal (Always a TRUE BUG)
                    if hnz < -0.35:
                        if first_failure is None:
                            first_failure = ("INVERTED_COL_NORMAL", nx, ny, hz, slope_deg, dz, f"Mặt .COL3 bị lật ngược pháp tuyến (nz={hnz:.2f}) gây lọt sàn", "failed")
                        continue

                    # 3) Check water submersion: distinguish Intended Lake Shoreline (with valid .COL3 lakebed) vs. Unintended Puddle / Void Shore!
                    if hz < water_level + 0.05:
                        if first_failure is None:
                            dip_cm = round((water_level + 0.05 - hz) * 100.0, 1)
                            # Probe 8m further ahead to see if there is a continuous .COL3 lakebed or if it's a void cutout / shallow road dip
                            ax2 = nx + math.cos(ang) * 8.0
                            ay2 = ny + math.sin(ang) * 8.0
                            ahead_hits = col_idx.raycast_vertical(ax2, ay2, z_top=water_level + 5.0, z_bottom=water_level - 35.0)
                            if not ahead_hits:
                                first_failure = (
                                    "MISSING_COL_VOID",
                                    ax2,
                                    ay2,
                                    hz,
                                    90.0,
                                    dz,
                                    f"Bờ hồ bị cắt cụt lơ lửng không có đáy hồ .COL3 tại (E={ax2:.1f}, N={ay2:.1f})",
                                    "failed",
                                )
                            else:
                                hz2 = min(ahead_hits, key=lambda h: abs(h[0] - hz))[0]
                                if hz2 < water_level - 0.35:
                                    first_failure = (
                                        "INTENDED_LAKE_SHORE",
                                        nx,
                                        ny,
                                        hz,
                                        slope_deg,
                                        dz,
                                        f"Bờ hồ tự nhiên: Đi bộ xuống lòng Hồ Cider có đáy .COL3 hợp lệ (Z={hz2:.2f}m)",
                                        "intended",
                                    )
                                else:
                                    first_failure = (
                                        "SUBMERGED_UNDER_WATER",
                                        nx,
                                        ny,
                                        hz,
                                        slope_deg,
                                        dz,
                                        f"Vũng trũng ngập đường/bờ ({dip_cm}cm dưới mực nước -58.0m, Z={hz:.2f}m)",
                                        "failed",
                                    )
                        continue

                    # 4) Check cliff drop: distinguish high scenic cliff overlook (Intended) vs. sudden seam drop on low ground
                    if dz < -(step_len_m * 1.15 + PHYS_JUMP_DROP):
                        if first_failure is None:
                            if cz > -32.0 or col_idx.sectors[h_idx] == "M0":
                                first_failure = (
                                    "INTENDED_CLIFF_OVERLOOK",
                                    nx,
                                    ny,
                                    hz,
                                    slope_deg,
                                    dz,
                                    f"Mép vách núi / tường thành cao tự nhiên (U={cz:.1f}m, dZ={dz:+.1f}m — dừng an toàn)",
                                    "intended",
                                )
                            else:
                                first_failure = (
                                    "VOID_CLIFF_DROP",
                                    nx,
                                    ny,
                                    hz,
                                    slope_deg,
                                    dz,
                                    f"Bậc sụt địa hình đột ngột ở vùng thấp (dZ={dz:+.2f}m)",
                                    "failed",
                                )
                        continue

                    # 5) Check high step lip
                    avg_nz = max(0.35, hnz)
                    tan_s = math.sqrt(max(0.0, 1.0 - avg_nz * avg_nz)) / avg_nz
                    max_up_walk = step_len_m * min(1.05, tan_s + 0.12) + PHYS_MAX_STEP
                    max_up_jump = step_len_m * min(1.30, tan_s + 0.15) + PHYS_JUMP_STEP
                    if dz > max_up_jump:
                        if first_failure is None:
                            if col_idx.sectors[h_idx] == "M0":
                                first_failure = (
                                    "INTENDED_CITY_WALL",
                                    nx,
                                    ny,
                                    hz,
                                    slope_deg,
                                    dz,
                                    f"Tường công trình / vách đá nội đô M0 (dZ={dz:+.2f}m)",
                                    "intended",
                                )
                            else:
                                first_failure = (
                                    "STEP_LIP_BLOCKER",
                                    nx,
                                    ny,
                                    hz,
                                    slope_deg,
                                    dz,
                                    f"Tường taluy cắt-đắp / Gờ bậc cao đột ngột (dZ={dz:+.2f}m > ngưỡng nhảy)",
                                    "failed",
                                )
                        continue

                    # 6) Check steep slope wall
                    if hnz < PHYS_JUMP_SLOPE_COS:
                        if first_failure is None:
                            if cz > -28.0 or col_idx.sectors[h_idx] == "M0":
                                first_failure = (
                                    "INTENDED_MOUNTAIN_SLOPE",
                                    nx,
                                    ny,
                                    hz,
                                    slope_deg,
                                    dz,
                                    f"Vách núi cao tự nhiên {slope_deg}° (giữ nguyên địa hình đồi núi)",
                                    "intended",
                                )
                            else:
                                first_failure = (
                                    "STEEP_SLOPE_BARRIER",
                                    nx,
                                    ny,
                                    hz,
                                    slope_deg,
                                    dz,
                                    f"Vách taluy cắt-đắp dốc gắt {slope_deg}° chặn ngang (vượt ngưỡng leo 54.5°)",
                                    "failed",
                                )
                        continue

                    # Valid step! Prefer low steering angle unless obstacle forces a turn
                    needs_jump = (dz > max_up_walk) or (hnz < PHYS_SLOPE_COS)
                    chosen = (nx, ny, hz, slope_deg, dz, ang, needs_jump)
                    break

                if chosen is not None:
                    nx, ny, hz, slope_deg, dz, ang, needs_jump = chosen
                    cx, cy, cz = nx, ny, hz
                    heading = 0.75 * heading + 0.25 * ang
                    dist_walked += step_len_m
                    st_code = 1 if needs_jump else 0
                    note = f"Nhảy bậc / Leo dốc ({slope_deg}°, dZ={dz:+.2f}m)" if needs_jump else f"Đi bộ êm ({slope_deg}°)"
                    frames.append([round(cx, 2), round(cy, 2), round(cz, 2), slope_deg, dz, st_code, note])
                else:
                    f_type, fx, fy, fz, f_slope, f_dz, f_msg, f_stat = first_failure or (
                        "STEEP_SLOPE_BARRIER",
                        cx + math.cos(heading) * step_len_m,
                        cy + math.sin(heading) * step_len_m,
                        cz,
                        75.0,
                        0.0,
                        "Bị chặn bởi địa hình",
                        "failed",
                    )
                    outcome = f_type
                    outcome_vi = f_msg
                    bot_status = f_stat
                    fail_idx = len(frames) if f_stat != "pass" else -1
                    fail_pos = [round(fx, 2), round(fy, 2), round(fz, 2)]

                    if f_stat == "pass":
                        frames.append([round(fx, 2), round(fy, 2), round(fz, 2), f_slope, f_dz, 0, f"✅ {f_msg}"])
                    elif f_stat == "intended":
                        frames.append([round(fx, 2), round(fy, 2), round(fz, 2), f_slope, f_dz, 1, f"ℹ️ [ĐỊA HÌNH TỰ NHIÊN] {f_msg}"])
                    elif f_type in ("MISSING_COL_VOID", "VOID_CLIFF_DROP", "INVERTED_COL_NORMAL"):
                        st_c = 6 if f_type == "INVERTED_COL_NORMAL" else 2
                        frames.append([round(fx, 2), round(fy, 2), round(fz, 2), f_slope, f_dz, st_c, f"❌ {f_msg}"])
                        for fall_i, fall_dz in enumerate((-3.0, -7.5, -13.5), start=1):
                            frames.append([
                                round(fx + math.cos(heading) * 0.4 * fall_i, 2),
                                round(fy + math.sin(heading) * 0.4 * fall_i, 2),
                                round(fz + fall_dz, 2),
                                90.0,
                                fall_dz,
                                st_c,
                                f"💥 Rơi tự do sau khi hụt chân ({fall_dz:+.1f}m)",
                            ])
                    elif f_type == "SUBMERGED_UNDER_WATER":
                        frames.append([round(fx, 2), round(fy, 2), round(fz, 2), f_slope, f_dz, 3, f"❌ {f_msg}"])
                        for sink_i, sink_dz in enumerate((-0.3, -0.7), start=1):
                            frames.append([
                                round(fx + math.cos(heading) * 0.3 * sink_i, 2),
                                round(fy + math.sin(heading) * 0.3 * sink_i, 2),
                                round(fz + sink_dz, 2),
                                f_slope,
                                sink_dz,
                                3,
                                f"Chìm xuống dưới mặt nước (#{sink_i})",
                            ])
                    else:
                        st_c = 5 if f_type == "STEP_LIP_BLOCKER" else 4
                        frames.append([round(fx, 2), round(fy, 2), round(fz, 2), f_slope, f_dz, st_c, f"❌ {f_msg}"])
                        for rec_i in (1, 2):
                            frames.append([
                                round(fx - math.cos(heading) * 0.6 * rec_i, 2),
                                round(fy - math.sin(heading) * 0.6 * rec_i, 2),
                                round(cz, 2),
                                f_slope,
                                0.0,
                                st_c,
                                f"Khựng lại / Bật lùi trước vật cản (#{rec_i})",
                            ])
                    break

            swarm_journeys.append(
                {
                    "bot_id": b_id,
                    "seed_id": sid,
                    "seed_name": sname,
                    "name": f"{sid} · Hướng {dir_label}",
                    "kind": "RADIAL_SWARM",
                    "status": bot_status,
                    "outcome": outcome,
                    "outcome_vi": outcome_vi,
                    "distance_m": round(dist_walked, 1),
                    "fail_step_idx": fail_idx,
                    "fail_pos": fail_pos,
                    "frames": frames,
                }
            )

    island_summaries: list[dict[str, Any]] = []
    for idx_isl, isl in enumerate(isolated_islands[:6]):
        isl_pts = np.array(
            [[origin + n[0] * grid_step, origin + n[1] * grid_step, layers_map[(n[0], n[1])][n[2]]["z"]] for n in isl],
            dtype=np.float32,
        )
        area_m2 = round(len(isl) * (grid_step * grid_step), 1)
        center = np.median(isl_pts, axis=0)
        c_idx = int(np.argmin(np.hypot(isl_pts[:, 0] - center[0], isl_pts[:, 1] - center[1])))
        rep_pos = [round(float(isl_pts[c_idx, 0]), 2), round(float(isl_pts[c_idx, 1]), 2), round(float(isl_pts[c_idx, 2]), 2)]
        island_summaries.append(
            {
                "island_id": idx_isl + 1,
                "cell_count": len(isl),
                "area_m2": area_m2,
                "pos": rep_pos,
                "min_xy": [round(float(isl_pts[:, 0].min()), 1), round(float(isl_pts[:, 1].min()), 1)],
                "max_xy": [round(float(isl_pts[:, 0].max()), 1), round(float(isl_pts[:, 1].max()), 1)],
            }
        )

    heatmap_reachable: list[list[float]] = []
    heatmap_isolated: list[list[float]] = []
    for (gx, gy, k) in walk_reachable:
        if gx % 3 == 0 and gy % 3 == 0:
            heatmap_reachable.append(
                [round(origin + gx * grid_step, 1), round(origin + gy * grid_step, 1), round(layers_map[(gx, gy)][k]["z"], 2)]
            )
    for isl in isolated_islands[:8]:
        for (gx, gy, k) in isl:
            if gx % 3 == 0 and gy % 3 == 0:
                heatmap_isolated.append(
                    [round(origin + gx * grid_step, 1), round(origin + gy * grid_step, 1), round(layers_map[(gx, gy)][k]["z"], 2)]
                )

    return {
        "grid_step_m": grid_step,
        "total_walkable_cells": len(all_walkable_nodes),
        "walk_reachable_cells": len(walk_reachable),
        "jump_reachable_cells": len(jump_reachable),
        "walk_coverage_pct": round(100.0 * len(walk_reachable) / max(1, len(all_walkable_nodes)), 2),
        "jump_coverage_pct": round(100.0 * len(jump_reachable) / max(1, len(all_walkable_nodes)), 2),
        "isolated_islands_count": len(isolated_islands),
        "top_isolated_islands": island_summaries,
        "bot_routes": bot_routes,
        "swarm_seeds": swarm_seeds,
        "swarm_journeys": swarm_journeys,
        "barrier_glitches": barrier_glitches,
        "headroom_glitches": dedup_headroom,
        "step_lip_glitches": dedup_step_lips,
        "heatmap_reachable": heatmap_reachable,
        "heatmap_isolated": heatmap_isolated,
    }


def _inline_three(root: Path = THREE_ROOT) -> tuple[str, str]:
    source = (root / "build" / "three.module.min.js").read_text(encoding="utf-8")
    match = re.search(r"export\s*\{([^}]+)\}\s*;?\s*$", source)
    if not match:
        raise ValueError("Unsupported Three.js bundle; expected one terminal ESM export list")
    pairs = []
    for item in match.group(1).split(","):
        bits = re.split(r"\s+as\s+", item.strip())
        local, public = (bits[0], bits[-1])
        pairs.append(f"{public}:{local}")
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    bundled = "/*\n" + license_text + "\n*/\n(function(){\n" + source[: match.start()] + "\nwindow.THREE={" + ",".join(pairs) + "};\n})();"
    return bundled.replace("</script", "<\\/script"), hashlib.sha256(source.encode()).hexdigest()


def build_glitch_lab_html(scene_data: dict[str, Any], report: dict[str, Any], out_html: Path) -> dict[str, Any]:
    template_path = Path(__file__).with_name("glitch_lab_viewer_template.html")
    template = template_path.read_text(encoding="utf-8")
    three_js, three_hash = _inline_three()

    scene_json = json.dumps(scene_data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    scene_b64 = base64.b64encode(gzip.compress(scene_json.encode("utf-8"), compresslevel=6)).decode("ascii")

    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    report_b64 = base64.b64encode(gzip.compress(report_json.encode("utf-8"), compresslevel=6)).decode("ascii")

    html = (
        template.replace("/*__THREE_BUNDLE__*/", three_js)
        .replace("/*__SCENE_DATA__*/{}", scene_b64)
        .replace("/*__GLITCH_REPORT_DATA__*/{}", report_b64)
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return {
        "output_html": str(out_html),
        "bytes": out_html.stat().st_size,
        "sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "three_sha256": three_hash,
    }


def run_full_glitch_audit(scene_path: Path, out_dir: Path, out_html: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    print(f"[1/6] Loading 3D scene from {scene_path} ...")
    scene_data = load_scene_from_html_or_json(scene_path)
    water_level = float(scene_data.get("audit", {}).get("waterLevel", -58.0))

    print("[2/6] Building Spatial 2D/3D Collision & Visual Triangle Indices ...")
    col_idx = build_collision_index(scene_data)
    vis_idx, near_models = build_visual_index(scene_data)
    print(
        f"      Indexed {len(col_idx.v0):,} COL3 triangles ({len(scene_data.get('collision', []))} COL entries) "
        f"& {len(vis_idx.v0):,} Near DFF triangles ({len(near_models)} Near models)."
    )

    print("[3/6] Running Scanner A & E: Multi-Layer 2.5D Swarm Capsule Walker, Chokepoints, Step-Lips & Headroom Traps ...")
    swarm_res = run_swarm_walkability_and_chokepoint_scanner(
        scene_data, col_idx, water_level=water_level, grid_step=2.5
    )

    print("[4/6] Running Scanner B: Multi-Point Footprint Raycast for Exposed Floating Models & Slope Overhangs ...")
    floating_glitches = scan_floating_and_overhang_models(
        scene_data, col_idx, near_models, water_level=water_level
    )

    print("[5/6] Running Scanner C: Exposed Visual (.DFF) vs Physics (.COL3) Mismatch ...")
    mismatch_glitches, buried_inverted_count = scan_visual_vs_collision_mismatches(
        vis_idx, col_idx, water_level=water_level
    )

    print("[6/6] Running Scanner D: Exposed Surface Void Freefall Traps & Seam Chasms ...")
    void_glitches = scan_void_traps_and_seams(col_idx, water_level=water_level)

    all_glitches: list[dict[str, Any]] = []
    for g in (
        swarm_res["barrier_glitches"]
        + void_glitches
        + floating_glitches
        + mismatch_glitches
        + swarm_res["headroom_glitches"]
        + swarm_res["step_lip_glitches"]
    ):
        all_glitches.append(g)

    sev_rank = {"high": 0, "medium": 1, "low": 2}
    all_glitches.sort(key=lambda item: (sev_rank.get(item.get("severity", "low"), 3), item.get("type", "")))
    for i, g in enumerate(all_glitches, start=1):
        g["glitch_id"] = f"GLITCH-{i:03d}"

    elapsed_s = round(time.perf_counter() - t0, 2)

    by_type = collections.Counter(g["type"] for g in all_glitches)
    by_subtype = collections.Counter(g["subtype"] for g in all_glitches)
    by_severity = collections.Counter(g["severity"] for g in all_glitches)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_file": str(scene_path),
        "elapsed_seconds": elapsed_s,
        "water_level_m": water_level,
        "geometry_stats": {
            "near_models": len(near_models),
            "col_models": len(scene_data.get("collision", [])),
            "col_triangles": int(len(col_idx.v0)),
            "vis_triangles": int(len(vis_idx.v0)),
            "foliage_instances": len(scene_data.get("foliage", [])),
            "buried_internal_hlod_inverted_faces": buried_inverted_count,
        },
        "walkability_summary": {
            "grid_step_m": swarm_res["grid_step_m"],
            "total_walkable_cells": swarm_res["total_walkable_cells"],
            "walk_reachable_cells": swarm_res["walk_reachable_cells"],
            "jump_reachable_cells": swarm_res["jump_reachable_cells"],
            "walk_coverage_pct": swarm_res["walk_coverage_pct"],
            "jump_coverage_pct": swarm_res["jump_coverage_pct"],
            "isolated_islands_count": swarm_res["isolated_islands_count"],
            "top_isolated_islands": swarm_res["top_isolated_islands"],
            "bot_routes": swarm_res["bot_routes"],
            "swarm_seeds": swarm_res["swarm_seeds"],
            "swarm_journeys": swarm_res["swarm_journeys"],
        },
        "glitch_counts": {
            "total": len(all_glitches),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "by_subtype": dict(by_subtype),
        },
        "glitches": all_glitches,
        "heatmap": {
            "reachable": swarm_res["heatmap_reachable"],
            "isolated": swarm_res["heatmap_isolated"],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "glitch_audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit completed in {elapsed_s}s! Saved JSON report to {report_path}")

    if out_html is not None:
        print(f"Building Interactive 3D Glitch Lab Studio HTML -> {out_html} ...")
        html_meta = build_glitch_lab_html(scene_data, report, out_html)
        (out_dir / "viewer_build_manifest.json").write_text(
            json.dumps(html_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Built HTML ({html_meta['bytes']:,} bytes, SHA256={html_meta['sha256'][:16]}...)")

    return scene_data, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eagle 3D Map Walker & Glitch Lab")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"c:\Users\LENOVO\Desktop\GTA\docs\project-eagle\mondstadt-m0-m1-3d-inspection-report-2026-10-23.html"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"c:\Users\LENOVO\Desktop\GTA\staging\project-eagle\map-walker-glitch-lab-2026-10-24"),
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path(r"c:\Users\LENOVO\Desktop\GTA\docs\project-eagle\eagle-3d-map-walker-glitch-lab-2026-10-24.html"),
    )
    args = parser.parse_args()
    _, rep = run_full_glitch_audit(args.source, args.out_dir, args.out_html)
    print("Summary:", json.dumps(rep["glitch_counts"], ensure_ascii=False))
    for r in rep["walkability_summary"]["bot_routes"]:
        print("Route:", r["name"], "|", r["status"], "| len:", r["length_m"], "| rem:", r["remaining_distance_m"], "| barrier:", r.get("barrier"))

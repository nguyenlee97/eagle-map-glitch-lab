# 🗺️ Eagle Map Glitch Lab

**3D map walkability analyzer & geometric glitch debugger for GTA San Andreas.**

Automated tool that ingests GTA SA binary map packages (.IMG/.COL3/.DFF/.IPL) and runs 5 geometric/physics scanners to detect walkability issues, collision mismatches, void traps, and step blockers — with an interactive 3D HTML viewer output.

## Features

### 5 Automated Geometric & Physics Glitch Scanners

| Scanner | Type | Detection |
|---------|------|-----------|
| **A** | Multi-Layer 2.5D Swarm Capsule Walker | Unreachable islands & barrier chokepoints |
| **B** | Multi-Point Footprint Raycast | Floating models & slope overhangs |
| **C** | Visual vs Physics Mismatch | DFF geometry vs COL3 collision differences |
| **D** | Void Freefall Traps | BigWorld/tile seam chasms |
| **E** | Step-Lip Blockers | Real overhead headroom traps |

### Player Capsule Physics
- Radius: 0.32m
- Height: 1.72m
- Max step: 0.45m
- Max drop: 0.65m

### Output
- **JSON diagnostics** — machine-readable glitch data
- **Self-contained interactive 3D HTML viewer** — Three.js-based, opens in any browser
- Color-coded glitch overlay on the 3D map

### Performance
- `SpatialTriangleIndex` with 2D grid for fast raycasting
- Efficient numpy-based spatial operations

## Requirements

- **Python 3.8+**
- **numpy** — `pip install numpy`

## Usage

```bash
python eagle_map_glitch_lab.py <source_html_or_json>
```

The tool generates an interactive HTML report that you can open in any browser.

## Technical Details

- Ingests GTA SA binary packages: `.IMG`, `.COL3`, `.DFF`, `.IPL`
- Also accepts embedded 3D inspection bundles (JSON/HTML)
- Uses Three.js (from CDN or local) for the 3D viewer output
- Template-based HTML generation with `glitch_lab_viewer_template.html`

## License

MIT License — see [LICENSE](LICENSE).

## Credits

- Built as part of **Project Eagle** by [@nguyenlee97](https://github.com/nguyenlee97)
- [Three.js](https://threejs.org/) for 3D visualization
- [NumPy](https://numpy.org/) for spatial computations

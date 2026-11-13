# Mesh Align

Align a 3D scan to a datum coordinate frame before importing it into CAD (Onshape, etc.).
You pick reference features on the mesh: a flat face, a hole, an edge. The tool fits ideal
geometry to them by least squares with outlier rejection, then rotates and moves the scan
so those features land on the origin and axes you choose.

**To use it, open <https://imrisaac.github.io/mesh-align/>**, or download `mesh-align.html` and
open it in Chrome, Firefox or Edge. It is a single file that runs offline. Nothing gets
uploaded, because everything happens in your browser, even on the hosted page.

## Workflow

1. **Open scan** (STL, OBJ or PLY), or drag the file onto the view.
2. **Add features** and click on the scan to pick points for each one:

   | Feature  | Use for | How to pick |
   |----------|---------|-------------|
   | Plane    | flat face | One click floods the connected flat area (stops where the surface turns more than *Max angle*). `Shift`+drag erases points you don't want. Shown with flatness and RMS. |
   | Circle   | hole edge / rim | Click 3+ points around the edge. If the scan left the hole open, use **Open edge** to snap to the whole gap. Shown with Ø. |
   | Cylinder | hole or boss wall | Click patches on the curved wall, spread around it. Shown with Ø, length, and whether it's a hole or a boss. |
   | Line     | straight edge | Click 2+ points along it. |
   | Point    | any point | Click. Several clicks are averaged. |

   **Removing points.** Hold `Shift` and click or drag over points to erase them with a brush,
   or turn on **Erase mode** so plain clicks and drags erase. Set the brush size in **Eraser**
   or with `[` / `]`. This is how you trim a plane patch that spilled onto a fillet, a rounded
   edge, or a noisy area. Erased points stay out when you click to add more patches. To bring an
   area back, click inside it. `Ctrl`+`Z` undoes the last click, erase drag or Clear. `Esc`
   finishes picking.

   **Parallel planes.** Faces that should be parallel, like the top and bottom of a plate or
   stepped shelves, rarely come out exactly parallel from a scan. Set **Parallel to** on a
   plane's card to link it to another plane. Linked planes (directly or through each other)
   are fitted together with one shared normal, so they come out exactly parallel, and each
   keeps its own offset. Opposite-facing faces count. The card shows how far apart the planes
   are and how far each one was turned to match. A plane more than 10° off the rest is left
   out and flagged. The parallel fit is used everywhere: datum frame, readout, STEP and JSON.

3. **Datum frame.** This fills itself in as you add features. Adjust anything:
   - **Primary.** This feature's normal or axis becomes an axis. For a plane, the default is
     *normal → −Z*: the face is the bottom and the part sits on the XY (Top) plane. Press ⇅ if
     the picked face should look up instead.
   - **Clocking.** Sets the rotation about the primary axis. Either use the feature's own
     direction (a side face or an edge), or point the axis from the origin *toward* the
     feature (a second hole).
   - **Origin.** Up to three features in priority order. The origin is where they meet:
     plane + hole → the hole centre on that face, plane + cylinder → where the axis pierces
     the plane, three planes → the corner.
   - **Rotate / Offset.** Extra spin about the primary axis, and a final shift.

   The readout lists every feature's position in the new frame, so you can check things like
   hole spacing or squareness before exporting.
4. **Export.** Set **Scan units** to whatever your scanner uses (usually millimeters).
   - **Download aligned STL** is the scan itself. Import it into Onshape with those same units.
   - **Simplify the STL** makes that download smaller by merging triangles. Flat areas
     become a few large triangles and curved areas keep their detail. Set **Keep** (% of
     triangles) and **Tolerance** (roughly how far the surface may move). Simplification stops
     at whichever limit it reaches first, and Keep 0 means "as far as the tolerance allows".
     The view previews the result with its edges, and lists the measured deviation from the
     scan. Only the export changes: picking and fitting always use the full-resolution scan.
   - **Reference surfaces (STEP)** holds your fitted features as real CAD geometry. Planes
     become planar faces, circles become flat disks (giving you the circular edge and centre),
     and cylinders become cylindrical faces (giving you the axis). STL can only hold
     triangles, so this is how you get true planes to sketch on, mate to, or build datum
     planes from. Lines and points aren't included. Bring both files into one Part Studio:
     they share coordinates, so the surfaces sit right on the scan.
   - **Transform (JSON)** saves the 4×4 matrix and the feature positions.

## Tips

- Scans are noisy. Use more points: patches beat single clicks, and the readout's RMS tells
  you how well the fit matched.
- Small holes are often captured poorly inside. A **Circle** on the rim is usually more
  reliable than a **Cylinder** on the wall.
- **Max angle** is measured from the clicked point for flat faces. Switch it to *neighbours*
  to flood a smoothly curved surface, and limit it with **Radius**.

## Development

```
npm install
npm test        # unit tests for the fitting / alignment math (src/geom.js)
npm run build   # bundles src/ + three.js into mesh-align.html
```

Every push to `main` runs the tests, rebuilds `mesh-align.html` and publishes it to GitHub
Pages (`.github/workflows/pages.yml`). Commit the rebuilt `mesh-align.html` too, so the
downloadable copy matches the page.

`src/geom.js` holds all the math: fits, datum solver, mesh welding, region growing, mesh
simplification (quadric edge collapse) and STL writing. `src/step.js` writes the STEP reference surfaces. Neither depends on three.js.
`src/main.js` is the UI and viewer.

// Mesh Align UI: load a scan, pick reference features on it, build a datum
// frame, export the aligned mesh. All the math lives in geom.js.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
import * as G from './geom.js';
import { writeStepReference } from './step.js';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const COLORS = ['#f59e0b', '#22d3ee', '#c084fc', '#4ade80', '#f472b6', '#facc15', '#60a5fa', '#fb7185'];
const HOLE_TYPES = ['circle', 'cylinder', 'point'];
const TYPES = {
  plane: {
    label: 'Plane', pick: { mode: 'patch', angle: 15, ref: 'seed', radius: 0 },
    help: 'Click the flat face. Each click grows a patch across the flat area; add a few to cover it. Shift+drag over points you don\'t want (fillets, rounded edges, noise) to erase them.',
  },
  circle: {
    label: 'Circle', pick: { mode: 'point', angle: 15, ref: 'seed' },
    help: 'Click 3 or more points spread around the edge of the hole. If the scan left the hole open, try Open edge.',
  },
  cylinder: {
    label: 'Cylinder', pick: { mode: 'patch', angle: 25, ref: 'seed' },
    help: 'Click patches on the curved wall of the hole or boss, spread around it.',
  },
  line: {
    label: 'Line', pick: { mode: 'point', angle: 15, ref: 'seed' },
    help: 'Click 2 or more points along a straight edge.',
  },
  point: {
    label: 'Point', pick: { mode: 'point', angle: 15, ref: 'seed' },
    help: 'Click the point. Several clicks are averaged.',
  },
};
const TARGETS = ['+Z', '-Z', '+X', '-X', '+Y', '-Y'];
const prettyTarget = (t) => t.replace('-', '−');

const state = {
  mesh: null,
  data: null, // { pos, nrm, index, vt } for picking
  boundary: null, // open-edge graph, built on first use
  fileName: '',
  centroid: [0, 0, 0],
  diag: 100,
  features: [],
  nextId: 0,
  counters: {},
  activeId: null,
  pickByType: {},
  eraser: 1, // eraser brush radius, scan units
  eraseMode: false, // plain clicks/drags erase (Shift does the same while held)
  // Export simplification. result: { positions, index, colors, maxDev, rmsDev }
  simplify: { on: false, keep: 10, tol: 0.05, edges: true, result: null, preview: null },
  align: freshAlign(),
  preview: true,
  result: null,
};

function freshAlign() {
  return {
    primary: { id: null, target: '+Z' },
    secondary: { id: null, use: 'auto', target: '+X' },
    angle: 0,
    origin: [null, null, null],
    offset: [0, 0, 0],
  };
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

const canvas = $('#c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15181d);

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1e5);
camera.up.set(0, 0, 1);
camera.position.set(150, -220, 160);
scene.add(camera);
const headlight = new THREE.DirectionalLight(0xffffff, 2.0);
headlight.position.set(0.4, 0.8, 0);
headlight.target.position.set(0, 0, -1);
camera.add(headlight, headlight.target);
scene.add(new THREE.HemisphereLight(0xe6edf5, 0x2a2e35, 1.2));

const controls = new OrbitControls(camera, canvas);
controls.zoomToCursor = true;
controls.screenSpacePanning = true;
controls.addEventListener('change', requestRender);

// Everything in scan coordinates lives under alignGroup; its matrix is the
// alignment, so the world frame is the new datum frame.
const alignGroup = new THREE.Group();
alignGroup.matrixAutoUpdate = false;
scene.add(alignGroup);
const helpers = new THREE.Group();
scene.add(helpers);

const meshMaterial = new THREE.MeshStandardMaterial({
  color: 0xc5c9cf, roughness: 0.8, metalness: 0, side: THREE.DoubleSide,
  polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
});

const cursor = new THREE.Mesh(
  new THREE.RingGeometry(0.9, 1, 64),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthTest: false }),
);
cursor.renderOrder = 20;
cursor.visible = false;
scene.add(cursor);

let renderQueued = false;
function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderer.render(scene, camera);
  });
}

new ResizeObserver(() => {
  const { clientWidth: w, clientHeight: h } = $('#view');
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  requestRender();
}).observe($('#view'));

function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x));
  const m = x / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

function textSprite(text, color, size) {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d');
  ctx.font = 'bold 44px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.fillText(text, 32, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.setScalar(size);
  s.renderOrder = 15;
  return s;
}

// XY grid (Onshape's Top plane) and an XYZ triad at the origin.
function buildHelpers() {
  for (const child of [...helpers.children]) {
    helpers.remove(child);
    child.traverse((o) => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose(); });
  }
  const step = niceStep(state.diag / 10);
  const grid = new THREE.GridHelper(step * 24, 24, 0x3d4552, 0x272c34);
  grid.rotation.x = Math.PI / 2;
  helpers.add(grid);
  const L = step * 2;
  const axes = [[1, 0, 0, '#ef4444', 'X'], [0, 1, 0, '#22c55e', 'Y'], [0, 0, 1, '#3b82f6', 'Z']];
  for (const [x, y, z, color, name] of axes) {
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(x * L, y * L, z * L)]);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }));
    line.renderOrder = 15;
    helpers.add(line);
    const label = textSprite(name, color, L * 0.22);
    label.position.set(x * L * 1.12, y * L * 1.12, z * L * 1.12);
    helpers.add(label);
  }
}

const VIEWS = {
  iso: [1, -1, 0.8],
  top: [0, -1e-4, 1],
  front: [0, -1, 0],
  right: [1, 0, 0],
  bottom: [0, 1e-4, -1],
};

function fitView(dirName) {
  if (!state.mesh) return;
  const dir = dirName
    ? new THREE.Vector3(...VIEWS[dirName]).normalize()
    : camera.position.clone().sub(controls.target).normalize();
  const sphere = state.mesh.geometry.boundingSphere.clone().applyMatrix4(state.mesh.matrixWorld);
  const dist = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.05;
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(dir, dist);
  camera.near = dist / 200;
  camera.far = dist * 50;
  camera.updateProjectionMatrix();
  controls.update();
  requestRender();
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function openFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['stl', 'obj', 'ply'].includes(ext)) {
    toast(`Can't open .${ext} files. Use STL, OBJ or PLY.`, true);
    return;
  }
  busy(`Reading ${file.name}…`);
  try {
    await nextFrame();
    let geom;
    if (ext === 'stl') geom = new STLLoader().parse(await file.arrayBuffer());
    else if (ext === 'ply') {
      geom = new PLYLoader().parse(await file.arrayBuffer());
      if (!geom.index) throw new Error('this PLY has no faces (it is a point cloud). Mesh it first.');
    } else geom = objToGeometry(new OBJLoader().parse(await file.text()));
    busy('Preparing mesh…');
    await nextFrame();
    setMesh(geom, file.name);
    if (state.simplify.on) await runSimplify();
  } catch (err) {
    console.error(err);
    toast(`Couldn't load ${file.name}: ${err.message}`, true);
  } finally {
    busy(null);
  }
}

function objToGeometry(group) {
  const parts = [];
  group.updateMatrixWorld(true);
  group.traverse((o) => {
    if (!o.isMesh) return;
    const g = o.geometry.index ? o.geometry.toNonIndexed() : o.geometry;
    const pos = g.attributes.position.clone().applyMatrix4(o.matrixWorld);
    parts.push({ pos: pos.array, color: g.attributes.color?.array });
  });
  const total = parts.reduce((s, p) => s + p.pos.length, 0);
  if (!total) throw new Error('no triangles found');
  const pos = new Float32Array(total);
  const withColor = parts.every((p) => p.color);
  const color = withColor ? new Float32Array(total) : null;
  let o = 0;
  for (const p of parts) {
    pos.set(p.pos, o);
    if (color) color.set(p.color, o);
    o += p.pos.length;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  if (color) geom.setAttribute('color', new THREE.BufferAttribute(color, 3));
  return geom;
}

function attributeArray(attr) {
  if (!attr.isInterleavedBufferAttribute && attr.array instanceof Float32Array) return attr.array;
  const out = new Float32Array(attr.count * 3);
  for (let i = 0; i < attr.count; i++) {
    out[3 * i] = attr.getX(i); out[3 * i + 1] = attr.getY(i); out[3 * i + 2] = attr.getZ(i);
  }
  return out;
}

function setMesh(src, name) {
  const posAttr = src.attributes.position;
  if (!posAttr || posAttr.count < 3) throw new Error('the file has no geometry');
  const { positions, index, remap } = G.mergeVertices(attributeArray(posAttr), src.index ? src.index.array : null);
  const vc = positions.length / 3;
  let colors = null;
  if (src.attributes.color) {
    const c = attributeArray(src.attributes.color);
    colors = new Float32Array(vc * 3);
    for (let i = 0; i < remap.length; i++) colors.set(c.subarray(3 * i, 3 * i + 3), 3 * remap[i]);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setIndex(new THREE.BufferAttribute(index, 1));
  if (colors) geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geom.computeVertexNormals();
  geom.computeBoundingBox();
  geom.computeBoundingSphere();
  geom.computeBoundsTree(); // note: reorders the index buffer, so build topology after

  clearScene();
  meshMaterial.vertexColors = !!colors;
  meshMaterial.color.set(colors ? 0xffffff : 0xc5c9cf);
  meshMaterial.needsUpdate = true;
  const mesh = new THREE.Mesh(geom, meshMaterial);
  alignGroup.add(mesh);

  const idx = geom.index.array;
  state.mesh = mesh;
  state.data = { pos: positions, nrm: geom.attributes.normal.array, index: idx, vt: G.vertexTriangles(idx, vc) };
  state.boundary = null;
  state.fileName = name;
  const c = [0, 0, 0];
  for (let i = 0; i < positions.length; i += 3) { c[0] += positions[i]; c[1] += positions[i + 1]; c[2] += positions[i + 2]; }
  state.centroid = G.scale(c, 1 / vc);
  const size = geom.boundingBox.getSize(new THREE.Vector3());
  state.diag = size.length() || 1;
  const r = Number(niceStep(state.diag * 0.025).toPrecision(2));
  state.pickByType = Object.fromEntries(Object.entries(TYPES).map(([t, info]) => [t, { radius: r, ...info.pick }]));
  state.eraser = r;
  state.simplify.tol = Number(niceStep(state.diag * 3e-4).toPrecision(2));
  syncSimplifyControls();

  const tris = idx.length / 3;
  $('#file-info').innerHTML =
    `<b>${escapeHtml(name)}</b><br>${tris.toLocaleString()} triangles · ` +
    `${fmtLen(size.x)} × ${fmtLen(size.y)} × ${fmtLen(size.z)}` +
    (state.diag < 2 ? '<br><span class="warn">That is very small. Is this scan in meters? Pick matching units when importing.</span>' : '');
  $('#empty').hidden = true;
  $('#btn-stl').disabled = $('#btn-step').disabled = $('#btn-json').disabled = false;

  buildHelpers();
  renderFeatureList();
  refreshAlignControls();
  updateAlignment();
  fitView('iso');
}

function clearScene() {
  clearSimplified();
  for (const f of state.features) disposeObject(f.group);
  state.features = [];
  state.nextId = 0;
  state.counters = {};
  state.activeId = null;
  state.align = freshAlign();
  if (state.mesh) {
    alignGroup.remove(state.mesh);
    state.mesh.geometry.disposeBoundsTree();
    state.mesh.geometry.dispose();
    state.mesh = null;
  }
  $('#angle').value = 0;
  for (const el of $$('[data-off]')) el.value = 0;
  setActive(null);
}

function disposeObject(obj) {
  if (!obj) return;
  obj.parent?.remove(obj);
  obj.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
}

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

const featureById = (id) => state.features.find((f) => f.id === id) || null;
const activeFeature = () => featureById(state.activeId);

function addFeature(type) {
  if (!state.mesh) { toast('Open a scan first.'); return; }
  state.counters[type] = (state.counters[type] || 0) + 1;
  const id = ++state.nextId;
  const f = {
    id, type,
    name: `${TYPES[type].label} ${state.counters[type]}`,
    color: COLORS[(id - 1) % COLORS.length],
    strokes: [], fit: null, error: null, visible: true, group: null, autoAssigned: false,
    excluded: new Set(), // mesh vertices erased from this feature; patches won't re-add them
    history: [], // undo snapshots of { strokes, excluded }
    ownFit: null, // fit to this feature's points alone (fit may differ: parallel planes)
    parallelTo: null, // plane: id of a plane it is kept parallel to
  };
  state.features.push(f);
  setActive(id);
  refreshAlignControls();
}

function deleteFeature(id) {
  const f = featureById(id);
  if (!f) return;
  disposeObject(f.group);
  // Keep the rest of its parallel group together.
  const heirs = state.features.filter((x) => x.parallelTo === id);
  const to = f.parallelTo ?? heirs[0]?.id ?? null;
  for (const h of heirs) h.parallelTo = h.id === to ? null : to;
  state.features = state.features.filter((x) => x !== f);
  const a = state.align;
  if (a.primary.id === id) a.primary.id = null;
  if (a.secondary.id === id) a.secondary.id = null;
  a.origin = a.origin.map((o) => (o === id ? null : o));
  if (state.activeId === id) setActive(null);
  refreshAlignControls();
  if (f.type === 'plane') updateFits(state.features.filter((x) => x.type === 'plane'));
  else { renderFeatureList(); updateAlignment(); }
}

function setActive(id) {
  if (erase) finishErase();
  state.activeId = id;
  state.eraseMode = false;
  const f = activeFeature();
  $('#pick-tools').hidden = !f;
  canvas.style.cursor = f ? 'crosshair' : '';
  if (!f) cursor.visible = false;
  updateStatus();
  if (f) syncPickControls();
  renderFeatureList();
  requestRender();
}

function updateStatus() {
  const f = activeFeature();
  const status = $('#status');
  status.hidden = !f;
  if (!f) return;
  status.style.setProperty('--c', f.color);
  status.innerHTML = state.eraseMode
    ? `Erasing from <b>${escapeHtml(f.name)}</b> — click or drag over points to remove them. <span class="muted">Drag off the scan to orbit.</span>`
    : `Picking <b>${escapeHtml(f.name)}</b> — click on the scan, <kbd>Shift</kbd>+drag to erase. <span class="muted">Esc when done.</span>`;
}

// Undo history. Strokes are never changed in place (erasing builds new ones)
// and a feature's excluded set is replaced rather than edited once it is in a
// snapshot, so a snapshot only needs shallow copies.
const HISTORY_MAX = 50;
function pushHistory(f, snap) {
  f.history.push(snap);
  if (f.history.length > HISTORY_MAX) f.history.shift();
}
const snapshot = (f) => ({ strokes: f.strokes.slice(), excluded: f.excluded });

function undo(f) {
  const prev = f.history.pop();
  if (!prev) return false;
  f.strokes = prev.strokes;
  f.excluded = prev.excluded;
  refit(f);
  return true;
}

function pickSettings() {
  const f = activeFeature();
  return f ? state.pickByType[f.type] : null;
}

// A feature's points and surface normals, as fit input.
function featureArrays(f) {
  const n = f.strokes.reduce((s, st) => s + st.pts.length / 3, 0);
  const P = new Float64Array(n * 3), N = new Float64Array(n * 3);
  let o = 0;
  for (const st of f.strokes) { P.set(st.pts, o); N.set(st.nrms, o); o += st.pts.length; }
  return { P, N };
}

function refit(f) {
  const { P, N } = featureArrays(f);
  f.ownFit = null;
  f.error = null;
  if (P.length) {
    const r = G.fitFeature(f.type, P, N);
    if (r.error) f.error = r.error; else f.ownFit = r;
  }
  updateFits(groupOf(f));
}

// Planes linked through "Parallel to", directly or via each other, as groups.
// Every plane is in exactly one group (most are on their own).
function parallelGroups() {
  const planes = state.features.filter((f) => f.type === 'plane');
  const root = new Map(planes.map((f) => [f.id, f.id]));
  const find = (id) => (root.get(id) === id ? id : find(root.get(id)));
  for (const f of planes) if (root.has(f.parallelTo)) root.set(find(f.id), find(f.parallelTo));
  const groups = new Map();
  for (const f of planes) {
    const r = find(f.id);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(f);
  }
  return [...groups.values()];
}
const groupOf = (f) => (f.type === 'plane' && parallelGroups().find((g) => g.includes(f))) || [f];

// Set the fit used everywhere (alignment, visuals, export) for these features,
// given as whole parallel groups: each feature's own fit, except that planes
// kept parallel are fitted together with one shared normal.
function updateFits(features) {
  for (const f of features) f.fit = f.ownFit;
  for (const g of parallelGroups()) {
    const fitted = g.filter((f) => f.ownFit);
    if (fitted.length < 2 || !g.some((f) => features.includes(f))) continue;
    const res = G.fitParallelPlanes(fitted.map(featureArrays));
    fitted.forEach((f, i) => { if (!res[i].error) f.fit = res[i]; });
  }
  for (const f of features) {
    if (f.fit && !f.autoAssigned) {
      f.autoAssigned = true;
      autoAssign(f);
      refreshAlignControls();
    }
    rebuildFeatureVisual(f);
  }
  renderFeatureList();
  updateAlignment();
}

// Fill empty datum slots as features get their first fit: the first plane
// (or axis) becomes the primary and the origin, the first hole goes into the
// origin, and the next hole/plane sets the clocking.
function autoAssign(f) {
  const a = state.align;
  if (a.primary.id == null && f.fit.dir) {
    a.primary.id = f.id;
    a.primary.target = f.type === 'plane' ? '-Z' : '+Z';
    if (a.origin[0] == null) a.origin[0] = f.id;
    return;
  }
  const fillOrigin = () => {
    const k = a.origin.indexOf(null);
    if (k >= 0 && !a.origin.includes(f.id)) a.origin[k] = f.id;
  };
  const primary = featureById(a.primary.id);
  if (HOLE_TYPES.includes(f.type)) {
    const originHasHole = a.origin.some((id) => HOLE_TYPES.includes(featureById(id)?.type));
    if (!originHasHole) fillOrigin();
    else if (a.secondary.id == null) a.secondary.id = f.id;
    return;
  }
  // A second plane or a line: clocking if it isn't parallel to the primary.
  const parallel = primary?.fit?.dir && Math.abs(G.dot(primary.fit.dir, f.fit.dir)) > 0.866;
  if (a.secondary.id == null && !parallel) a.secondary.id = f.id;
  fillOrigin();
}

function featurePoints(f) {
  const n = f.strokes.reduce((s, st) => s + st.pts.length, 0);
  const out = new Float32Array(n);
  let o = 0;
  for (const st of f.strokes) { out.set(st.pts, o); o += st.pts.length; }
  return out;
}

const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function overlayLine(points, color, closed = false) {
  const geo = new THREE.BufferGeometry().setFromPoints(points.map(V));
  const mat = new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 });
  const line = closed ? new THREE.LineLoop(geo, mat) : new THREE.Line(geo, mat);
  line.renderOrder = 6;
  return line;
}

function rebuildFeatureVisual(f) {
  disposeObject(f.group);
  const g = new THREE.Group();
  g.visible = f.visible;
  f.group = g;
  alignGroup.add(g);
  const color = new THREE.Color(f.color);

  const pts = featurePoints(f);
  if (pts.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
    const size = (pts.length / 3 < 50 ? 7 : 3.5) * renderer.getPixelRatio();
    g.add(new THREE.Points(geo, new THREE.PointsMaterial({ color, size, sizeAttenuation: false })));
  }
  if (f.fit) g.add(primitiveFor(f.fit, color));
  requestRender();
}

function primitiveFor(fit, color) {
  const g = new THREE.Group();
  const ghost = () => new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false });
  const { point: p, dir: d } = fit;
  switch (fit.type) {
    case 'plane': {
      const [u0, u1, v0, v1] = fit.extent;
      const pad = 0.08 * Math.max(u1 - u0, v1 - v0, state.diag * 0.02);
      const w = u1 - u0 + 2 * pad, h = v1 - v0 + 2 * pad;
      const c = G.add(p, G.add(G.scale(fit.u, (u0 + u1) / 2), G.scale(fit.v, (v0 + v1) / 2)));
      const m = new THREE.Matrix4().makeBasis(V(fit.u), V(fit.v), V(d)).setPosition(V(c));
      const quad = new THREE.Mesh(new THREE.PlaneGeometry(w, h), ghost());
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w, h)), new THREE.LineBasicMaterial({ color }));
      quad.applyMatrix4(m);
      edge.applyMatrix4(m);
      const L = 0.3 * Math.max(w, h);
      g.add(quad, edge, new THREE.ArrowHelper(V(d), V(c), L, color, L * 0.25, L * 0.12));
      break;
    }
    case 'cylinder': {
      const h = Math.max(fit.height, fit.radius * 0.5);
      const cyl = new THREE.Mesh(new THREE.CylinderGeometry(fit.radius, fit.radius, h, 64, 1, true), ghost());
      cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), V(d));
      cyl.position.copy(V(p));
      const ext = h / 2 + Math.max(h, fit.radius * 2) * 0.6;
      g.add(cyl, overlayLine([G.add(p, G.scale(d, -ext)), G.add(p, G.scale(d, ext))], color));
      break;
    }
    case 'circle': {
      const ring = [];
      for (let i = 0; i < 96; i++) {
        const a = (2 * Math.PI * i) / 96;
        ring.push(G.add(p, G.add(G.scale(fit.u, fit.radius * Math.cos(a)), G.scale(fit.v, fit.radius * Math.sin(a)))));
      }
      const k = fit.radius * 0.35;
      g.add(
        overlayLine(ring, color, true),
        overlayLine([G.add(p, G.scale(fit.u, -k)), G.add(p, G.scale(fit.u, k))], color),
        overlayLine([G.add(p, G.scale(fit.v, -k)), G.add(p, G.scale(fit.v, k))], color),
        overlayLine([p, G.add(p, G.scale(d, fit.radius * 1.5))], color),
      );
      break;
    }
    case 'line': {
      const [t0, t1] = fit.range;
      const pad = Math.max(t1 - t0, state.diag * 0.02) * 0.1;
      g.add(overlayLine([G.add(p, G.scale(d, t0 - pad)), G.add(p, G.scale(d, t1 + pad))], color));
      break;
    }
    case 'point': {
      const s = new THREE.Mesh(new THREE.SphereGeometry(state.diag * 0.004, 16, 12),
        new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }));
      s.position.copy(V(p));
      s.renderOrder = 6;
      g.add(s);
      break;
    }
  }
  return g;
}

const fmt = (x) => (Math.abs(x) < 1000 ? x.toFixed(3) : x.toFixed(1));
const fmtLen = (x) => (x < 10 ? x.toFixed(2) : x < 1000 ? x.toFixed(1) : x.toFixed(0));
const fmtVec = (v) => `(${v.map((x) => fmt(Math.abs(x) < 5e-4 ? 0 : x)).join(', ')})`;

function statsText(f) {
  const n = f.strokes.reduce((s, st) => s + st.pts.length / 3, 0);
  if (!n) return 'No points yet';
  if (f.error) return `<span class="warn">${n.toLocaleString()} pts: ${escapeHtml(f.error)}</span>`;
  const r = f.fit;
  const pts = r.used < r.total ? `${r.used.toLocaleString()}/${r.total.toLocaleString()} pts` : `${r.total.toLocaleString()} pts`;
  switch (f.type) {
    case 'plane': return `${pts} · flatness ${fmt(r.flatness)} · RMS ${fmt(r.rms)}`;
    case 'circle': return `Ø ${fmt(2 * r.radius)} · ${pts} · RMS ${fmt(r.rms)}`;
    case 'cylinder': return `Ø ${fmt(2 * r.radius)} ${r.kind || ''} · length ${fmt(r.height)} · ${pts} · RMS ${fmt(r.rms)}`;
    case 'line': return `length ${fmt(r.range[1] - r.range[0])} · ${pts} · RMS ${fmt(r.rms)}`;
    default: return `${pts} · spread ${fmt(r.rms)}`;
  }
}

// The parallel-group line under a plane's stats: who it's parallel to, how far
// apart they are, and how much its own fit was turned to match.
function parallelText(f, group) {
  const r = f.fit;
  if (!r || group.length < 2) return '';
  if (r.offBy != null) {
    return `<span class="warn">∥ Not parallel to the others (${r.offBy.toFixed(1)}° off), so it is fitted on its own.</span>`;
  }
  if (r.tilt == null) return '';
  const others = group.filter((g) => g !== f && g.fit?.tilt != null);
  const ref = others.find((g) => g.id === f.parallelTo) || others[0];
  if (!ref) return '';
  const gap = Math.abs(G.dot(G.sub(r.point, ref.fit.point), r.dir));
  return `∥ ${escapeHtml(ref.name)}, ${fmt(gap)} apart · turned ${r.tilt.toFixed(2)}° to match`;
}

const EYE = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3l18 18M10.6 5.1A10.9 10.9 0 0 1 12 5c7 0 11 7 11 7a18 18 0 0 1-3.2 3.9M6.6 6.6A18 18 0 0 0 1 12s4 7 11 7a10.6 10.6 0 0 0 5.4-1.5"/></svg>';

function renderFeatureList() {
  const list = $('#feature-list');
  if (!state.features.length) {
    list.innerHTML = `<p class="empty-hint">${state.mesh ? 'Add a feature, then click on the scan to pick points for it.' : 'Open a scan to start.'}</p>`;
    return;
  }
  const planes = state.features.filter((f) => f.type === 'plane');
  const groups = parallelGroups();
  list.innerHTML = state.features.map((f) => {
    const active = f.id === state.activeId;
    const par = f.type === 'plane' ? parallelText(f, groups.find((g) => g.includes(f))) : '';
    const parSelect = f.type === 'plane' && planes.length > 1 ? `<div class="feat-par">
        <span class="hint">Parallel to</span>
        <select data-act="par" title="Fit this plane together with another so they come out exactly parallel">
          <option value="">—</option>
          ${planes.filter((p) => p !== f).map((p) => `<option value="${p.id}" ${p.id === f.parallelTo ? 'selected' : ''}>${escapeHtml(p.name)}</option>`).join('')}
        </select>
      </div>` : '';
    return `<div class="feat${active ? ' active' : ''}" data-id="${f.id}">
      <div class="feat-head">
        <span class="swatch" style="--c:${f.color}"></span>
        <span class="feat-name">${escapeHtml(f.name)}</span>
        <span class="grow"></span>
        <button data-act="pick" class="${active ? 'on' : ''}">${active ? 'Done' : 'Pick'}</button>
        <button data-act="vis" class="icon" title="${f.visible ? 'Hide' : 'Show'}">${f.visible ? EYE : EYE_OFF}</button>
        <button data-act="del" class="icon" title="Delete">✕</button>
      </div>
      <div class="feat-stats">${statsText(f)}</div>
      ${par ? `<div class="feat-stats">${par}</div>` : ''}
      ${parSelect}
      ${active ? `<div class="feat-tools">
        <span class="muted">${TYPES[f.type].help}</span>
        <div class="row"><button data-act="undo" ${f.history.length ? '' : 'disabled'}>Undo</button><button data-act="clear" ${f.strokes.length || f.excluded.size ? '' : 'disabled'}>Clear</button></div>
      </div>` : ''}
    </div>`;
  }).join('');
}

$('#feature-list').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-act]');
  const card = e.target.closest('.feat');
  if (!card) return;
  const f = featureById(Number(card.dataset.id));
  if (!btn) { if (state.activeId !== f.id) setActive(f.id); return; }
  switch (btn.dataset.act) {
    case 'pick': setActive(state.activeId === f.id ? null : f.id); break;
    case 'vis':
      f.visible = !f.visible;
      if (f.group) f.group.visible = f.visible;
      renderFeatureList();
      requestRender();
      break;
    case 'del': deleteFeature(f.id); break;
    case 'undo': undo(f); break;
    case 'clear':
      pushHistory(f, snapshot(f));
      f.strokes = [];
      f.excluded = new Set();
      refit(f);
      break;
  }
});

$('#feature-list').addEventListener('change', (e) => {
  if (e.target.dataset.act !== 'par') return;
  const f = featureById(Number(e.target.closest('.feat').dataset.id));
  f.parallelTo = e.target.value === '' ? null : Number(e.target.value);
  updateFits(state.features.filter((x) => x.type === 'plane'));
});

$('#add-row').addEventListener('click', (e) => {
  const b = e.target.closest('[data-add]');
  if (b) addFeature(b.dataset.add);
});

// ---------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------

const raycaster = new THREE.Raycaster();
raycaster.firstHitOnly = true;
const ndc = new THREE.Vector2();

function hitAt(ev) {
  if (!state.mesh) return null;
  const r = canvas.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  return raycaster.intersectObject(state.mesh, false)[0] || null;
}

// A stroke from mesh vertices, skipping vertices the feature already has and,
// unless reclaim is set, vertices that were erased from it.
function gather(f, verts, reclaim = false) {
  const have = new Set();
  for (const st of f.strokes) if (st.ids) for (const id of st.ids) have.add(id);
  const ids = Uint32Array.from(verts.filter((v) => !have.has(v) && (reclaim || !f.excluded.has(v))));
  if (!ids.length) return null;
  const { pos, nrm } = state.data;
  const pts = new Float32Array(ids.length * 3), nrms = new Float32Array(ids.length * 3);
  ids.forEach((v, k) => {
    pts.set(pos.subarray(3 * v, 3 * v + 3), 3 * k);
    nrms.set(nrm.subarray(3 * v, 3 * v + 3), 3 * k);
  });
  return { pts, nrms, ids };
}

const localPoint = (hit) => state.mesh.worldToLocal(hit.point.clone()).toArray();

function pickAt(ev) {
  const f = activeFeature();
  if (!f) return;
  const hit = hitAt(ev);
  if (!hit) return;
  const p = localPoint(hit);
  const s = pickSettings();

  let stroke, reclaim = false;
  if (s.mode === 'point') {
    stroke = { pts: Float32Array.from(p), nrms: Float32Array.from(hit.face.normal.toArray()) };
  } else if (s.mode === 'patch') {
    const { pos, nrm } = state.data;
    const tri = [hit.face.a, hit.face.b, hit.face.c];
    const seedNormal = tri.reduce((acc, v) => G.add(acc, [nrm[3 * v], nrm[3 * v + 1], nrm[3 * v + 2]]), [0, 0, 0]);
    // Erased points stay out of later patches, except when the click lands
    // inside an erased area: then the user is asking for it back.
    const d2 = (v) => (pos[3 * v] - p[0]) ** 2 + (pos[3 * v + 1] - p[1]) ** 2 + (pos[3 * v + 2] - p[2]) ** 2;
    reclaim = f.excluded.has(tri.reduce((a, b) => (d2(b) < d2(a) ? b : a)));
    stroke = gather(f, G.growRegion(state.data, tri, p, { radius: s.radius, maxAngle: s.angle, angleRef: s.ref, seedNormal }), reclaim);
  } else {
    if (!state.boundary) state.boundary = G.boundaryGraph(state.data);
    const loop = G.boundaryLoopNear(state.data, state.boundary, p, s.radius, Math.max(s.radius, state.diag * 0.02));
    if (!loop) {
      toast(state.boundary.size
        ? 'No open edge near that click. Click right next to the gap.'
        : 'This scan has no open edges (it is watertight). Use Single point on the hole edge instead.');
      return;
    }
    reclaim = true;
    stroke = gather(f, loop, reclaim);
  }
  if (!stroke) {
    toast(f.excluded.size
      ? 'Nothing new there: the rest was erased. Click inside an erased spot to add it back.'
      : 'Those points are already selected.');
    return;
  }
  pushHistory(f, snapshot(f));
  if (reclaim && f.excluded.size && stroke.ids) {
    const ex = new Set(f.excluded);
    for (const v of stroke.ids) ex.delete(v);
    f.excluded = ex;
  }
  f.strokes.push(stroke);
  refit(f);
}

// Remove the feature's points within r of the segment a–b: one brush step
// between two pointer samples (a === b for a plain click). Erased mesh
// vertices go into f.excluded so later patches don't flood back over them;
// the caller gives f a fresh excluded set first, since old ones are in undo
// snapshots.
function eraseAlong(f, a, b, r) {
  const r2 = r * r;
  const ab = G.sub(b, a);
  const L2 = G.dot(ab, ab);
  let removed = 0;
  f.strokes = f.strokes.map((st) => {
    const keep = [];
    for (let i = 0; i < st.pts.length; i += 3) {
      const px = st.pts[i] - a[0], py = st.pts[i + 1] - a[1], pz = st.pts[i + 2] - a[2];
      const t = L2 > 0 ? Math.min(1, Math.max(0, (px * ab[0] + py * ab[1] + pz * ab[2]) / L2)) : 0;
      const dx = px - t * ab[0], dy = py - t * ab[1], dz = pz - t * ab[2];
      if (dx * dx + dy * dy + dz * dz > r2) keep.push(i / 3);
      else if (st.ids) f.excluded.add(st.ids[i / 3]);
    }
    if (keep.length === st.pts.length / 3) return st;
    removed += st.pts.length / 3 - keep.length;
    const pts = new Float32Array(keep.length * 3), nrms = new Float32Array(keep.length * 3);
    keep.forEach((k, j) => {
      pts.set(st.pts.subarray(3 * k, 3 * k + 3), 3 * j);
      nrms.set(st.nrms.subarray(3 * k, 3 * k + 3), 3 * j);
    });
    return { pts, nrms, ids: st.ids && Uint32Array.from(keep, (k) => st.ids[k]) };
  }).filter((st) => st.pts.length);
  return removed;
}

// Erasing is a brush: Shift+click/drag, or plain click/drag in Erase mode,
// wipes the active feature's points under the cursor. A whole drag is one
// undo step, and the fit updates when the button is released.
let erase = null; // { f, before, last, removed } while erasing
let shiftHeld = false;
const erasing = () => !!activeFeature() && (state.eraseMode || shiftHeld);

function eraseTo(p) {
  const n = eraseAlong(erase.f, erase.last, p, state.eraser);
  erase.last = p;
  if (n) {
    erase.removed += n;
    rebuildFeatureVisual(erase.f);
  }
}

function finishErase(ev) {
  if (ev) {
    const hit = hitAt(ev);
    if (hit) eraseTo(localPoint(hit));
  }
  const { f, before, removed } = erase;
  erase = null;
  controls.enabled = true;
  if (removed) {
    pushHistory(f, before);
    refit(f);
  } else f.excluded = before.excluded;
}

// Capture phase on the parent, so this runs before OrbitControls sees the
// press and can switch it off. A press that misses the scan still orbits.
$('#view').addEventListener('pointerdown', (e) => {
  shiftHeld = e.shiftKey;
  if (e.target !== canvas || e.button !== 0 || !erasing() || e.ctrlKey || e.metaKey) return;
  const hit = hitAt(e);
  if (!hit) return;
  const f = activeFeature();
  controls.enabled = false;
  canvas.setPointerCapture(e.pointerId);
  const p = localPoint(hit);
  erase = { f, before: snapshot(f), last: p, removed: 0 };
  f.excluded = new Set(f.excluded);
  eraseTo(p);
  updateCursor();
}, { capture: true });

let pointerDown = null;
canvas.addEventListener('pointerdown', (e) => {
  pointerDown = { x: e.clientX, y: e.clientY, button: e.button };
});
canvas.addEventListener('pointerup', (e) => {
  const d = pointerDown;
  pointerDown = null;
  if (erase) { finishErase(e); return; }
  if (d && d.button === 0 && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 5) pickAt(e);
});
canvas.addEventListener('pointercancel', () => { if (erase) finishErase(); });

let lastPointer = null; // latest pointer event over the view, for cursor updates
let moveQueued = false;
canvas.addEventListener('pointermove', (e) => {
  lastPointer = e;
  shiftHeld = e.shiftKey;
  if (moveQueued) return;
  moveQueued = true;
  requestAnimationFrame(() => { moveQueued = false; updateCursor(); });
});
canvas.addEventListener('pointerleave', () => {
  if (erase) return;
  lastPointer = null;
  cursor.visible = false;
  requestRender();
});

function updateCursor() {
  const e = lastPointer;
  const f = activeFeature();
  const hit = f && e && (!e.buttons || erase) ? hitAt(e) : null;
  if (erase && hit) eraseTo(localPoint(hit));
  cursor.visible = !!hit;
  if (hit) {
    const s = pickSettings();
    const rubber = !!erase || erasing();
    const n = hit.face.normal.clone().transformDirection(state.mesh.matrixWorld);
    cursor.position.copy(hit.point);
    cursor.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
    cursor.scale.setScalar(rubber ? state.eraser : s.mode === 'patch' && s.radius > 0 ? s.radius : state.diag * 0.006);
    cursor.material.color.set(rubber ? '#f87171' : f.color);
  }
  requestRender();
}

function setEraser(r) {
  if (!(r > 0)) return;
  state.eraser = Number(r.toPrecision(2));
  $('#pick-eraser').value = state.eraser;
  updateCursor();
}

// Pick tool controls
function syncPickControls() {
  const s = pickSettings();
  if (!s) return;
  for (const b of $$('#pick-mode [data-mode]')) b.classList.toggle('on', b.dataset.mode === s.mode);
  $('#pick-radius').value = s.radius;
  $('#pick-angle').value = s.angle;
  $('#pick-ref').value = s.ref;
  $('#pick-angle').disabled = $('#pick-ref').disabled = s.mode !== 'patch';
  $('#pick-eraser').value = state.eraser;
  $('#pick-erase').classList.toggle('on', state.eraseMode);
}

$('#pick-mode').addEventListener('click', (e) => {
  const b = e.target.closest('[data-mode]');
  const s = pickSettings();
  if (!b || !s) return;
  s.mode = b.dataset.mode;
  syncPickControls();
});
$('#pick-radius').addEventListener('change', (e) => {
  const s = pickSettings();
  const v = Number(e.target.value);
  if (s && v >= 0) s.radius = v;
  syncPickControls();
});
$('#pick-angle').addEventListener('change', (e) => {
  const s = pickSettings();
  const v = Number(e.target.value);
  if (s && v > 0 && v < 90) s.angle = v;
  syncPickControls();
});
$('#pick-ref').addEventListener('change', (e) => {
  const s = pickSettings();
  if (s) s.ref = e.target.value;
});
$('#pick-eraser').addEventListener('change', (e) => {
  setEraser(Number(e.target.value));
  syncPickControls();
});
$('#pick-erase').addEventListener('click', () => {
  if (!activeFeature()) return;
  state.eraseMode = !state.eraseMode;
  syncPickControls();
  updateStatus();
  updateCursor();
});

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

function optionsHtml(features, selected, noneLabel) {
  const opts = [`<option value="">${noneLabel}</option>`];
  for (const f of features) {
    opts.push(`<option value="${f.id}" ${f.id === selected ? 'selected' : ''}>${escapeHtml(f.name)}${f.fit ? '' : ' (not fitted)'}</option>`);
  }
  return opts.join('');
}

function refreshAlignControls() {
  const a = state.align;
  const directional = state.features.filter((f) => f.type !== 'point');
  $('#pri-id').innerHTML = optionsHtml(directional, a.primary.id, '— scan axis —');
  $('#sec-id').innerHTML = optionsHtml(state.features, a.secondary.id, '— scan axis —');
  for (const el of $$('select.origin')) {
    el.innerHTML = optionsHtml(state.features, a.origin[Number(el.dataset.k)], '—');
  }
  $('#pri-target').innerHTML = TARGETS.map((t) => `<option value="${t}" ${t === a.primary.target ? 'selected' : ''}>${prettyTarget(t)}</option>`).join('');
  const secTargets = TARGETS.filter((t) => t[1] !== a.primary.target[1]);
  if (!secTargets.includes(a.secondary.target)) a.secondary.target = secTargets[0];
  $('#sec-target').innerHTML = secTargets.map((t) => `<option value="${t}" ${t === a.secondary.target ? 'selected' : ''}>${prettyTarget(t)}</option>`).join('');
  $('#sec-use').value = a.secondary.use;
}

const idFrom = (v) => (v === '' ? null : Number(v));
const flipTarget = (t) => (t[0] === '-' ? '+' : '-') + t[1];

$('#pri-id').addEventListener('change', (e) => { state.align.primary.id = idFrom(e.target.value); updateAlignment(); });
$('#pri-target').addEventListener('change', (e) => { state.align.primary.target = e.target.value; refreshAlignControls(); updateAlignment(); });
$('#pri-flip').addEventListener('click', () => { state.align.primary.target = flipTarget(state.align.primary.target); refreshAlignControls(); updateAlignment(); });
$('#sec-id').addEventListener('change', (e) => { state.align.secondary.id = idFrom(e.target.value); updateAlignment(); });
$('#sec-target').addEventListener('change', (e) => { state.align.secondary.target = e.target.value; updateAlignment(); });
$('#sec-flip').addEventListener('click', () => { state.align.secondary.target = flipTarget(state.align.secondary.target); refreshAlignControls(); updateAlignment(); });
$('#sec-use').addEventListener('change', (e) => { state.align.secondary.use = e.target.value; updateAlignment(); });
for (const el of $$('select.origin')) {
  el.addEventListener('change', () => { state.align.origin[Number(el.dataset.k)] = idFrom(el.value); updateAlignment(); });
}
$('#angle').addEventListener('input', (e) => { state.align.angle = Number(e.target.value) || 0; updateAlignment(); });
$('#rot90').addEventListener('click', () => {
  state.align.angle = (((state.align.angle + 90) % 360) + 360) % 360;
  $('#angle').value = state.align.angle;
  updateAlignment();
});
for (const el of $$('[data-off]')) {
  el.addEventListener('input', () => { state.align.offset[Number(el.dataset.off)] = Number(el.value) || 0; updateAlignment(); });
}
$('#preview').addEventListener('change', (e) => { state.preview = e.target.checked; updateAlignment(); fitView(); });

function updateAlignment() {
  if (!state.mesh) { $('#notes').innerHTML = ''; $('#readout').innerHTML = ''; return; }
  const fits = new Map(state.features.filter((f) => f.fit).map((f) => [f.id, f.fit]));
  const res = G.computeAlignment(fits, state.align, state.centroid);
  state.result = res;
  alignGroup.matrix.copy(state.preview ? new THREE.Matrix4().set(...res.matrix) : new THREE.Matrix4());
  alignGroup.updateMatrixWorld(true);
  renderAlignInfo(res);

  // Keep the part in view if the alignment moved it far from the camera target.
  const sphere = state.mesh.geometry.boundingSphere.clone().applyMatrix4(state.mesh.matrixWorld);
  if (sphere.center.distanceTo(controls.target) > sphere.radius * 1.5) fitView();
  requestRender();
}

function renderAlignInfo(res) {
  const a = state.align;
  const pf = featureById(a.primary.id);
  const sf = featureById(a.secondary.id);
  const M = res.matrix;

  let hint = '';
  if (pf?.fit) {
    const t = prettyTarget(a.primary.target);
    if (pf.type === 'plane' && a.primary.target[1] === 'Z') {
      hint = a.primary.target[0] === '-'
        ? `Face normal → ${t}: it is the bottom face, the part sits on the XY plane.`
        : `Face normal → ${t}: it is a top face, the part is below the XY plane.`;
    } else {
      hint = `${pf.type === 'plane' || pf.type === 'circle' ? 'Normal' : 'Axis'} of ${escapeHtml(pf.name)} → ${t}.`;
    }
  } else if (pf) hint = `${escapeHtml(pf.name)} has no fit yet.`;
  else hint = 'Pick a feature to set an axis; otherwise the scan\'s own axes are kept.';
  $('#pri-hint').innerHTML = hint;

  const notes = [...res.notes];
  if (sf?.fit && res.secondaryMode === 'direction' && pf?.fit) {
    const ang = (Math.acos(Math.min(1, Math.abs(G.dot(sf.fit.dir, pf.fit.dir)))) * 180) / Math.PI;
    notes.push(`${sf.name} sets ${prettyTarget(a.secondary.target)} from its ${sf.type === 'plane' ? 'normal' : 'axis'} (${ang.toFixed(2)}° from the primary).`);
  } else if (sf?.fit && res.secondaryMode === 'toward') {
    notes.push(`${prettyTarget(a.secondary.target)} points from the origin toward ${sf.name}.`);
  }
  $('#notes').innerHTML = notes.map((n) => `<p class="muted">${escapeHtml(n)}</p>`).join('');

  // Features and mesh extents in the new frame.
  const rows = [];
  for (const f of state.features) {
    if (!f.fit) continue;
    const r = f.fit;
    let p = G.transformPoint(M, r.point);
    const d = r.dir ? G.transformDir(M, r.dir) : null;
    let where = `at ${fmtVec(p)}`;
    if ((r.type === 'cylinder' || r.type === 'line') && Math.abs(d[2]) > 0.7) {
      p = G.add(p, G.scale(d, -p[2] / d[2])); // where the axis crosses Z = 0
      where = `crosses Z=0 at ${fmtVec(p.slice(0, 2))}`;
    }
    const size = r.radius ? `Ø ${fmt(2 * r.radius)} · ` : '';
    const dirLabel = d ? `<br><span class="k">${r.type === 'plane' || r.type === 'circle' ? 'normal' : 'axis'}</span> ${fmtVec(d)}` : '';
    rows.push(`<tr><td><span class="swatch" style="--c:${f.color};display:inline-block"></span> ${escapeHtml(f.name)}</td><td>${size}${where}${dirLabel}</td></tr>`);
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const pos = state.data.pos;
  for (let k = 0; k < 3; k++) {
    const [a, b, c, t] = M.slice(4 * k, 4 * k + 4);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < pos.length; i += 3) {
      const q = a * pos[i] + b * pos[i + 1] + c * pos[i + 2] + t;
      if (q < mn) mn = q;
      if (q > mx) mx = q;
    }
    lo[k] = mn;
    hi[k] = mx;
  }
  rows.push(`<tr><td class="k">Part extents</td><td>${'XYZ'.split('').map((ax, k) => `${ax} ${fmt(lo[k])} … ${fmt(hi[k])}`).join('<br>')}</td></tr>`);
  $('#readout').innerHTML = `<table>${rows.join('')}</table>`;
}

// ---------------------------------------------------------------------------
// Simplify (export only: picking and fitting always use the full scan)
// ---------------------------------------------------------------------------

// Keep only the vertices the index uses, renumbered.
function compactMesh(pos, index, colors) {
  const map = new Int32Array(pos.length / 3).fill(-1);
  let n = 0;
  for (const v of index) if (map[v] < 0) map[v] = n++;
  const positions = new Float32Array(3 * n), cols = colors ? new Float32Array(3 * n) : null;
  for (let v = 0; v < map.length; v++) {
    if (map[v] < 0) continue;
    positions.set(pos.subarray(3 * v, 3 * v + 3), 3 * map[v]);
    if (cols) cols.set(colors.subarray(3 * v, 3 * v + 3), 3 * map[v]);
  }
  return { positions, index: Uint32Array.from(index, (v) => map[v]), colors: cols };
}

async function runSimplify() {
  const S = state.simplify;
  clearSimplified();
  if (!S.on || !state.mesh) { syncSimplifyControls(); return; }
  const total = state.data.index.length / 3;
  busy(`Simplifying ${total.toLocaleString()} triangles…`);
  await nextFrame();
  try {
    const { index } = G.simplifyMesh(state.data.pos, state.data.index, {
      targetTriangles: Math.round((total * S.keep) / 100),
      maxDeviation: S.tol,
    });
    const colorAttr = state.mesh.geometry.attributes.color;
    const res = compactMesh(state.data.pos, index, colorAttr ? colorAttr.array : null);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(res.positions, 3));
    if (res.colors) geom.setAttribute('color', new THREE.BufferAttribute(res.colors, 3));
    geom.setIndex(new THREE.BufferAttribute(res.index, 1));
    geom.computeBoundingSphere();
    geom.computeBoundsTree(); // reorders res.index in place; same triangles

    // How far did the surface move? Distance from a sample of the original
    // vertices to the simplified surface.
    busy('Measuring deviation…');
    await nextFrame();
    const pos = state.data.pos, vc = pos.length / 3;
    const step = Math.max(1, Math.floor(vc / 20000));
    const p = new THREE.Vector3(), hit = {};
    let max = 0, ss = 0, count = 0;
    for (let v = 0; v < vc; v += step) {
      geom.boundsTree.closestPointToPoint(p.fromArray(pos, 3 * v), hit);
      max = Math.max(max, hit.distance);
      ss += hit.distance * hit.distance;
      count++;
    }
    res.maxDev = max;
    res.rmsDev = Math.sqrt(ss / count);
    res.sampled = step > 1;
    S.result = res;

    // Preview: show the simplified mesh in place of the scan. The scan stays
    // in the scene, hidden, so clicks still pick on the full-resolution surface.
    const mat = meshMaterial.clone();
    mat.flatShading = true;
    const preview = new THREE.Group();
    preview.add(new THREE.Mesh(geom, mat));
    const edges = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ color: 0x0b0d10, wireframe: true, transparent: true, opacity: 0.45 }));
    edges.visible = S.edges;
    preview.add(edges);
    alignGroup.add(preview);
    S.preview = preview;
    state.mesh.visible = false;
  } catch (err) {
    console.error(err);
    S.on = false;
    toast(`Simplify failed: ${err.message}`, true);
  } finally {
    busy(null);
    syncSimplifyControls();
    requestRender();
  }
}

function clearSimplified() {
  const S = state.simplify;
  if (S.preview) {
    S.preview.children[0].geometry.disposeBoundsTree();
    disposeObject(S.preview);
  }
  S.preview = null;
  S.result = null;
  if (state.mesh) state.mesh.visible = true;
  requestRender();
}

function syncSimplifyControls() {
  const S = state.simplify;
  $('#simp-on').checked = S.on;
  $('#simp-opts').hidden = !S.on;
  $('#simp-keep').value = S.keep;
  $('#simp-tol').value = S.tol;
  $('#simp-edges').checked = S.edges;
  const r = S.result;
  if (!r || !state.mesh) { $('#simp-info').innerHTML = ''; return; }
  const total = state.data.index.length / 3, n = r.index.length / 3;
  const short = S.keep > 0 && n > 1.02 * Math.round((total * S.keep) / 100);
  $('#simp-info').innerHTML =
    `${total.toLocaleString()} → <b>${n.toLocaleString()}</b> triangles (${((100 * n) / total).toPrecision(2)}%)<br>` +
    `Deviation from the scan: max ${fmt(r.maxDev)}, RMS ${fmt(r.rmsDev)}${r.sampled ? ' (sampled)' : ''}` +
    (short ? '<br><span class="warn">Stopped at the tolerance before reaching the target. Raise the tolerance to go further.</span>' : '');
}

$('#simp-on').addEventListener('change', (e) => { state.simplify.on = e.target.checked; runSimplify(); });
$('#simp-keep').addEventListener('change', (e) => {
  const v = Number(e.target.value);
  if (v >= 0 && v <= 100) { state.simplify.keep = v; runSimplify(); } else syncSimplifyControls();
});
$('#simp-tol').addEventListener('change', (e) => {
  const v = Number(e.target.value);
  if (v >= 0) { state.simplify.tol = v; runSimplify(); } else syncSimplifyControls();
});
$('#simp-edges').addEventListener('change', (e) => {
  state.simplify.edges = e.target.checked;
  if (state.simplify.preview) state.simplify.preview.children[1].visible = state.simplify.edges;
  requestRender();
});

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
}

const baseName = () => state.fileName.replace(/\.[^.]+$/, '') || 'scan';

$('#btn-stl').addEventListener('click', async () => {
  if (!state.mesh) return;
  busy('Writing STL…');
  await nextFrame();
  try {
    const simp = state.simplify.on ? state.simplify.result : null;
    const buf = simp
      ? G.writeBinarySTL(simp.positions, simp.index, state.result.matrix)
      : G.writeBinarySTL(state.data.pos, state.data.index, state.result.matrix);
    download(new Blob([buf], { type: 'model/stl' }), `${baseName()}_aligned${simp ? '_simplified' : ''}.stl`);
  } finally {
    busy(null);
  }
});

$('#btn-step').addEventListener('click', () => {
  if (!state.mesh) return;
  const features = state.features
    .filter((f) => f.fit && ['plane', 'circle', 'cylinder'].includes(f.type))
    .map((f) => ({ name: f.name, fit: f.fit }));
  if (!features.length) { toast('Fit at least one plane, circle or cylinder first.'); return; }
  const name = `${baseName()}_reference`;
  const { text, count } = writeStepReference(features, state.result.matrix, { units: $('#units').value, name });
  download(new Blob([text], { type: 'model/step' }), `${name}.step`);
  toast(`Exported ${count} reference surface${count > 1 ? 's' : ''}.`);
});

$('#btn-json').addEventListener('click', () => {
  if (!state.mesh) return;
  const M = state.result.matrix;
  const round = (v) => v.map((x) => Number(x.toFixed(6)));
  const out = {
    source: state.fileName,
    note: 'aligned = matrix · [x, y, z, 1] with x, y, z in the original scan coordinates (row-major).',
    matrix: [0, 1, 2, 3].map((r) => round(M.slice(4 * r, 4 * r + 4))),
    features: state.features.filter((f) => f.fit).map((f) => ({
      name: f.name,
      type: f.type,
      ...(f.fit.radius ? { diameter: Number((2 * f.fit.radius).toFixed(4)) } : {}),
      point: round(G.transformPoint(M, f.fit.point)),
      ...(f.fit.dir ? { direction: round(G.transformDir(M, f.fit.dir)) } : {}),
      ...(f.fit.tilt != null ? { parallelGroup: groupOf(f).filter((g) => g !== f && g.fit?.tilt != null).map((g) => g.name) } : {}),
      rms: Number(f.fit.rms.toFixed(4)),
      points: f.fit.total,
    })),
  };
  download(new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' }), `${baseName()}_transform.json`);
});

// ---------------------------------------------------------------------------
// Shell: file input, drag & drop, views, keys, feedback
// ---------------------------------------------------------------------------

$('#btn-open').addEventListener('click', () => $('#file').click());
$('#file').addEventListener('change', (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (file) openFile(file);
});

const view = $('#view');
view.addEventListener('dragover', (e) => { e.preventDefault(); view.classList.add('dragging'); });
view.addEventListener('dragleave', () => view.classList.remove('dragging'));
view.addEventListener('drop', (e) => {
  e.preventDefault();
  view.classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (file) openFile(file);
});

$('#viewbar').addEventListener('click', (e) => {
  const b = e.target.closest('[data-view]');
  if (b) fitView(b.dataset.view === 'fit' ? undefined : b.dataset.view);
});

// Shift turns the cursor into the eraser as soon as it's pressed.
for (const type of ['keydown', 'keyup']) {
  window.addEventListener(type, (e) => {
    if (e.key !== 'Shift' || shiftHeld === (type === 'keydown')) return;
    shiftHeld = type === 'keydown';
    updateCursor();
  });
}
window.addEventListener('blur', () => { shiftHeld = false; });

window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  const f = activeFeature();
  if (e.key === 'Escape') setActive(null);
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
    if (f?.history.length && !erase) { e.preventDefault(); undo(f); }
  } else if (f && (e.code === 'BracketLeft' || e.code === 'BracketRight')) {
    setEraser(state.eraser * (e.code === 'BracketRight' ? 1.25 : 0.8));
  } else if (e.key === 'f' || e.key === 'F') fitView();
});

function busy(text) {
  $('#busy').hidden = !text;
  if (text) $('#busy-text').textContent = text;
}

let toastTimer = 0;
function toast(msg, isError = false) {
  const el = $('#toast');
  el.textContent = msg;
  el.className = `show${isError ? ' error' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ''; }, isError ? 6000 : 3500);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

renderFeatureList();
refreshAlignControls();
syncSimplifyControls();
updateAlignment();
requestRender();

// Test hook: lets an automated browser session drive the app.
window.__meshAlign = { state, openFile, addFeature, pickAt, setActive, updateAlignment, camera, controls, fitView, undo, runSimplify };

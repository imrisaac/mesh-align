import { test } from 'node:test';
import assert from 'node:assert/strict';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import * as G from '../src/geom.js';

// Deterministic RNG + helpers for synthetic "scans".
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const R = rng(1234);
const gauss = () => Math.sqrt(-2 * Math.log(R() + 1e-12)) * Math.cos(2 * Math.PI * R());
const noisy = (p, s) => [p[0] + s * gauss(), p[1] + s * gauss(), p[2] + s * gauss()];

// Random rigid pose: rotation matrix (rows) + translation.
function randomPose() {
  const q = G.normalize([gauss(), gauss(), gauss()]);
  const w = gauss();
  const n = Math.hypot(q[0], q[1], q[2], w);
  const [x, y, z, s] = [q[0] / n, q[1] / n, q[2] / n, w / n];
  const rot = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - s * z), 2 * (x * z + s * y)],
    [2 * (x * y + s * z), 1 - 2 * (x * x + z * z), 2 * (y * z - s * x)],
    [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x * x + y * y)],
  ];
  const t = [200 * R() - 100, 200 * R() - 100, 200 * R() - 100];
  return {
    p: (v) => G.add([G.dot(rot[0], v), G.dot(rot[1], v), G.dot(rot[2], v)], t),
    d: (v) => [G.dot(rot[0], v), G.dot(rot[1], v), G.dot(rot[2], v)],
  };
}

function pack(pts) {
  const a = new Float64Array(pts.length * 3);
  pts.forEach((p, i) => a.set(p, 3 * i));
  return a;
}

const angleDeg = (a, b) => (Math.acos(Math.min(1, Math.abs(G.dot(G.normalize(a), G.normalize(b))))) * 180) / Math.PI;
const dist = (a, b) => G.len(G.sub(a, b));

test('eigSym3 decomposes a symmetric matrix', () => {
  const M = [[4, 1, -2], [1, 2, 0.5], [-2, 0.5, 3]];
  const { values, vectors } = G.eigSym3(M);
  assert.ok(values[0] <= values[1] && values[1] <= values[2]);
  for (let k = 0; k < 3; k++) {
    const v = vectors[k];
    const Mv = [G.dot(M[0], v), G.dot(M[1], v), G.dot(M[2], v)];
    assert.ok(dist(Mv, G.scale(v, values[k])) < 1e-10);
  }
});

test('plane fit: noisy patch with outliers, outward normal', () => {
  const pose = randomPose();
  const P = [], N = [];
  for (let i = 0; i < 4000; i++) {
    const local = [60 * R() - 30, 40 * R() - 20, 0];
    if (i % 25 === 0) local[2] = 2 + 3 * R(); // 4% junk (e.g. a chamfer)
    P.push(pose.p(noisy(local, 0.03)));
    N.push(G.normalize(pose.d(noisy([0, 0, 1], 0.1))));
  }
  const fit = G.fitFeature('plane', pack(P), pack(N));
  assert.ok(!fit.error);
  const nTrue = pose.d([0, 0, 1]);
  assert.ok(G.dot(fit.dir, nTrue) > 0, 'normal should follow surface normals');
  assert.ok(angleDeg(fit.dir, nTrue) < 0.1, `normal error ${angleDeg(fit.dir, nTrue)}`);
  assert.ok(Math.abs(G.dot(G.sub(fit.point, pose.p([0, 0, 0])), nTrue)) < 0.01);
  assert.ok(fit.used < 4000 && fit.used > 3700, `used ${fit.used}`);
  assert.ok(fit.rms < 0.05);
});

test('circle fit: exact through 3 points, accurate on noisy rim', () => {
  const exact = G.fitFeature('circle', pack([[5, 0, 1], [0, 5, 1], [-5, 0, 1]]), null);
  assert.ok(dist(exact.point, [0, 0, 1]) < 1e-9);
  assert.ok(Math.abs(exact.radius - 5) < 1e-9);

  const pose = randomPose();
  const P = [], N = [];
  for (let i = 0; i < 40; i++) {
    const a = 2 * Math.PI * R();
    P.push(pose.p(noisy([12 + 3.1 * Math.cos(a), -4 + 3.1 * Math.sin(a), 0], 0.02)));
    N.push(pose.d([0, 0, 1]));
  }
  const fit = G.fitFeature('circle', pack(P), pack(N));
  assert.ok(dist(fit.point, pose.p([12, -4, 0])) < 0.02, `center err ${dist(fit.point, pose.p([12, -4, 0]))}`);
  assert.ok(Math.abs(fit.radius - 3.1) < 0.02);
  assert.ok(G.dot(fit.dir, pose.d([0, 0, 1])) > 0.999);
});

test('cylinder fit: partial hole wall with noisy normals', () => {
  const pose = randomPose();
  const P = [], N = [];
  for (let i = 0; i < 3000; i++) {
    const a = 1.5 * Math.PI * R(); // 270 degrees of wall
    const z = 8 * R();
    P.push(pose.p(noisy([3 * Math.cos(a) + 10, 3 * Math.sin(a) + 20, z], 0.02)));
    // hole: normals point toward the axis
    N.push(G.normalize(pose.d(noisy([-Math.cos(a), -Math.sin(a), 0], 0.15))));
  }
  const fit = G.fitFeature('cylinder', pack(P), pack(N));
  assert.ok(!fit.error, fit.error);
  const axisTrue = pose.d([0, 0, 1]);
  assert.ok(angleDeg(fit.dir, axisTrue) < 0.5, `axis err ${angleDeg(fit.dir, axisTrue)}`);
  assert.ok(Math.abs(fit.radius - 3) < 0.02, `radius ${fit.radius}`);
  assert.equal(fit.kind, 'hole');
  assert.ok(Math.abs(fit.height - 8) < 0.3);
  // mid-height axis point
  assert.ok(dist(fit.point, pose.p([10, 20, 4])) < 0.1, `point err ${dist(fit.point, pose.p([10, 20, 4]))}`);
});

test('cylinder fit without normals (boss) still finds the axis', () => {
  const P = [];
  for (let i = 0; i < 2000; i++) {
    const a = 2 * Math.PI * R();
    P.push(noisy([6 * Math.cos(a), 30 * R(), 6 * Math.sin(a)], 0.01));
  }
  const fit = G.fitFeature('cylinder', pack(P), null);
  assert.ok(angleDeg(fit.dir, [0, 1, 0]) < 0.2);
  assert.ok(Math.abs(fit.radius - 6) < 0.01);
});

test('line and point fits', () => {
  const line = G.fitFeature('line', pack([[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0]]), null);
  assert.ok(angleDeg(line.dir, [1, 1, 0]) < 1e-4);
  const pt = G.fitFeature('point', pack([[1, 2, 3], [3, 2, 1]]), null);
  assert.ok(dist(pt.point, [2, 2, 2]) < 1e-12);
  assert.ok(G.fitFeature('plane', pack([[0, 0, 0], [1, 1, 1]]), null).error);
});

// A noisy scanned patch of the plane through `origin` with normal `n`
// (unit), w x h, as { P, N }.
function scannedPatch(origin, n, w, h, count, noise = 0.02) {
  const [u, v] = G.perpBasis(n);
  const P = [], N = [];
  for (let i = 0; i < count; i++) {
    const p = G.add(origin, G.add(G.scale(u, w * (R() - 0.5)), G.scale(v, h * (R() - 0.5))));
    P.push(noisy(p, noise));
    N.push(G.normalize(noisy(n, 0.05)));
  }
  return { P: pack(P), N: pack(N) };
}
const tilted = (deg, axis = 0) => {
  const a = (deg * Math.PI) / 180;
  return axis === 0 ? [0, -Math.sin(a), Math.cos(a)] : [Math.sin(a), 0, Math.cos(a)];
};

test('parallel planes: top and bottom of a plate share one normal', () => {
  const pose = randomPose();
  // The scan has the faces 0.4° apart in opposite directions.
  const top = scannedPatch([0, 0, 10], tilted(0.2), 60, 40, 3000);
  const bot = scannedPatch([0, 0, 0], G.scale(tilted(-0.2), -1), 60, 40, 3000);
  const place = ({ P, N }) => {
    const Q = new Float64Array(P.length), M = new Float64Array(N.length);
    for (let i = 0; i < P.length; i += 3) {
      Q.set(pose.p([P[i], P[i + 1], P[i + 2]]), i);
      M.set(pose.d([N[i], N[i + 1], N[i + 2]]), i);
    }
    return { P: Q, N: M };
  };
  const [a, b] = G.fitParallelPlanes([place(top), place(bot)]);
  assert.ok(!a.error && !b.error);
  assert.ok(G.dot(a.dir, b.dir) < -1 + 1e-12, 'exactly antiparallel');
  assert.ok(angleDeg(a.dir, pose.d([0, 0, 1])) < 0.05, `shared normal ${angleDeg(a.dir, pose.d([0, 0, 1]))}° off`);
  assert.ok(Math.abs(a.tilt - 0.2) < 0.05 && Math.abs(b.tilt - 0.2) < 0.05, `tilts ${a.tilt}, ${b.tilt}`);
  assert.ok(G.dot(a.dir, pose.d([0, 0, 1])) > 0 && G.dot(b.dir, pose.d([0, 0, -1])) > 0, 'each keeps its outward side');
  const gap = Math.abs(G.dot(G.sub(a.point, b.point), a.dir));
  assert.ok(Math.abs(gap - 10) < 0.01, `thickness ${gap}`);
  assert.ok(a.flatness > 0 && a.extent.length === 4 && a.used <= a.total);
});

test('parallel planes: bigger faces steer the normal, junk is rejected', () => {
  const big = scannedPatch([0, 0, 0], tilted(0.5, 1), 100, 100, 6000);
  const small = scannedPatch([0, 0, 5], [0, 0, 1], 10, 10, 300);
  const step = scannedPatch([30, 0, 2], tilted(-0.3, 1), 20, 40, 1500);
  // Some junk on the step (an edge fillet caught in the patch).
  for (let i = 0; i < 60; i++) step.P[3 * i + 2] += 1 + R();
  const r = G.fitParallelPlanes([big, small, step]);
  for (const f of r) assert.ok(!f.error && f.offBy === undefined);
  assert.ok(Math.abs(G.dot(r[0].dir, r[1].dir)) > 1 - 1e-12 && Math.abs(G.dot(r[0].dir, r[2].dir)) > 1 - 1e-12);
  assert.ok(angleDeg(r[0].dir, tilted(0.5, 1)) < angleDeg(r[0].dir, [0, 0, 1]), 'closer to the big face');
  assert.ok(r[2].used <= r[2].total - 60, `junk kept: ${r[2].total - r[2].used} dropped`);
  assert.ok(Math.abs(G.dot(G.sub(r[2].point, [30, 0, 2]), r[2].dir)) < 0.05, 'step plane stays on its points');
});

test('parallel planes: a face that is not parallel is left out, not forced', () => {
  const a = scannedPatch([0, 0, 0], [0, 0, 1], 50, 50, 2000);
  const b = scannedPatch([0, 0, 8], tilted(0.3), 50, 50, 2000);
  const side = scannedPatch([25, 0, 4], [1, 0, 0], 50, 8, 2000);
  const [fa, fb, fs] = G.fitParallelPlanes([a, b, side]);
  assert.ok(Math.abs(fs.offBy - 90) < 1, `offBy ${fs.offBy}`);
  assert.ok(fs.tilt === undefined && angleDeg(fs.dir, [1, 0, 0]) < 0.2, 'keeps its own fit');
  assert.ok(G.dot(fa.dir, fb.dir) > 1 - 1e-12, 'the other two are still parallel');
  const [only] = G.fitParallelPlanes([scannedPatch([0, 0, 0], [0, 0, 1], 5, 5, 2)]);
  assert.ok(only.error);
});

test('solveOrigin: plane ∩ axis, plane + point, three planes', () => {
  const plane = { type: 'plane', point: [5, 5, 2], dir: [0, 0, 1] };
  const hole = { type: 'cylinder', point: [10, -3, 7], dir: G.normalize([0.1, 0, 1]) };
  const o1 = G.solveOrigin([plane, hole]);
  assert.ok(Math.abs(o1[2] - 2) < 1e-9);
  assert.ok(G.len(G.cross(G.sub(o1, hole.point), hole.dir)) < 1e-9, 'on the hole axis');

  const o2 = G.solveOrigin([plane, { type: 'point', point: [1, 2, 9] }]);
  assert.ok(dist(o2, [1, 2, 2]) < 1e-9);

  const corner = G.solveOrigin([
    { type: 'plane', point: [0, 0, 3], dir: [0, 0, -1] },
    { type: 'plane', point: [7, 0, 0], dir: [1, 0, 0] },
    { type: 'plane', point: [0, -4, 0], dir: [0, -1, 0] },
  ]);
  assert.ok(dist(corner, [7, -4, 3]) < 1e-9);
});

test('end to end: scanned plate with two holes aligns to the datum frame', () => {
  // Design frame: bottom face z=0 (outward normal -Z), hole A at (10, 5),
  // hole B at (40, 5), plate 50x30x6. The scan is in a random pose.
  const pose = randomPose();
  const bottom = [], bottomN = [];
  for (let i = 0; i < 3000; i++) {
    bottom.push(pose.p(noisy([50 * R(), 30 * R(), 0], 0.03)));
    bottomN.push(pose.d([0, 0, -1]));
  }
  const rim = (cx, cy) => {
    const P = [], N = [];
    for (let i = 0; i < 12; i++) {
      const a = (2 * Math.PI * i) / 12 + 0.1 * R();
      P.push(pose.p(noisy([cx + 2.5 * Math.cos(a), cy + 2.5 * Math.sin(a), 6], 0.03)));
      N.push(pose.d([0, 0, 1]));
    }
    return [pack(P), pack(N)];
  };
  const fits = new Map([
    [1, G.fitFeature('plane', pack(bottom), pack(bottomN))],
    [2, G.fitFeature('circle', ...rim(10, 5))],
    [3, G.fitFeature('circle', ...rim(40, 5))],
  ]);
  const centroid = pose.p([25, 15, 3]);
  const cfg = {
    primary: { id: 1, target: '-Z' },
    secondary: { id: 3, use: 'auto', target: '+X' },
    angle: 0,
    origin: [1, 2, null],
    offset: [0, 0, 0],
  };
  const res = G.computeAlignment(fits, cfg, centroid);
  assert.equal(res.secondaryMode, 'toward');
  // Design point (x, y, z) should land at (x - 10, y - 5, z).
  for (const q of [[0, 0, 0], [50, 30, 6], [10, 5, 0], [40, 5, 6], [25, 0, 3]]) {
    const got = G.transformPoint(res.matrix, pose.p(q));
    const want = [q[0] - 10, q[1] - 5, q[2]];
    assert.ok(dist(got, want) < 0.08, `${q} -> ${got.map((x) => x.toFixed(3))}`);
  }

  // Rotate the part 90 degrees about Z and shift it.
  const res2 = G.computeAlignment(fits, { ...cfg, angle: 90, offset: [1, 2, 3] }, centroid);
  const got = G.transformPoint(res2.matrix, pose.p([40, 5, 0]));
  assert.ok(dist(got, [1, 32, 3]) < 0.08, `${got}`);

  // Make the face look up instead: Z flips, X kept, so Y flips (right-handed).
  const res3 = G.computeAlignment(fits, { ...cfg, primary: { id: 1, target: '+Z' } }, centroid);
  const g3 = G.transformPoint(res3.matrix, pose.p([40, 25, 6]));
  assert.ok(dist(g3, [30, -20, -6]) < 0.08, `${g3}`);
});

test('alignment with no features is the identity about the mesh centroid', () => {
  const res = G.computeAlignment(new Map(), {
    primary: { id: null, target: '+Z' },
    secondary: { id: null, use: 'auto', target: '+X' },
    angle: 0, origin: [null, null, null], offset: [0, 0, 0],
  }, [1, 2, 3]);
  assert.deepEqual(G.transformPoint(res.matrix, [1, 2, 3]), [0, 0, 0]);
  assert.deepEqual(G.transformPoint(res.matrix, [2, 2, 3]), [1, 0, 0]);
});

// A bent sheet: flat for x <= 0, tilted 45 degrees for x > 0, with a square
// hole in the flat part. Built as triangle soup like an STL.
function bentSheet() {
  const nx = 20, ny = 10;
  const vert = (i, j) => {
    const x = i - 10, y = j;
    return x <= 0 ? [x, y, 0] : [x * Math.SQRT1_2, y, x * Math.SQRT1_2];
  };
  const soup = [];
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      if (i >= 3 && i <= 4 && j >= 4 && j <= 5) continue; // hole
      const a = vert(i, j), b = vert(i + 1, j), c = vert(i + 1, j + 1), d = vert(i, j + 1);
      soup.push(...a, ...b, ...c, ...a, ...c, ...d);
    }
  }
  return new Float32Array(soup);
}

function meshFrom(soup) {
  const { positions, index } = G.mergeVertices(soup, null);
  const vc = positions.length / 3;
  const nrm = new Float32Array(vc * 3);
  for (let t = 0; t < index.length; t += 3) {
    const [a, b, c] = [index[t], index[t + 1], index[t + 2]].map((k) => [positions[3 * k], positions[3 * k + 1], positions[3 * k + 2]]);
    const n = G.cross(G.sub(b, a), G.sub(c, a));
    for (const k of [index[t], index[t + 1], index[t + 2]]) for (let j = 0; j < 3; j++) nrm[3 * k + j] += n[j];
  }
  for (let v = 0; v < vc; v++) {
    const n = G.normalize([nrm[3 * v], nrm[3 * v + 1], nrm[3 * v + 2]]);
    nrm.set(n, 3 * v);
  }
  return { pos: positions, nrm, index, vt: G.vertexTriangles(index, vc) };
}

test('mergeVertices welds triangle soup', () => {
  const soup = bentSheet();
  const { positions, index } = G.mergeVertices(soup, null);
  assert.equal(positions.length / 3, 21 * 11 - 1); // the hole removes one grid vertex
  for (let k = 0; k < index.length; k++) {
    for (let j = 0; j < 3; j++) assert.equal(positions[3 * index[k] + j], soup[3 * k + j]);
  }
});

test('growRegion stops at the bend in seed mode', () => {
  const mesh = meshFrom(bentSheet());
  const seed = [0, 1, 2].map((j) => mesh.index[j]); // first triangle, in the flat part
  const sel = G.growRegion(mesh, seed, [-10, 0, 0], { radius: 0, maxAngle: 20, angleRef: 'seed', seedNormal: [0, 0, 1] });
  const xs = Array.from(sel, (v) => mesh.pos[3 * v]);
  assert.ok(Math.max(...xs) <= 0, 'nothing past the bend');
  const flat = Array.from(mesh.pos.filter((_, k) => k % 3 === 0)).filter((x) => x < -0.5).length;
  assert.equal(xs.filter((x) => x < -0.5).length, flat, 'every vertex of the flat part');
  const limited = G.growRegion(mesh, seed, [-10, 0, 0], { radius: 2.5, maxAngle: 20, angleRef: 'seed', seedNormal: [0, 0, 1] });
  assert.ok(limited.length < sel.length);
});

test('boundary loop around the hole', () => {
  const mesh = meshFrom(bentSheet());
  const graph = G.boundaryGraph(mesh);
  const loop = G.boundaryLoopNear(mesh, graph, [-6, 5, 0], 4, 2);
  assert.equal(loop.length, 8, 'hole perimeter is 2x2 cells -> 8 vertices');
  assert.equal(G.boundaryLoopNear(mesh, graph, [-6, 5, 20], 4, 2), null);
});

test('writeBinarySTL round-trips through three.js STLLoader', () => {
  const mesh = meshFrom(bentSheet());
  const M = [0, -1, 0, 5, 1, 0, 0, 0, 0, 0, 1, -2, 0, 0, 0, 1]; // 90 deg about Z, then shift
  const buf = G.writeBinarySTL(mesh.pos, mesh.index, M);
  const geom = new STLLoader().parse(buf);
  const p = geom.attributes.position.array;
  assert.equal(p.length / 9, mesh.index.length / 3);
  const v0 = [mesh.pos[3 * mesh.index[0]], mesh.pos[3 * mesh.index[0] + 1], mesh.pos[3 * mesh.index[0] + 2]];
  assert.ok(dist([p[0], p[1], p[2]], G.transformPoint(M, v0)) < 1e-5);
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as G from '../src/geom.js';

// Height field z = f(x, y) over a 100 x 100 square, n x n cells. `skip(i, j)`
// leaves a cell out (to make holes).
function plate(n, f = () => 0, skip = () => false) {
  const pos = new Float32Array((n + 1) * (n + 1) * 3);
  const id = (i, j) => i * (n + 1) + j;
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= n; j++) {
      const x = (100 * i) / n - 50, y = (100 * j) / n - 50;
      pos.set([x, y, f(x, y)], 3 * id(i, j));
    }
  }
  const index = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (skip(i, j)) continue;
      index.push(id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1));
    }
  }
  return { pos, index: Uint32Array.from(index) };
}

// Icosphere: closed, evenly tessellated.
function icosphere(r, levels) {
  const t = (1 + Math.sqrt(5)) / 2;
  const V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map(G.normalize);
  let F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
    [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    .map((f) => f.map((i) => V[i]));
  for (let l = 0; l < levels; l++) {
    const mid = (a, b) => G.normalize(G.add(a, b));
    F = F.flatMap(([a, b, c]) => {
      const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
    });
  }
  const soup = new Float32Array(F.flat(2).map((x) => x * r));
  const { positions, index } = G.mergeVertices(soup, null);
  return { pos: positions, index };
}

const tri = (pos, index, t) => [0, 1, 2].map((k) => Array.from(pos.subarray(3 * index[3 * t + k], 3 * index[3 * t + k] + 3)));
const area = (pos, index) => {
  let s = 0;
  for (let t = 0; t < index.length / 3; t++) {
    const [a, b, c] = tri(pos, index, t);
    s += G.len(G.cross(G.sub(b, a), G.sub(c, a))) / 2;
  }
  return s;
};
const volume = (pos, index) => {
  let s = 0;
  for (let t = 0; t < index.length / 3; t++) {
    const [a, b, c] = tri(pos, index, t);
    s += G.dot(a, G.cross(b, c)) / 6;
  }
  return s;
};
// Every directed edge used once, and its twin present: closed and manifold.
function isClosed(index) {
  const edges = new Map();
  for (let t = 0; t < index.length; t += 3) {
    for (let j = 0; j < 3; j++) {
      const k = `${index[t + j]},${index[t + (j + 1) % 3]}`;
      if (edges.has(k)) return false;
      edges.set(k, true);
    }
  }
  for (const k of edges.keys()) {
    const [a, b] = k.split(',');
    if (!edges.has(`${b},${a}`)) return false;
  }
  return true;
}
// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5).
function closestOnTriangle(p, a, b, c) {
  const ab = G.sub(b, a), ac = G.sub(c, a), ap = G.sub(p, a);
  const d1 = G.dot(ab, ap), d2 = G.dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const bp = G.sub(p, b), d3 = G.dot(ab, bp), d4 = G.dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return G.add(a, G.scale(ab, d1 / (d1 - d3)));
  const cp = G.sub(p, c), d5 = G.dot(ab, cp), d6 = G.dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return G.add(a, G.scale(ac, d2 / (d2 - d6)));
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return G.add(b, G.scale(G.sub(c, b), (d4 - d3) / (d4 - d3 + d5 - d6)));
  const den = 1 / (va + vb + vc);
  return G.add(a, G.add(G.scale(ab, vb * den), G.scale(ac, vc * den)));
}
// Largest distance from an original vertex to the simplified surface.
function maxDeviation(pos, origIndex, index) {
  const used = new Set(origIndex);
  const tris = Array.from({ length: index.length / 3 }, (_, t) => tri(pos, index, t));
  let worst = 0;
  for (const v of used) {
    const p = Array.from(pos.subarray(3 * v, 3 * v + 3));
    let best = Infinity;
    for (const [a, b, c] of tris) best = Math.min(best, G.len(G.sub(p, closestOnTriangle(p, a, b, c))));
    worst = Math.max(worst, best);
  }
  return worst;
}

const noise = (() => { let s = 7; return () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5; })();
const bump = (x, y) => 2 * Math.exp(-(x * x + y * y) / 50);

test('simplify: a flat plate collapses to two triangles', () => {
  const m = plate(40);
  const r = G.simplifyMesh(m.pos, m.index, { maxDeviation: 1e-6 });
  assert.equal(r.index.length / 3, 2);
  assert.ok(Math.abs(area(m.pos, r.index) - 10000) < 1e-3);
});

test('simplify: open borders keep their outline (plate with a square hole)', () => {
  const m = plate(40, () => 0, (i, j) => i >= 15 && i < 25 && j >= 10 && j < 20);
  const r = G.simplifyMesh(m.pos, m.index, { maxDeviation: 1e-6 });
  assert.ok(r.index.length / 3 <= 10, `${r.index.length / 3} triangles`);
  assert.ok(Math.abs(area(m.pos, r.index) - 9375) < 1e-3, 'outer edge and hole stay put');
});

test('simplify: closed sphere to a triangle target stays closed and round', () => {
  const m = icosphere(20, 4);
  const r = G.simplifyMesh(m.pos, m.index, { targetTriangles: 512 });
  const n = r.index.length / 3;
  assert.ok(n <= 512 && n > 400, `${n} triangles`);
  assert.ok(isClosed(r.index), 'still closed and manifold');
  const dv = Math.abs(volume(m.pos, r.index) / volume(m.pos, m.index) - 1);
  assert.ok(dv < 0.05, `volume changed ${(100 * dv).toFixed(1)}%`);
  assert.ok(maxDeviation(m.pos, m.index, r.index) < 0.6);
});

test('simplify: tolerance trades triangles for accuracy on curved detail', () => {
  const m = plate(60, bump);
  const runs = [0.02, 0.05].map((tol) => {
    const r = G.simplifyMesh(m.pos, m.index, { maxDeviation: tol });
    return { tol, n: r.index.length / 3, dev: maxDeviation(m.pos, m.index, r.index) };
  });
  for (const { tol, n, dev } of runs) {
    assert.ok(n < 0.1 * 7200, `tol ${tol}: ${n} triangles`);
    assert.ok(dev < 2 * tol, `tol ${tol}: deviation ${dev}`);
  }
  assert.ok(runs[0].n > runs[1].n && runs[0].dev < runs[1].dev);
});

test('simplify: scan noise on a flat face merges into large triangles', () => {
  const m = plate(60, () => 0.01 * noise());
  const r = G.simplifyMesh(m.pos, m.index, { maxDeviation: 0.02 });
  assert.ok(r.index.length / 3 < 0.1 * 7200, `${r.index.length / 3} triangles`);
  assert.ok(maxDeviation(m.pos, m.index, r.index) < 0.02);
});

test('simplify: non-manifold edges stay put, degenerate triangles are dropped', () => {
  // A plate with a fin standing on its middle row: the junction edges have
  // three triangles each.
  const n = 20;
  const m = plate(n);
  const pos = Array.from(m.pos), index = Array.from(m.index);
  const mid = n / 2, base = pos.length / 3;
  for (let j = 0; j <= n; j++) pos.push(0, (100 * j) / n - 50, 10);
  for (let j = 0; j < n; j++) {
    const a = mid * (n + 1) + j, b = a + 1;
    index.push(a, b, base + j + 1, a, base + j + 1, base + j);
  }
  index.push(0, 0, 1); // degenerate
  const P = Float32Array.from(pos), I = Uint32Array.from(index);
  const r = G.simplifyMesh(P, I, { maxDeviation: 1e-6 });
  const used = new Set(r.index);
  for (let j = 1; j < n; j++) assert.ok(used.has(mid * (n + 1) + j), `junction vertex ${j} kept`);
  assert.ok(Math.abs(area(P, r.index) - (10000 + 1000)) < 1e-3, 'plate and fin keep their area');
  assert.ok(r.index.length / 3 < 0.25 * (I.length / 3), `${r.index.length / 3} triangles`);
});

test('simplify: nothing to do when the target is already met', () => {
  const m = icosphere(5, 2);
  const r = G.simplifyMesh(m.pos, m.index, { targetTriangles: m.index.length / 3 });
  assert.deepEqual(Array.from(r.index), Array.from(m.index));
  assert.equal(r.error, 0);
});

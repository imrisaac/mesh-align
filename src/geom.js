// Geometry math for Mesh Align: feature fitting (plane, cylinder, circle, line,
// point), datum/alignment construction, mesh topology helpers, mesh
// simplification and STL export.
// Pure JS with no three.js dependency so it can be unit-tested in Node.
//
// Conventions: vectors are plain [x, y, z] arrays; point sets are flat
// Float64Array/Float32Array buffers [x0, y0, z0, x1, ...]; matrices are
// row-major.

// ---------------------------------------------------------------------------
// Vector / matrix helpers
// ---------------------------------------------------------------------------

export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const len = (a) => Math.hypot(a[0], a[1], a[2]);
export function normalize(a) {
  const l = len(a);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
}

const BASIS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

// Two unit vectors u, v such that (u, v, n) is a right-handed orthonormal frame.
export function perpBasis(n) {
  const helper = Math.abs(n[0]) < 0.9 ? BASIS[0] : BASIS[1];
  const u = normalize(cross(n, helper));
  return [u, cross(n, u)];
}

const matVec = (M, v) => [dot(M[0], v), dot(M[1], v), dot(M[2], v)];

function matMul3(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
}

// Eigen-decomposition of a symmetric 3x3 matrix (cyclic Jacobi).
// Returns eigenvalues ascending with matching unit eigenvectors.
export function eigSym3(M) {
  const a = [M[0].slice(), M[1].slice(), M[2].slice()];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 64; sweep++) {
    const off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2;
    const diag = a[0][0] ** 2 + a[1][1] ** 2 + a[2][2] ** 2;
    if (off === 0 || off <= 1e-30 * diag) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[p][q];
      if (apq === 0) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      // a <- P^T a P, V <- V P   with P[p][p]=P[q][q]=c, P[p][q]=s, P[q][p]=-s
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return {
    values: order.map((i) => a[i][i]),
    vectors: order.map((i) => normalize([V[0][i], V[1][i], V[2][i]])),
  };
}

// Solve a small dense linear system (Gaussian elimination, partial pivoting).
// A is an array of rows; returns null when singular.
export function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  let maxAbs = 0;
  for (const row of A) for (const x of row) maxAbs = Math.max(maxAbs, Math.abs(x));
  if (maxAbs === 0) return null;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-14 * maxAbs) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / M[col][col];
      for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// Levenberg-Marquardt with a forward-difference Jacobian.
// fn(params, out) writes m residuals into `out`. `scales` gives the typical
// magnitude of each parameter, used to size the finite-difference steps.
export function levenbergMarquardt(p0, m, fn, scales, maxIter = 60) {
  const n = p0.length;
  const p = Float64Array.from(p0);
  const r = new Float64Array(m);
  const rt = new Float64Array(m);
  const pt = new Float64Array(n);
  const J = new Float64Array(m * n);
  const sumSq = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return s; };

  fn(p, r);
  let cost = sumSq(r);
  let lambda = 1e-3;
  for (let it = 0; it < maxIter && cost > 0; it++) {
    for (let j = 0; j < n; j++) {
      const h = 1e-7 * (scales[j] || 1) + 1e-9 * Math.abs(p[j]);
      pt.set(p);
      pt[j] += h;
      fn(pt, rt);
      for (let i = 0; i < m; i++) J[i * n + j] = (rt[i] - r[i]) / h;
    }
    const JTJ = Array.from({ length: n }, () => new Array(n).fill(0));
    const g = new Array(n).fill(0);
    for (let i = 0; i < m; i++) {
      const row = i * n;
      for (let a = 0; a < n; a++) {
        const ja = J[row + a];
        g[a] += ja * r[i];
        for (let b = a; b < n; b++) JTJ[a][b] += ja * J[row + b];
      }
    }
    for (let a = 0; a < n; a++) for (let b = 0; b < a; b++) JTJ[a][b] = JTJ[b][a];

    let accepted = false;
    let converged = false;
    for (let k = 0; k < 12; k++) {
      const A = JTJ.map((row, a) => row.map((x, b) => (a === b ? x * (1 + lambda) + 1e-15 : x)));
      const delta = solveLinear(A, g.map((x) => -x));
      if (!delta) { lambda *= 10; continue; }
      for (let j = 0; j < n; j++) pt[j] = p[j] + delta[j];
      fn(pt, rt);
      const c = sumSq(rt);
      if (c < cost) {
        converged = (cost - c) <= 1e-12 * cost ||
          delta.every((d, j) => Math.abs(d) <= 1e-10 * (scales[j] || 1));
        p.set(pt);
        r.set(rt);
        cost = c;
        lambda = Math.max(lambda * 0.3, 1e-12);
        accepted = true;
        break;
      }
      lambda *= 10;
    }
    if (!accepted || converged) break;
  }
  return p;
}

// ---------------------------------------------------------------------------
// Point-set statistics
// ---------------------------------------------------------------------------

function iota(n) {
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  return a;
}

// Evenly strided subset of `idx` with at most `max` entries.
function subsample(idx, max) {
  if (idx.length <= max) return idx;
  const out = new Int32Array(max);
  const step = idx.length / max;
  for (let k = 0; k < max; k++) out[k] = idx[Math.floor(k * step)];
  return out;
}

function meanCov(P, idx) {
  const n = idx.length;
  let cx = 0, cy = 0, cz = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 3;
    cx += P[i]; cy += P[i + 1]; cz += P[i + 2];
  }
  cx /= n; cy /= n; cz /= n;
  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k] * 3;
    const x = P[i] - cx, y = P[i + 1] - cy, z = P[i + 2] - cz;
    xx += x * x; xy += x * y; xz += x * z; yy += y * y; yz += y * z; zz += z * z;
  }
  return {
    mean: [cx, cy, cz],
    cov: [[xx / n, xy / n, xz / n], [xy / n, yy / n, yz / n], [xz / n, yz / n, zz / n]],
  };
}

function sumVectors(N, idx) {
  const s = [0, 0, 0];
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k] * 3;
    s[0] += N[i]; s[1] += N[i + 1]; s[2] += N[i + 2];
  }
  return s;
}

const pointAt = (P, i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];

function median(values) {
  const a = Float64Array.from(values).sort();
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

// Fit, then repeatedly drop points farther than 3 robust sigmas (MAD) and
// refit. Adds rms / maxDev / used / total / inliers to the returned fit.
function robustFit(P, N, minPts, fitOnce, residual) {
  const n = P.length / 3;
  let idx = iota(n);
  let fit = fitOnce(P, N, idx, null);
  if (!fit) return null;
  if (n >= 10) {
    const r = new Float64Array(n);
    for (let iter = 0; iter < 4; iter++) {
      for (let i = 0; i < n; i++) r[i] = Math.abs(residual(fit, P, i));
      const med = median(Array.from(idx, (i) => r[i]));
      const thr = Math.max(3 * 1.4826 * med, 1e-9);
      const next = [];
      for (let i = 0; i < n; i++) if (r[i] <= thr) next.push(i);
      if (next.length < minPts || next.length === idx.length) break;
      const nextIdx = Int32Array.from(next);
      const refit = fitOnce(P, N, nextIdx, fit);
      if (!refit) break;
      idx = nextIdx;
      fit = refit;
    }
  }
  let ss = 0, maxDev = 0;
  for (let k = 0; k < idx.length; k++) {
    const d = residual(fit, P, idx[k]);
    ss += d * d;
    maxDev = Math.max(maxDev, Math.abs(d));
  }
  fit.rms = Math.sqrt(ss / idx.length);
  fit.maxDev = maxDev;
  fit.used = idx.length;
  fit.total = n;
  fit.inliers = idx;
  return fit;
}

// ---------------------------------------------------------------------------
// Feature fits
// ---------------------------------------------------------------------------

function fitPlaneOnce(P, N, idx) {
  if (idx.length < 3) return null;
  const { mean, cov } = meanCov(P, idx);
  const { values, vectors } = eigSym3(cov);
  if (values[1] <= 1e-12 * Math.max(values[2], 1e-300)) return null; // collinear
  let n = vectors[0];
  if (N && dot(n, sumVectors(N, idx)) < 0) n = scale(n, -1);
  const u = vectors[2];
  return { type: 'plane', point: mean, dir: n, u, v: cross(n, u) };
}
const planeResidual = (f, P, i) => dot(sub(pointAt(P, i), f.point), f.dir);

function fitLineOnce(P, N, idx) {
  if (idx.length < 2) return null;
  const { mean, cov } = meanCov(P, idx);
  const { values, vectors } = eigSym3(cov);
  if (values[2] <= 0) return null;
  return { type: 'line', point: mean, dir: vectors[2] };
}
const lineResidual = (f, P, i) => len(cross(sub(pointAt(P, i), f.point), f.dir));

// Circle through 2D points: algebraic (Kasa) fit refined geometrically.
export function fitCircle2D(xs, ys) {
  const n = xs.length;
  if (n < 3) return null;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  const X = new Float64Array(n), Y = new Float64Array(n);
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const x = xs[i] - mx, y = ys[i] - my, z = x * x + y * y;
    X[i] = x; Y[i] = y;
    const row = [x, y, 1];
    for (let a = 0; a < 3; a++) {
      b[a] -= row[a] * z;
      for (let c = 0; c < 3; c++) A[a][c] += row[a] * row[c];
    }
  }
  const sol = solveLinear(A, b);
  if (!sol) return null;
  const cx = -sol[0] / 2, cy = -sol[1] / 2;
  const r2 = cx * cx + cy * cy - sol[2];
  if (!(r2 > 0)) return null;
  const r0 = Math.sqrt(r2);
  const p = levenbergMarquardt([cx, cy, r0], n, (q, out) => {
    for (let i = 0; i < n; i++) out[i] = Math.hypot(X[i] - q[0], Y[i] - q[1]) - q[2];
  }, [r0, r0, r0]);
  return { cx: p[0] + mx, cy: p[1] + my, r: Math.abs(p[2]) };
}

function fitCircleOnce(P, N, idx) {
  if (idx.length < 3) return null;
  const { mean, cov } = meanCov(P, idx);
  const { values, vectors } = eigSym3(cov);
  if (values[1] <= 1e-12 * Math.max(values[2], 1e-300)) return null;
  let n = vectors[0];
  if (N && dot(n, sumVectors(N, idx)) < 0) n = scale(n, -1);
  const u = vectors[2], v = cross(n, u);
  const s = subsample(idx, 20000);
  const xs = new Float64Array(s.length), ys = new Float64Array(s.length);
  for (let k = 0; k < s.length; k++) {
    const d = sub(pointAt(P, s[k]), mean);
    xs[k] = dot(d, u);
    ys[k] = dot(d, v);
  }
  const c = fitCircle2D(xs, ys);
  if (!c) return null;
  const center = add(mean, add(scale(u, c.cx), scale(v, c.cy)));
  return { type: 'circle', point: center, dir: n, radius: c.r, u, v };
}
function circleResidual(f, P, i) {
  const d = sub(pointAt(P, i), f.point);
  const h = dot(d, f.dir);
  const radial = len(sub(d, scale(f.dir, h))) - f.radius;
  return Math.hypot(h, radial);
}

// Cylinder whose axis starts near direction a0. Works in a local frame where
// a0 is +z, fitting [cx, cy, tiltX, tiltY, R].
function fitCylinderFromAxis(P, idx, a0) {
  const { mean } = meanCov(P, idx);
  const [u, v] = perpBasis(a0);
  const m = idx.length;
  const qx = new Float64Array(m), qy = new Float64Array(m), qz = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const d = sub(pointAt(P, idx[k]), mean);
    qx[k] = dot(d, u); qy[k] = dot(d, v); qz[k] = dot(d, a0);
  }
  const c = fitCircle2D(qx, qy);
  if (!c) return null;
  const p = levenbergMarquardt([c.cx, c.cy, 0, 0, c.r], m, (q, out) => {
    const L = Math.hypot(q[2], q[3], 1);
    const dx = q[2] / L, dy = q[3] / L, dz = 1 / L;
    for (let i = 0; i < m; i++) {
      const wx = qx[i] - q[0], wy = qy[i] - q[1], wz = qz[i];
      const cx = wy * dz - wz * dy, cy = wz * dx - wx * dz, cz = wx * dy - wy * dx;
      out[i] = Math.sqrt(cx * cx + cy * cy + cz * cz) - q[4];
    }
  }, [c.r, c.r, 1, 1, c.r]);
  const dir = normalize(add(a0, add(scale(u, p[2]), scale(v, p[3]))));
  const base = add(mean, add(scale(u, p[0]), scale(v, p[1])));
  // Re-anchor the axis point at the foot of the centroid.
  const point = add(base, scale(dir, dot(sub(mean, base), dir)));
  return { type: 'cylinder', point, dir, radius: Math.abs(p[4]) };
}
const cylinderResidual = (f, P, i) => len(cross(sub(pointAt(P, i), f.point), f.dir)) - f.radius;

function fitCylinderOnce(P, N, idx, prev) {
  if (idx.length < 6) return null;
  const s = subsample(idx, 20000);
  // Candidate axes: the previous fit when refining; otherwise the direction
  // most perpendicular to the surface normals (best), plus the point-cloud
  // principal axes as fallbacks.
  const cands = [];
  if (prev) cands.push({ axis: prev.dir, bias: 1 });
  else {
    if (N) cands.push({ axis: eigSym3(meanCovOfVectors(N, s)).vectors[0], bias: 0.7 });
    const { vectors } = eigSym3(meanCov(P, s).cov);
    for (const a of [vectors[0], vectors[2], vectors[1]]) cands.push({ axis: a, bias: 1 });
  }
  let best = null, bestScore = Infinity;
  for (const { axis, bias } of cands) {
    const f = fitCylinderFromAxis(P, s, axis);
    if (!f) continue;
    let ss = 0;
    for (let k = 0; k < s.length; k++) ss += cylinderResidual(f, P, s[k]) ** 2;
    const score = Math.sqrt(ss / s.length) * bias;
    if (score < bestScore) { bestScore = score; best = f; }
  }
  return best;
}

// Second-moment matrix of unit vectors (not mean-centred).
function meanCovOfVectors(N, idx) {
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k] * 3;
    const n = [N[i], N[i + 1], N[i + 2]];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) M[a][b] += n[a] * n[b];
  }
  return M;
}

const MIN_POINTS = { plane: 3, cylinder: 6, circle: 3, line: 2, point: 1 };

// Fit a feature of `type` to points P (and optional unit normals N, used to
// orient planes/circles outward and to seed cylinder axes).
// Returns a fit object, or { error } when the points don't define the feature.
export function fitFeature(type, P, N) {
  const n = P.length / 3;
  const need = MIN_POINTS[type];
  if (n < need) return { error: `needs at least ${need} point${need > 1 ? 's' : ''}` };
  let fit;
  switch (type) {
    case 'plane':
      fit = robustFit(P, N, 3, fitPlaneOnce, planeResidual);
      if (fit) fit = planeResult(P, fit, fit.inliers);
      break;
    case 'circle':
      fit = robustFit(P, N, 3, fitCircleOnce, circleResidual);
      break;
    case 'cylinder':
      fit = robustFit(P, N, 6, fitCylinderOnce, cylinderResidual);
      if (fit) {
        let lo = Infinity, hi = -Infinity, outward = 0;
        for (const i of fit.inliers) {
          const d = sub(pointAt(P, i), fit.point);
          const t = dot(d, fit.dir);
          lo = Math.min(lo, t); hi = Math.max(hi, t);
          if (N) outward += dot(normalize(sub(d, scale(fit.dir, t))), [N[3 * i], N[3 * i + 1], N[3 * i + 2]]);
        }
        fit.point = add(fit.point, scale(fit.dir, (lo + hi) / 2));
        fit.height = hi - lo;
        // Surface normals of a hole wall point toward the axis; a boss's point away.
        fit.kind = N ? (outward < 0 ? 'hole' : 'boss') : null;
      }
      break;
    case 'line':
      fit = robustFit(P, N, 2, fitLineOnce, lineResidual);
      if (fit) {
        let lo = Infinity, hi = -Infinity;
        for (const i of fit.inliers) {
          const t = dot(sub(pointAt(P, i), fit.point), fit.dir);
          lo = Math.min(lo, t); hi = Math.max(hi, t);
        }
        fit.range = [lo, hi];
      }
      break;
    case 'point': {
      const idx = iota(n);
      const { mean } = meanCov(P, idx);
      let ss = 0;
      for (let i = 0; i < n; i++) ss += len(sub(pointAt(P, i), mean)) ** 2;
      fit = { type: 'point', point: mean, dir: null, rms: Math.sqrt(ss / n), used: n, total: n };
      break;
    }
    default:
      return { error: `unknown feature type ${type}` };
  }
  if (!fit) {
    return { error: type === 'cylinder' ? 'could not fit a cylinder to these points'
      : 'points are degenerate (all in a line?)' };
  }
  delete fit.inliers;
  return fit;
}

// A finished plane fit: residual stats, flatness (peak to valley) and the
// in-plane extent of the inlier points `idx`.
function planeResult(P, fit, idx, extra = {}) {
  let ss = 0, maxDev = 0, lo = Infinity, hi = -Infinity;
  const ext = [Infinity, -Infinity, Infinity, -Infinity];
  for (const i of idx) {
    const d = sub(pointAt(P, i), fit.point);
    const h = dot(d, fit.dir), a = dot(d, fit.u), b = dot(d, fit.v);
    ss += h * h;
    maxDev = Math.max(maxDev, Math.abs(h));
    lo = Math.min(lo, h); hi = Math.max(hi, h);
    ext[0] = Math.min(ext[0], a); ext[1] = Math.max(ext[1], a);
    ext[2] = Math.min(ext[2], b); ext[3] = Math.max(ext[3], b);
  }
  return {
    type: 'plane', point: fit.point, dir: fit.dir, u: fit.u, v: fit.v,
    rms: Math.sqrt(ss / idx.length), maxDev, used: idx.length, total: P.length / 3,
    flatness: hi - lo, extent: ext, ...extra,
  };
}

// Planes that must be parallel (facing the same way or opposite, like the top
// and bottom of a plate): one shared normal, each plane with its own offset.
// It's a least-squares fit over all their points together, with the same
// outlier rejection as a single plane, so planes with more points and a wider
// spread steer the normal more.
//
// sets: [{ P, N }], the points (and surface normals) of each plane.
// Returns one entry per set, in order:
//   - a plane fit like fitFeature's, its normal along the shared axis (signed
//     to match that plane's surface), plus `tilt`: how many degrees its own
//     best fit was off the shared normal;
//   - for a plane more than maxTilt degrees off the rest: its own fit, with
//     `offBy` (degrees) and no constraint applied;
//   - { error } if it can't be fitted at all.
export function fitParallelPlanes(sets, maxTilt = 10) {
  const own = sets.map(({ P, N }) => (P.length >= 9 ? robustFit(P, N, 3, fitPlaneOnce, planeResidual) : null));
  const out = own.map((f) => (f ? null : { error: 'needs at least 3 points' }));
  const cosMax = Math.cos((maxTilt * Math.PI) / 180);
  const near = (a, b) => Math.abs(dot(a.dir, b.dir)) >= cosMax;

  // The direction most of the planes agree on (ties: the one with most points).
  let ref = null, best = -1;
  for (const f of own) {
    if (!f) continue;
    const votes = own.filter((g) => g && near(f, g)).length;
    if (votes > best || (votes === best && f.used > ref.used)) { ref = f; best = votes; }
  }
  if (!ref) return out;
  const members = [];
  own.forEach((f, i) => {
    if (!f) return;
    if (near(f, ref)) members.push(i);
    else out[i] = planeResult(sets[i].P, f, f.inliers, { offBy: (Math.acos(Math.abs(dot(f.dir, ref.dir))) * 180) / Math.PI });
  });
  if (members.length < 2) {
    for (const i of members) out[i] = planeResult(sets[i].P, own[i], own[i].inliers);
    return out;
  }

  // Shared normal: smallest eigenvector of the pooled scatter, each plane's
  // points taken about their own centroid. Then re-pick every plane's inliers
  // against it (3 robust sigmas, as in robustFit) and repeat.
  let idx = members.map((i) => own[i].inliers);
  let n, means;
  for (let iter = 0; ; iter++) {
    const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    means = members.map((i, k) => {
      const { mean, cov } = meanCov(sets[i].P, idx[k]);
      const m = idx[k].length;
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += cov[r][c] * m;
      return mean;
    });
    n = eigSym3(S).vectors[0];
    if (iter === 4) break;
    let changed = false;
    const next = members.map((i, k) => {
      const P = sets[i].P, cnt = P.length / 3;
      const r = new Float64Array(cnt);
      for (let j = 0; j < cnt; j++) r[j] = Math.abs(dot(sub(pointAt(P, j), means[k]), n));
      const thr = Math.max(3 * 1.4826 * median(Array.from(idx[k], (j) => r[j])), 1e-9);
      const keep = [];
      for (let j = 0; j < cnt; j++) if (r[j] <= thr) keep.push(j);
      if (keep.length < 3 || keep.length === idx[k].length) return idx[k];
      changed = true;
      return Int32Array.from(keep);
    });
    if (!changed) break;
    idx = next;
  }

  members.forEach((i, k) => {
    const f = own[i];
    const dir = dot(n, f.dir) < 0 ? scale(n, -1) : n;
    const u = normalize(sub(f.u, scale(dir, dot(f.u, dir))));
    const tilt = (Math.acos(Math.min(1, Math.abs(dot(f.dir, n)))) * 180) / Math.PI;
    out[i] = planeResult(sets[i].P, { point: means[k], dir, u, v: cross(dir, u) }, idx[k], { tilt });
  });
  return out;
}

// ---------------------------------------------------------------------------
// Alignment (datum reference frame)
// ---------------------------------------------------------------------------

export function parseTarget(t) {
  return { axis: 'XYZ'.indexOf(t[1]), sign: t[0] === '-' ? -1 : 1 };
}

// The direction a feature contributes. Planes and circles keep their outward
// normal. Axes have no inherent sign, so point them toward the bulk of the
// mesh (deterministic, and "the part extends along +axis").
export function naturalDirection(fit, meshCentroid) {
  if (!fit.dir) return null;
  if (fit.type === 'plane' || fit.type === 'circle') return fit.dir;
  return dot(sub(meshCentroid, fit.point), fit.dir) < 0 ? scale(fit.dir, -1) : fit.dir;
}

// Linear constraints a feature places on the origin: rows a with a·O = b.
// A circle counts as a point: the normal of a few rim picks is too noisy to
// intersect its axis with a distant plane, so after a plane it projects
// straight onto it instead.
function originConstraints(fit) {
  if (fit.type === 'plane') return [[fit.dir, dot(fit.dir, fit.point)]];
  if (fit.type === 'point' || fit.type === 'circle') return BASIS.map((a) => [a, dot(a, fit.point)]);
  const [u, v] = perpBasis(fit.dir); // cylinder / line: lie on the axis
  return [[u, dot(u, fit.point)], [v, dot(v, fit.point)]];
}

// Origin from an ordered list of features, with strict priority: each feature
// is satisfied (least-squares) only in the directions earlier features left
// free. So plane + hole axis -> their intersection, plane + point -> point
// projected onto the plane, three planes -> the corner. Directions nothing
// constrains stay at the first feature's location.
export function solveOrigin(fits) {
  if (!fits.length) return null;
  let O = fits[0].point.slice();
  let free = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; // projector onto unconstrained directions
  for (const f of fits) {
    const G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const g = [0, 0, 0];
    for (const [a, b] of originConstraints(f)) {
      const ap = matVec(free, a);
      const res = b - dot(a, O);
      for (let i = 0; i < 3; i++) {
        g[i] += ap[i] * res;
        for (let j = 0; j < 3; j++) G[i][j] += ap[i] * ap[j];
      }
    }
    const { values, vectors } = eigSym3(G);
    let delta = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      // ~1e-3 means the constraint is within ~2 degrees of being redundant
      if (values[k] <= 1e-3) continue;
      const e = vectors[k];
      delta = add(delta, scale(e, dot(e, g) / values[k]));
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) free[i][j] -= e[i] * e[j];
    }
    O = add(O, delta);
  }
  return O;
}

function axisRotation(axis, deg) {
  const t = (deg * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
  if (axis === 0) return [[1, 0, 0], [0, c, -s], [0, s, c]];
  if (axis === 1) return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
}

// Build the rigid transform that maps scan coordinates into the datum frame.
//   fits: Map(featureId -> fit)
//   cfg: {
//     primary:   { id, target: '+Z' | '-Z' | '+X' ... },  // feature direction -> this axis
//     secondary: { id, use: 'auto'|'direction'|'toward', target },  // fixes rotation about primary
//     angle,     // extra rotation (deg) of the part about the primary target axis
//     origin:    [id, id, id],  // features defining the origin, in priority order
//     offset:    [x, y, z],     // added after alignment
//   }
// Returns { matrix (row-major 4x4), R, origin, primaryDir, secondaryMode, notes }.
export function computeAlignment(fits, cfg, meshCentroid) {
  const notes = [];
  const get = (id) => (id != null && fits.get(id)) || null;

  const originFits = cfg.origin.map(get).filter(Boolean);
  let O = solveOrigin(originFits);
  if (!O) {
    O = meshCentroid.slice();
    notes.push('No origin features chosen: origin is at the mesh centroid.');
  }

  const pt = parseTarget(cfg.primary.target);
  let st = parseTarget(cfg.secondary.target);
  if (st.axis === pt.axis) st = { axis: (pt.axis + 1) % 3, sign: 1 };

  const pf = get(cfg.primary.id);
  let ep = pf && naturalDirection(pf, meshCentroid);
  if (!ep) {
    ep = scale(BASIS[pt.axis], pt.sign); // keep the scan's own axis
    if (pf) notes.push(`${pf.type} has no direction and can't be the primary.`);
  }

  const sf = get(cfg.secondary.id);
  let s = null;
  let mode = null;
  if (sf && sf !== pf) {
    const d = naturalDirection(sf, meshCentroid);
    mode = cfg.secondary.use;
    if (mode === 'auto') mode = d && Math.abs(dot(d, ep)) < 0.866 ? 'direction' : 'toward';
    if (mode === 'direction' && !d) mode = 'toward';
    s = mode === 'direction' ? d : sub(sf.point, O);
    s = sub(s, scale(ep, dot(s, ep)));
    const ref = mode === 'direction' ? 1 : Math.max(len(sub(sf.point, O)), 1e-9);
    if (len(s) < 1e-6 * ref) {
      notes.push(mode === 'direction'
        ? 'Secondary direction is parallel to the primary, so it cannot set the rotation.'
        : 'Secondary feature lies on the primary axis through the origin, so it cannot set the rotation.');
      s = null;
      mode = null;
    }
  }
  if (!s) {
    // Fall back to the scan axis matching the secondary target.
    for (let k = 0; k < 3 && !s; k++) {
      const a = BASIS[(st.axis + k) % 3];
      const q = sub(a, scale(ep, dot(a, ep)));
      if (len(q) > 0.3) s = scale(q, k === 0 ? st.sign : 1);
    }
  }
  s = normalize(s);

  let R = [null, null, null];
  R[pt.axis] = scale(ep, pt.sign);
  R[st.axis] = scale(s, st.sign);
  const k = 3 - pt.axis - st.axis;
  R[k] = cross(R[(k + 1) % 3], R[(k + 2) % 3]);
  if (cfg.angle) R = matMul3(axisRotation(pt.axis, cfg.angle), R);

  const RO = matVec(R, O);
  const off = cfg.offset || [0, 0, 0];
  const t = [off[0] - RO[0], off[1] - RO[1], off[2] - RO[2]];
  const matrix = [
    R[0][0], R[0][1], R[0][2], t[0],
    R[1][0], R[1][1], R[1][2], t[1],
    R[2][0], R[2][1], R[2][2], t[2],
    0, 0, 0, 1,
  ];
  return { matrix, R, origin: O, primaryDir: ep, secondaryMode: mode, notes };
}

export function transformPoint(M, p) {
  return [
    M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3],
    M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7],
    M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11],
  ];
}

export function transformDir(M, d) {
  return [
    M[0] * d[0] + M[1] * d[1] + M[2] * d[2],
    M[4] * d[0] + M[5] * d[1] + M[6] * d[2],
    M[8] * d[0] + M[9] * d[1] + M[10] * d[2],
  ];
}

// ---------------------------------------------------------------------------
// Mesh topology
// ---------------------------------------------------------------------------

// Weld bit-identical vertex positions (STL triangle soup -> indexed mesh).
// `index` may be null for non-indexed input.
// Returns { positions, index, remap } where remap[oldVertex] = newVertex.
export function mergeVertices(pos, index) {
  const n = pos.length / 3;
  const src = new Float32Array(pos.length);
  for (let i = 0; i < pos.length; i++) src[i] = pos[i] + 0; // folds -0 into +0
  const sb = new Uint32Array(src.buffer);
  const out = new Float32Array(pos.length);
  const ob = new Uint32Array(out.buffer);
  const remap = new Uint32Array(n);
  const hash = (x, y, z) =>
    (Math.imul(x ^ (x >>> 16), 0x9e3779b1) ^ Math.imul(y ^ (y >>> 15), 0x85ebca77) ^
      Math.imul(z ^ (z >>> 13), 0xc2b2ae3d)) >>> 0;

  let cap = 1024;
  while (cap < n / 2) cap *= 2;
  let mask = cap - 1;
  let table = new Int32Array(cap).fill(-1);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const x = sb[3 * i], y = sb[3 * i + 1], z = sb[3 * i + 2];
    let h = hash(x, y, z) & mask;
    for (;;) {
      const id = table[h];
      if (id === -1) {
        table[h] = count;
        ob[3 * count] = x; ob[3 * count + 1] = y; ob[3 * count + 2] = z;
        remap[i] = count++;
        break;
      }
      if (ob[3 * id] === x && ob[3 * id + 1] === y && ob[3 * id + 2] === z) {
        remap[i] = id;
        break;
      }
      h = (h + 1) & mask;
    }
    if (count * 2 > cap) {
      cap *= 2;
      mask = cap - 1;
      table = new Int32Array(cap).fill(-1);
      for (let id = 0; id < count; id++) {
        let hh = hash(ob[3 * id], ob[3 * id + 1], ob[3 * id + 2]) & mask;
        while (table[hh] !== -1) hh = (hh + 1) & mask;
        table[hh] = id;
      }
    }
  }
  let newIndex;
  if (index) {
    newIndex = new Uint32Array(index.length);
    for (let k = 0; k < index.length; k++) newIndex[k] = remap[index[k]];
  } else {
    newIndex = remap.slice();
  }
  return { positions: out.slice(0, count * 3), index: newIndex, remap };
}

// Vertex -> incident triangles, in CSR form: triangles of vertex v are
// list[offsets[v] .. offsets[v + 1]).
export function vertexTriangles(index, vertexCount) {
  const offsets = new Uint32Array(vertexCount + 1);
  for (let k = 0; k < index.length; k++) offsets[index[k] + 1]++;
  for (let v = 0; v < vertexCount; v++) offsets[v + 1] += offsets[v];
  const fill = offsets.slice(0, vertexCount);
  const list = new Uint32Array(index.length);
  for (let k = 0; k < index.length; k++) list[fill[index[k]]++] = (k / 3) | 0;
  return { offsets, list };
}

// Flood-fill connected vertices from `seeds`, keeping those within `radius`
// of `center` (radius <= 0: unlimited) whose normal is within `maxAngle`
// degrees of the seed normal ('seed' mode, for flat faces) or of the vertex it
// was reached from ('neighbor' mode, for curved faces).
// mesh: { pos, nrm, index, vt: { offsets, list } }.
export function growRegion(mesh, seeds, center, opts) {
  const { pos, nrm, index, vt } = mesh;
  const vc = pos.length / 3;
  if (!mesh._stamp || mesh._stamp.length !== vc) { mesh._stamp = new Uint32Array(vc); mesh._gen = 0; }
  const stamp = mesh._stamp;
  const gen = ++mesh._gen;
  const r2 = opts.radius > 0 ? opts.radius * opts.radius : Infinity;
  const cosMax = Math.cos((opts.maxAngle * Math.PI) / 180);
  const neighborMode = opts.angleRef === 'neighbor';
  const sn = normalize(opts.seedNormal);
  const maxCount = opts.maxCount || 5e6;
  const [cx, cy, cz] = center;

  const out = [];
  for (const s of seeds) {
    if (stamp[s] !== gen) { stamp[s] = gen; out.push(s); }
  }
  for (let head = 0; head < out.length && out.length < maxCount; head++) {
    const v = out[head];
    const nvx = nrm[3 * v], nvy = nrm[3 * v + 1], nvz = nrm[3 * v + 2];
    for (let k = vt.offsets[v]; k < vt.offsets[v + 1]; k++) {
      const t = vt.list[k] * 3;
      for (let j = 0; j < 3; j++) {
        const w = index[t + j];
        if (stamp[w] === gen) continue;
        const dx = pos[3 * w] - cx, dy = pos[3 * w + 1] - cy, dz = pos[3 * w + 2] - cz;
        if (dx * dx + dy * dy + dz * dz > r2) { stamp[w] = gen; continue; }
        const wx = nrm[3 * w], wy = nrm[3 * w + 1], wz = nrm[3 * w + 2];
        const c = neighborMode ? wx * nvx + wy * nvy + wz * nvz : wx * sn[0] + wy * sn[1] + wz * sn[2];
        if (c < cosMax) {
          // In neighbor mode another path may still reach w, so don't mark it.
          if (!neighborMode) stamp[w] = gen;
          continue;
        }
        stamp[w] = gen;
        out.push(w);
      }
    }
  }
  return Uint32Array.from(out);
}

// Open-boundary edges (used by exactly one triangle), as vertex -> boundary
// neighbours. Only boundary vertices appear in the map.
export function boundaryGraph(mesh) {
  const { index, vt } = mesh;
  const vc = vt.offsets.length - 1;
  const graph = new Map();
  const ws = [], counts = [];
  for (let v = 0; v < vc; v++) {
    ws.length = 0;
    counts.length = 0;
    for (let k = vt.offsets[v]; k < vt.offsets[v + 1]; k++) {
      const t = vt.list[k] * 3;
      for (let j = 0; j < 3; j++) {
        const w = index[t + j];
        if (w === v) continue;
        const at = ws.indexOf(w);
        if (at === -1) { ws.push(w); counts.push(1); } else counts[at]++;
      }
    }
    for (let i = 0; i < ws.length; i++) {
      if (counts[i] !== 1) continue;
      if (!graph.has(v)) graph.set(v, []);
      graph.get(v).push(ws[i]);
    }
  }
  return graph;
}

// The open-boundary loop nearest `center` (within `radius`, <= 0: unlimited
// along the loop but the start must be within `snap`). Returns vertex ids, or
// null if no boundary vertex is within `snap` of the click.
export function boundaryLoopNear(mesh, graph, center, radius, snap) {
  const { pos } = mesh;
  let start = -1, best = snap * snap;
  for (const v of graph.keys()) {
    const d2 = (pos[3 * v] - center[0]) ** 2 + (pos[3 * v + 1] - center[1]) ** 2 + (pos[3 * v + 2] - center[2]) ** 2;
    if (d2 <= best) { best = d2; start = v; }
  }
  if (start < 0) return null;
  const r2 = radius > 0 ? radius * radius : Infinity;
  const seen = new Set([start]);
  const out = [start];
  for (let head = 0; head < out.length; head++) {
    for (const w of graph.get(out[head]) || []) {
      if (seen.has(w)) continue;
      seen.add(w);
      const d2 = (pos[3 * w] - center[0]) ** 2 + (pos[3 * w + 1] - center[1]) ** 2 + (pos[3 * w + 2] - center[2]) ** 2;
      if (d2 <= r2) out.push(w);
    }
  }
  return Uint32Array.from(out);
}

// ---------------------------------------------------------------------------
// Mesh simplification
// ---------------------------------------------------------------------------

// Quadric error decimation (Garland & Heckbert) by edge collapse.
//
// Every vertex carries a quadric: the area-weighted sum of squared distances
// to the planes of the triangles it stands for. Collapsing u onto v costs the
// combined quadric at v divided by the combined weight, which is the mean
// squared distance from v to all of those planes. Flat areas collapse at zero
// cost and merge into a few large triangles, while curved detail stays.
//
// Vertices only ever collapse onto existing vertices, so the result is a new
// index buffer into the same `pos`. Open borders keep their outline: a border
// vertex can only slide along the border, and border edges carry extra planes
// that hold it in place. Non-manifold vertices never move. A collapse is
// skipped if it would flip or flatten a triangle, or pinch the surface.
//
// Runs in passes, like meshoptimizer: each pass rebuilds adjacency, ranks the
// edges by their cheapest legal collapse, and applies them in that order,
// touching each vertex at most once.
//
// opts.targetTriangles: stop at this many triangles.
// opts.maxDeviation: how far (scan units) the surface may move, 0 or missing
//   for no limit. The cost is an RMS distance, and on smooth surfaces the
//   worst point ends up about twice that far, so collapses are capped at
//   half of maxDeviation. Measure the result if you need a guarantee.
// Returns { index, error }, where error is the largest cost applied (as a distance).
export function simplifyMesh(pos, index, opts = {}) {
  const vc = pos.length / 3;
  const target = Math.max(0, Math.floor(opts.targetTriangles || 0));
  const limit = opts.maxDeviation > 0 ? (opts.maxDeviation / 2) ** 2 : Infinity;
  const BORDER_WEIGHT = 4;
  const MIN_COS = 0.25; // a collapse may turn a triangle's normal by at most ~75°

  // Centered copy, so plane offsets stay small and the quadrics keep precision.
  const c = [0, 0, 0];
  for (let i = 0; i < vc; i++) for (let k = 0; k < 3; k++) c[k] += pos[3 * i + k];
  const P = new Float64Array(3 * vc);
  for (let i = 0; i < 3 * vc; i++) P[i] = pos[i] - c[i % 3] / (vc || 1);

  // Start from the triangles that have three distinct corners.
  let ib = new Uint32Array(index.length);
  let tc = 0;
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t], b = index[t + 1], d = index[t + 2];
    if (a !== b && b !== d && a !== d) { ib[3 * tc] = a; ib[3 * tc + 1] = b; ib[3 * tc + 2] = d; tc++; }
  }
  ib = ib.subarray(0, 3 * tc);
  let adj = vertexTriangles(ib, vc);
  const corner = (t, v) => (ib[3 * t] === v ? 0 : ib[3 * t + 1] === v ? 1 : 2);
  const NEXT = [1, 2, 0], PREV = [2, 0, 1];
  // Is there a triangle with the directed edge a -> b?
  const hasEdge = (a, b) => {
    for (let k = adj.offsets[a]; k < adj.offsets[a + 1]; k++) {
      const t = adj.list[k];
      if (ib[3 * t + NEXT[corner(t, a)]] === b) return true;
    }
    return false;
  };

  // Quadrics: the 10 coefficients of a symmetric 4x4, plus a weight.
  const Q = new Float64Array(10 * vc), W = new Float64Array(vc);
  const addPlane = (v, a, b, cc, d, w) => {
    const q = 10 * v;
    Q[q] += w * a * a; Q[q + 1] += w * a * b; Q[q + 2] += w * a * cc; Q[q + 3] += w * a * d;
    Q[q + 4] += w * b * b; Q[q + 5] += w * b * cc; Q[q + 6] += w * b * d;
    Q[q + 7] += w * cc * cc; Q[q + 8] += w * cc * d; Q[q + 9] += w * d * d;
    W[v] += w;
  };
  for (let t = 0; t < tc; t++) {
    const v = [ib[3 * t], ib[3 * t + 1], ib[3 * t + 2]];
    const p = v.map((i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]]);
    const n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
    const l = len(n);
    if (!(l > 0)) continue;
    const u = scale(n, 1 / l);
    const d = -dot(u, p[0]);
    for (const i of v) addPlane(i, u[0], u[1], u[2], d, l / 2);
    for (let j = 0; j < 3; j++) {
      const a = v[j], b = v[NEXT[j]];
      if (hasEdge(b, a)) continue;
      const e = sub(p[NEXT[j]], p[j]);
      const m = normalize(cross(e, u));
      const dm = -dot(m, p[j]);
      const w = BORDER_WEIGHT * dot(e, e);
      addPlane(a, m[0], m[1], m[2], dm, w);
      addPlane(b, m[0], m[1], m[2], dm, w);
    }
  }
  const evalQ = (q, x, y, z) =>
    Q[q] * x * x + Q[q + 4] * y * y + Q[q + 7] * z * z +
    2 * (Q[q + 1] * x * y + Q[q + 2] * x * z + Q[q + 5] * y * z + Q[q + 3] * x + Q[q + 6] * y + Q[q + 8] * z) +
    Q[q + 9];
  const collapseCost = (u, v) => {
    const w = W[u] + W[v];
    if (!(w > 0)) return 0;
    const x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2];
    return Math.max(0, (evalQ(10 * u, x, y, z) + evalQ(10 * v, x, y, z)) / w);
  };

  // Vertex kinds, rebuilt every pass.
  const UNUSED = 0, INTERIOR = 1, BORDER = 2, LOCKED = 3;
  const kind = new Uint8Array(vc);
  let nx = new Uint32Array(64), pv = new Uint32Array(64);
  function classify() {
    const { offsets, list } = adj;
    for (let v = 0; v < vc; v++) {
      const s = offsets[v], k = offsets[v + 1] - s;
      if (!k) { kind[v] = UNUSED; continue; }
      if (k > nx.length) { nx = new Uint32Array(2 * k); pv = new Uint32Array(2 * k); }
      for (let i = 0; i < k; i++) {
        const t = list[s + i], j = corner(t, v);
        nx[i] = ib[3 * t + NEXT[j]];
        pv[i] = ib[3 * t + PREV[j]];
      }
      // Each edge at v must be used once in each direction (interior) or
      // once in total (border). Anything else is non-manifold.
      let bad = false, borders = 0, start = 0;
      for (let i = 0; i < k && !bad; i++) {
        let sameOut = 0, twinOut = 0, sameIn = 0, twinIn = 0;
        for (let m = 0; m < k; m++) {
          if (nx[m] === nx[i]) sameOut++;
          if (pv[m] === nx[i]) twinOut++;
          if (pv[m] === pv[i]) sameIn++;
          if (nx[m] === pv[i]) twinIn++;
        }
        if (sameOut > 1 || sameIn > 1 || twinOut > 1 || twinIn > 1) bad = true;
        if (!twinIn) { borders++; start = i; }
        if (!twinOut) borders++;
      }
      if (borders !== 0 && borders !== 2) bad = true;
      if (!bad) {
        // The triangles must form a single fan: walking across shared edges
        // from the first one has to reach all of them.
        let i = start, steps = 1;
        for (; steps <= k; steps++) {
          let m = 0;
          while (m < k && pv[m] !== nx[i]) m++;
          if (m === k || m === start) break;
          i = m;
        }
        if (steps !== k) bad = true;
      }
      kind[v] = bad ? LOCKED : borders ? BORDER : INTERIOR;
    }
  }

  // Can u collapse onto v right now? Returns how many triangles that removes
  // (0 = not allowed). `remap` holds this pass's earlier collapses.
  const remap = Uint32Array.from({ length: vc }, (_, i) => i);
  const mark = new Uint32Array(vc);
  let gen = 0;
  let sa = new Uint32Array(64), sb = new Uint32Array(64);
  function checkCollapse(u, v) {
    const { offsets, list } = adj;
    const g = (gen += 2); // mark: g = neighbour of u, g + 1 = also a neighbour of v
    let shared = 0, kept = 0;
    const ux = P[3 * u], uy = P[3 * u + 1], uz = P[3 * u + 2];
    const vx = P[3 * v], vy = P[3 * v + 1], vz = P[3 * v + 2];
    if (offsets[u + 1] - offsets[u] > sa.length) {
      sa = new Uint32Array(2 * (offsets[u + 1] - offsets[u]));
      sb = new Uint32Array(sa.length);
    }
    for (let k = offsets[u]; k < offsets[u + 1]; k++) {
      const t = list[k], j = corner(t, u);
      const a = remap[ib[3 * t + NEXT[j]]], b = remap[ib[3 * t + PREV[j]]];
      if (a === b) continue; // already collapsed away this pass
      if (a !== v) mark[a] = g;
      if (b !== v) mark[b] = g;
      if (a === v || b === v) { shared++; continue; }
      // Normal before (u, a, b) and after (v, a, b).
      const ax = P[3 * a], ay = P[3 * a + 1], az = P[3 * a + 2];
      const bx = P[3 * b], by = P[3 * b + 1], bz = P[3 * b + 2];
      const n0 = cross([ax - ux, ay - uy, az - uz], [bx - ux, by - uy, bz - uz]);
      const n1 = cross([ax - vx, ay - vy, az - vz], [bx - vx, by - vy, bz - vz]);
      const l0 = dot(n0, n0), l1 = dot(n1, n1);
      if (l0 > 0 && dot(n0, n1) <= MIN_COS * Math.sqrt(l0 * l1)) return 0;
      sa[kept] = a; sb[kept] = b; kept++;
    }
    if (!shared) return 0;
    // Link condition: u and v may only have the vertices opposite their
    // shared edge in common, and no surviving triangle of u may already
    // exist at v.
    let common = 0;
    for (let k = offsets[v]; k < offsets[v + 1]; k++) {
      const t = list[k], j = corner(t, v);
      const a = remap[ib[3 * t + NEXT[j]]], b = remap[ib[3 * t + PREV[j]]];
      if (a === b) continue;
      if (a !== u && mark[a] === g) { mark[a] = g + 1; common++; }
      if (b !== u && mark[b] === g) { mark[b] = g + 1; common++; }
      for (let m = 0; m < kept; m++) {
        if ((sa[m] === a && sb[m] === b) || (sa[m] === b && sb[m] === a)) return 0;
      }
    }
    return common > shared ? 0 : shared;
  }

  // Candidate buffers, sized for the first pass (the largest).
  const cap = 3 * tc;
  const cu = new Uint32Array(cap), cv = new Uint32Array(cap);
  const cost = new Float32Array(cap), bits = new Uint32Array(cost.buffer);
  const order = new Uint32Array(cap), buckets = new Uint32Array(2049);
  const lock = new Uint8Array(vc);
  let maxCost = 0;

  for (let pass = 0; pass < 500 && tc > target; pass++) {
    classify();
    let n = 0;
    for (let t = 0; t < tc; t++) {
      for (let j = 0; j < 3; j++) {
        const a = ib[3 * t + j], b = ib[3 * t + NEXT[j]];
        const border = !hasEdge(b, a);
        if (!border && a > b) continue; // interior edges show up twice; take one
        const ka = kind[a], kb = kind[b];
        const aMoves = ka === INTERIOR || (ka === BORDER && border && kb >= BORDER);
        const bMoves = kb === INTERIOR || (kb === BORDER && border && ka >= BORDER);
        const cab = aMoves ? collapseCost(a, b) : Infinity;
        const cba = bMoves ? collapseCost(b, a) : Infinity;
        const best = Math.min(cab, cba);
        if (!(best <= limit)) continue;
        cu[n] = cab <= cba ? a : b;
        cv[n] = cab <= cba ? b : a;
        cost[n++] = best;
      }
    }
    if (!n) break;

    // Approximate sort: bucket by the top bits of the float (exponent plus
    // 3 bits of mantissa), which is plenty to apply cheap collapses first.
    buckets.fill(0);
    for (let i = 0; i < n; i++) buckets[(bits[i] >>> 20) + 1]++;
    for (let k = 1; k < buckets.length; k++) buckets[k] += buckets[k - 1];
    for (let i = 0; i < n; i++) order[buckets[bits[i] >>> 20]++] = i;

    // Don't go much past the cost needed to reach the target in this pass:
    // later passes re-rank with updated quadrics.
    const goal = tc - target;
    const atGoal = cost[order[Math.min(n, Math.ceil(goal / 2)) - 1]];
    const passLimit = Math.min(limit, atGoal * 1.5);

    lock.fill(0);
    let removed = 0, applied = 0;
    for (let r = 0; r < n && removed < goal; r++) {
      const i = order[r];
      if (cost[i] > passLimit && applied) break;
      const u = cu[i], v = cv[i];
      if (lock[u] || lock[v]) continue;
      const gone = checkCollapse(u, v);
      if (!gone) continue;
      remap[u] = v;
      lock[u] = lock[v] = 1;
      for (let m = 0; m < 10; m++) Q[10 * v + m] += Q[10 * u + m];
      W[v] += W[u];
      removed += gone;
      applied++;
      if (cost[i] > maxCost) maxCost = cost[i];
    }
    if (!applied) break;

    let o = 0;
    for (let t = 0; t < tc; t++) {
      const a = remap[ib[3 * t]], b = remap[ib[3 * t + 1]], d = remap[ib[3 * t + 2]];
      if (a === b || b === d || a === d) continue;
      ib[o++] = a; ib[o++] = b; ib[o++] = d;
    }
    ib = ib.subarray(0, o);
    tc = o / 3;
    adj = vertexTriangles(ib, vc);
  }
  return { index: ib.slice(), error: Math.sqrt(maxCost) };
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

// Binary STL of an indexed mesh with every vertex transformed by M
// (row-major 4x4, rigid).
export function writeBinarySTL(pos, index, M) {
  const vc = pos.length / 3;
  const tp = new Float32Array(pos.length);
  for (let i = 0; i < vc; i++) {
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    tp[3 * i] = M[0] * x + M[1] * y + M[2] * z + M[3];
    tp[3 * i + 1] = M[4] * x + M[5] * y + M[6] * z + M[7];
    tp[3 * i + 2] = M[8] * x + M[9] * y + M[10] * z + M[11];
  }
  const tc = index.length / 3;
  const buf = new ArrayBuffer(84 + 50 * tc);
  const dv = new DataView(buf);
  const header = 'Binary STL aligned by Mesh Align';
  for (let i = 0; i < header.length; i++) dv.setUint8(i, header.charCodeAt(i));
  dv.setUint32(80, tc, true);
  let o = 84;
  for (let t = 0; t < tc; t++) {
    const a = 3 * index[3 * t], b = 3 * index[3 * t + 1], c = 3 * index[3 * t + 2];
    const e1 = [tp[b] - tp[a], tp[b + 1] - tp[a + 1], tp[b + 2] - tp[a + 2]];
    const e2 = [tp[c] - tp[a], tp[c + 1] - tp[a + 1], tp[c + 2] - tp[a + 2]];
    const n = normalize(cross(e1, e2));
    for (const val of n) { dv.setFloat32(o, val, true); o += 4; }
    for (const vi of [a, b, c]) {
      dv.setFloat32(o, tp[vi], true);
      dv.setFloat32(o + 4, tp[vi + 1], true);
      dv.setFloat32(o + 8, tp[vi + 2], true);
      o += 12;
    }
    dv.setUint16(o, 0, true);
    o += 2;
  }
  return buf;
}

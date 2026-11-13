// STEP (ISO 10303-21, AP214) export of fitted features as B-rep reference
// surfaces, in the aligned frame: planes -> rectangular planar faces,
// circles -> planar disks, cylinders -> cylindrical faces. CAD imports them as
// surface bodies with true planes and axes to sketch on and mate to.

import { add, sub, scale, dot, cross, len, normalize, perpBasis, transformPoint, transformDir } from './geom.js';

const BASIS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

const SI_LENGTH = { mm: '.MILLI.,.METRE.', cm: '.CENTI.,.METRE.', m: '$,.METRE.' };
export const STEP_UNITS = ['mm', 'cm', 'm', 'in'];

// STEP REAL: always has a decimal point, exponent as E.
export function stepReal(x) {
  if (!Number.isFinite(x)) throw new Error('non-finite coordinate');
  if (Math.abs(x) < 1e-12) return '0.';
  const [m, e] = x.toPrecision(15).split('e');
  const mant = m.includes('.') ? m.replace(/0+$/, '') : `${m}.`;
  return e ? `${mant}E${e}` : mant;
}

function stepString(s) {
  const ascii = String(s).replace(/[^\x20-\x7e]/g, '_');
  return `'${ascii.replace(/\\/g, '\\\\').replace(/'/g, "''")}'`;
}

// features: [{ name, fit }] with fits in scan coordinates (other types are
// skipped); M: row-major 4x4 scan -> aligned transform; units: STEP_UNITS
// entry describing what the coordinate numbers mean.
// Returns { text, count } where count is the number of surfaces written.
export function writeStepReference(features, M, { units = 'mm', name = 'reference', timestamp } = {}) {
  const out = [];
  let n = 0;
  const E = (s) => { out.push(`#${++n}=${s};`); return `#${n}`; };
  const P = (p) => E(`CARTESIAN_POINT('',(${p.map(stepReal).join(',')}))`);
  const D = (d) => E(`DIRECTION('',(${normalize(d).map(stepReal).join(',')}))`);
  const A2 = (o, z, x) => E(`AXIS2_PLACEMENT_3D('',${P(o)},${D(z)},${D(x)})`);
  const vertex = (p) => E(`VERTEX_POINT('',${P(p)})`);
  const lineEdge = (va, a, vb, b) => {
    const d = sub(b, a);
    const line = E(`LINE('',${P(a)},${E(`VECTOR('',${D(d)},${stepReal(len(d))})`)})`);
    return E(`EDGE_CURVE('',${va},${vb},${line},.T.)`);
  };
  const circleEdge = (center, axis, ref, r) => {
    const v = vertex(add(center, scale(ref, r)));
    return { v, edge: E(`EDGE_CURVE('',${v},${v},${E(`CIRCLE('',${A2(center, axis, ref)},${stepReal(r)})`)},.T.)`) };
  };
  const oriented = (edge, sense) => E(`ORIENTED_EDGE('',*,*,${edge},${sense ? '.T.' : '.F.'})`);
  const face = (label, edges, surface) => {
    const loop = E(`EDGE_LOOP('',(${edges.join(',')}))`);
    return E(`ADVANCED_FACE(${stepString(label)},(${E(`FACE_OUTER_BOUND('',${loop},.T.)`)}),${surface},.T.)`);
  };

  const bodies = [];
  for (const { name: label, fit } of features) {
    let f;
    if (fit.type === 'plane') {
      const [u0, u1, v0, v1] = fit.extent;
      const p0 = transformPoint(M, fit.point), nrm = normalize(transformDir(M, fit.dir));
      const fu = transformDir(M, fit.u), fv = transformDir(M, fit.v);
      const outline = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]].map(([s, t]) => add(p0, add(scale(fu, s), scale(fv, t))));
      // Square the patch to the aligned axes (first axis not near the normal)
      // so it lines up with the part in CAD, then pad each side by 10%.
      const u = normalize(BASIS.map((ax) => sub(ax, scale(nrm, dot(ax, nrm)))).find((q) => len(q) > 0.5));
      const v = cross(nrm, u);
      const ss = outline.map((p) => dot(sub(p, p0), u)), ts = outline.map((p) => dot(sub(p, p0), v));
      const [s0, s1, t0, t1] = [Math.min(...ss), Math.max(...ss), Math.min(...ts), Math.max(...ts)];
      const a = 0.6 * (s1 - s0), b = 0.6 * (t1 - t0);
      const c = add(p0, add(scale(u, (s0 + s1) / 2), scale(v, (t0 + t1) / 2)));
      // counter-clockwise seen from the normal, since (u, v, n) is right-handed
      const corners = [[-a, -b], [a, -b], [a, b], [-a, b]].map(([s, t]) => add(c, add(scale(u, s), scale(v, t))));
      const vs = corners.map(vertex);
      const edges = corners.map((p, i) => lineEdge(vs[i], p, vs[(i + 1) % 4], corners[(i + 1) % 4]));
      f = face(label, edges.map((e) => oriented(e, true)), E(`PLANE('',${A2(c, nrm, u)})`));
    } else if (fit.type === 'circle') {
      const c = transformPoint(M, fit.point), nrm = transformDir(M, fit.dir), u = transformDir(M, fit.u);
      const { edge } = circleEdge(c, nrm, u, fit.radius);
      f = face(label, [oriented(edge, true)], E(`PLANE('',${A2(c, nrm, u)})`));
    } else if (fit.type === 'cylinder') {
      const h = Math.max(fit.height, fit.radius);
      const c = transformPoint(M, fit.point), d = normalize(transformDir(M, fit.dir));
      const [u] = perpBasis(d);
      const r = fit.radius;
      const bottom = add(c, scale(d, -h / 2)), top = add(c, scale(d, h / 2));
      const lo = circleEdge(bottom, d, u, r), hi = circleEdge(top, d, u, r);
      const seam = lineEdge(lo.v, add(bottom, scale(u, r)), hi.v, add(top, scale(u, r)));
      // (angle, height) parameter rectangle walked counter-clockwise:
      // bottom circle, up the seam, top circle backwards, down the seam
      f = face(label, [oriented(lo.edge, true), oriented(seam, true), oriented(hi.edge, false), oriented(seam, false)],
        E(`CYLINDRICAL_SURFACE('',${A2(bottom, d, u)},${stepReal(r)})`));
    } else {
      continue;
    }
    bodies.push(E(`SHELL_BASED_SURFACE_MODEL(${stepString(label)},(${E(`OPEN_SHELL('',(${f}))`)}))`));
  }

  let lengthUnit;
  if (units === 'in') {
    const mm = E('(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))');
    const measure = E(`LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),${mm})`);
    const dims = E('DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)');
    lengthUnit = E(`(CONVERSION_BASED_UNIT('INCH',${measure})LENGTH_UNIT()NAMED_UNIT(${dims}))`);
  } else {
    if (!SI_LENGTH[units]) throw new Error(`unknown units ${units}`);
    lengthUnit = E(`(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(${SI_LENGTH[units]}))`);
  }
  const angleUnit = E('(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))');
  const solidUnit = E('(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())');
  const uncertainty = E(`UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),${lengthUnit},'distance_accuracy_value','confusion accuracy')`);
  const context = E(`(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((${uncertainty}))` +
    `GLOBAL_UNIT_ASSIGNED_CONTEXT((${lengthUnit},${angleUnit},${solidUnit}))` +
    `REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))`);

  const app = E("APPLICATION_CONTEXT('core data for automotive mechanical design processes')");
  E(`APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,${app})`);
  const product = E(`PRODUCT(${stepString(name)},${stepString(name)},'',(${E(`PRODUCT_CONTEXT('',${app},'mechanical')`)}))`);
  E(`PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(${product}))`);
  const formation = E(`PRODUCT_DEFINITION_FORMATION('','',${product})`);
  const definition = E(`PRODUCT_DEFINITION('design','',${formation},${E(`PRODUCT_DEFINITION_CONTEXT('part definition',${app},'design')`)})`);
  const shape = E(`PRODUCT_DEFINITION_SHAPE('','',${definition})`);
  const origin = A2([0, 0, 0], [0, 0, 1], [1, 0, 0]);
  const rep = E(`MANIFOLD_SURFACE_SHAPE_REPRESENTATION(${stepString(name)},(${[origin, ...bodies].join(',')}),${context})`);
  E(`SHAPE_DEFINITION_REPRESENTATION(${shape},${rep})`);

  const stamp = timestamp || new Date().toISOString().slice(0, 19);
  const text = [
    'ISO-10303-21;',
    'HEADER;',
    "FILE_DESCRIPTION(('Reference geometry exported by Mesh Align'),'2;1');",
    `FILE_NAME(${stepString(`${name}.step`)},'${stamp}',(''),(''),'Mesh Align','Mesh Align','');`,
    "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
    'ENDSEC;',
    'DATA;',
    ...out,
    'ENDSEC;',
    'END-ISO-10303-21;',
    '',
  ].join('\n');
  return { text, count: bodies.length };
}

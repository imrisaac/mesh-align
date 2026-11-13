import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as G from '../src/geom.js';
import { writeStepReference, stepReal } from '../src/step.js';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const pack = (pts) => Float64Array.from(pts.flat());

function sampleFeatures() {
  const plane = [], planeN = [], wall = [], wallN = [];
  for (let i = 0; i < 400; i++) {
    plane.push([(i % 20) * 3, Math.floor(i / 20) * 2, 0]);
    planeN.push([0, 0, 1]);
    const a = i * 0.7;
    wall.push([10 + 2 * Math.cos(a), 5 + 2 * Math.sin(a), (i % 10) * 0.5]);
    wallN.push([-Math.cos(a), -Math.sin(a), 0]);
  }
  return [
    { name: "Plane 'A'", fit: G.fitFeature('plane', pack(plane), pack(planeN)) },
    { name: 'Circle 1', fit: G.fitFeature('circle', pack([[3, 0, 0], [0, 3, 0], [-3, 0, 0]]), null) },
    { name: 'Cylinder 1', fit: G.fitFeature('cylinder', pack(wall), pack(wallN)) },
    { name: 'Point 1', fit: G.fitFeature('point', pack([[1, 2, 3]]), null) },
  ];
}

test('stepReal always writes a STEP REAL', () => {
  assert.equal(stepReal(0), '0.');
  assert.equal(stepReal(-1e-15), '0.');
  assert.equal(stepReal(100), '100.');
  assert.equal(stepReal(-12.34), '-12.34');
  assert.equal(stepReal(1e-7), '1.E-7');
  assert.equal(stepReal(2.5e21), '2.5E+21');
  assert.throws(() => stepReal(NaN));
});

test('STEP file is well formed and references resolve', () => {
  const { text, count } = writeStepReference(sampleFeatures(), IDENTITY, { units: 'mm', name: 'part', timestamp: '2026-01-01T00:00:00' });
  assert.equal(count, 3, 'points are skipped');
  assert.ok(text.startsWith('ISO-10303-21;\nHEADER;') && text.trimEnd().endsWith('END-ISO-10303-21;'));
  const data = text.slice(text.indexOf('DATA;') + 5, text.lastIndexOf('ENDSEC;'));
  const defined = new Set();
  for (const line of data.trim().split('\n')) {
    const m = line.match(/^#(\d+)=(.*);$/);
    assert.ok(m, `bad line ${line}`);
    assert.ok(!defined.has(m[1]), `duplicate #${m[1]}`);
    defined.add(m[1]);
    const body = m[2].replace(/'(?:[^']|'')*'/g, "''");
    assert.equal((body.match(/\(/g) || []).length, (body.match(/\)/g) || []).length, `parens in ${line}`);
  }
  for (const [, id] of data.replace(/'(?:[^']|'')*'/g, "''").matchAll(/#(\d+)/g)) assert.ok(defined.has(id), `undefined #${id}`);
  assert.match(text, /SHELL_BASED_SURFACE_MODEL\('Plane ''A'''/);
  assert.equal((text.match(/ADVANCED_FACE/g) || []).length, 3);
  assert.match(text, /CYLINDRICAL_SURFACE\('',#\d+,2\.\d*\)/);
  assert.match(text, /SI_UNIT\(\.MILLI\.,\.METRE\.\)/);
});

test('plane patch is squared to the frame and covers the fitted points', () => {
  const [plane] = sampleFeatures();
  const { text } = writeStepReference([plane], IDENTITY);
  const pts = [...text.matchAll(/CARTESIAN_POINT\('',\(([^)]*)\)\)/g)].map((m) => m[1].split(',').map(Number)).slice(0, 4);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  assert.deepEqual(new Set(xs.map((x) => x.toFixed(6))).size, 2, 'edges parallel to X/Y');
  assert.ok(Math.min(...xs) < 0 && Math.max(...xs) > 57 && Math.min(...ys) < 0 && Math.max(...ys) > 38);
  assert.ok(pts.every((p) => Math.abs(p[2]) < 1e-9));
});

test('inch units use a conversion-based unit', () => {
  const { text } = writeStepReference(sampleFeatures(), IDENTITY, { units: 'in' });
  assert.match(text, /CONVERSION_BASED_UNIT\('INCH',#\d+\)/);
  assert.match(text, /LENGTH_MEASURE\(25\.4\)/);
  assert.throws(() => writeStepReference(sampleFeatures(), IDENTITY, { units: 'furlong' }));
});

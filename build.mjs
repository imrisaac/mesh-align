// Bundles src/main.js (three.js included) into one self-contained HTML file,
// mesh-align.html, that works offline straight from disk.
import { build } from 'esbuild';
import { readFile, writeFile } from 'node:fs/promises';

const result = await build({
  entryPoints: ['src/main.js'],
  bundle: true,
  minify: true,
  format: 'iife',
  target: 'es2020',
  write: false,
  legalComments: 'none',
});
const js = result.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const html = (await readFile('src/index.html', 'utf8')).replace('<!-- APP_SCRIPT -->', () => `<script>${js}</script>`);
await writeFile('mesh-align.html', html);
console.log(`mesh-align.html: ${(html.length / 1024).toFixed(0)} KB`);

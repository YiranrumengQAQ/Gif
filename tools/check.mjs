#!/usr/bin/env node
/**
 * GIFX self-check — the gate that must stay green.
 *
 *   1. every source/test/tool file parses
 *   2. every module imports (catches bad re-exports, cycles and typos)
 *   3. the barrel (`src/index.js`) exposes the names the docs promise
 *   4. the shipped library stays browser-pure and dependency-free
 *   5. `package.json` points at files that exist (dist/types are build output, so
 *      their absence is a warning, not a failure)
 *   6. the test suite passes
 *
 * Usage: node tools/check.mjs [--quick] (skips the test run)
 */
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(import.meta.dirname, '..');
const quick = process.argv.includes('--quick');
const dirs = ['src', 'test', 'tools'];
const files = [];
for (const dir of dirs) {
  const abs = path.join(root, dir);
  if (!existsSync(abs)) continue;
  (function walk(d) {
    for (const entry of readdirSync(d)) {
      const p = path.join(d, entry);
      if (statSync(p).isDirectory()) walk(p);
      else if (/\.(js|mjs)$/.test(p)) files.push(p);
    }
  })(abs);
}

let failures = 0;
const fail = (msg) => {
  failures++;
  console.log(`  ✖ ${msg}`);
};
const rel = (p) => path.relative(root, p);

console.log(`GIFX check — ${files.length} files`);

// --- 1. parse -------------------------------------------------------------
console.log('\n[1/6] syntax');
for (const file of files) {
  const res = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
  if (res.status !== 0) fail(`${rel(file)}: ${res.stderr.split('\n').slice(0, 2).join(' ')}`);
}
console.log(failures ? '' : '  ✓ all files parse');

// --- 2 + 3. import + public surface --------------------------------------
console.log('\n[2/6] module imports');
const before = failures;
const srcFiles = files.filter((f) => f.includes(`${path.sep}src${path.sep}`));
for (const file of srcFiles) {
  try {
    await import(pathToFileURL(file).href);
  } catch (err) {
    fail(`${rel(file)}: ${err.message.split('\n')[0]}`);
  }
}
console.log(failures === before ? `  ✓ ${srcFiles.length} modules import cleanly` : '');

console.log('\n[3/6] public surface');
try {
  const api = await import(pathToFileURL(path.join(root, 'src/index.js')).href);
  const missing = api.PUBLIC_SURFACE.filter((name) => api[name] === undefined);
  if (missing.length) fail(`src/index.js is missing: ${missing.join(', ')}`);
  else console.log(`  ✓ ${api.PUBLIC_SURFACE.length} promised names present (${Object.keys(api).length} exports)`);
  if (!/^\d+\.\d+\.\d+$/.test(String(api.VERSION))) fail('VERSION is not a semver string');
} catch (err) {
  fail(`src/index.js did not load: ${err.message.split('\n')[0]}`);
}

// --- 4. browser purity ----------------------------------------------------
console.log('\n[4/6] dependency-free + browser-only');
let offenders = 0;
for (const file of srcFiles) {
  const text = stripComments(readFileSync(file, 'utf8'));
  const specifiers = [...text.matchAll(/(?:^|\n)\s*(?:import|export)[^\n]*?\bfrom\s+['"]([^'"]+)['"]/g)]
    .map((m) => m[1])
    .concat([...text.matchAll(/(?:^|\n)\s*import\s+['"]([^'"]+)['"]/g)].map((m) => m[1]));
  for (const spec of specifiers) {
    if (spec.startsWith('.') || spec.startsWith('/')) continue;
    fail(`${rel(file)} imports "${spec}" — the library must have no runtime dependencies`);
    offenders++;
  }
  const nodeApi = /\b(?:require\(\s*['"]node:|from\s+['"]node:)/.test(text);
  if (nodeApi) {
    fail(`${rel(file)} uses a node: built-in — it will not run in a browser`);
    offenders++;
  }
}
console.log(failures === before || offenders === 0 ? '  ✓ no bare or node: imports in src/' : '');

// --- 5. package wiring ---------------------------------------------------
console.log('\n[5/6] package.json wiring');
const pkg = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
const buildOutputs = ['./dist/', './types/'];
for (const [key, target] of Object.entries(flattenExports(pkg.exports))) {
  if (target.includes('*')) continue;
  const abs = path.join(root, target);
  if (existsSync(abs)) continue;
  const line = `${key} → ${target}`;
  if (buildOutputs.some((prefix) => target.startsWith(prefix))) console.log(`  ! ${line} (build output, run npm run build)`);
  else fail(`${line} does not exist`);
}
for (const dep of Object.keys(pkg.dependencies || {})) fail(`runtime dependency "${dep}" is not allowed`);
for (const [name, ver] of Object.entries(pkg.devDependencies || {})) {
  if (!/omggif|gifenc|esbuild/.test(name)) console.log(`  ! unexpected devDependency "${name}" (document it in the README)`);
  if (!ver) fail(`devDependency "${name}" has no version range`);
}
console.log('  ✓ exports map and dependency policy checked');

/** Comments legitimately mention module names and specifiers; ignore them. */
function stripComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
}

function flattenExports(exports, prefix = '') {
  const out = {};
  for (const [k, v] of Object.entries(exports || {})) {
    const key = prefix ? `${prefix} ${k}` : k;
    if (typeof v === 'string') out[key] = v;
    else if (v && typeof v === 'object') Object.assign(out, flattenExports(v, key));
  }
  return out;
}

// --- 6. tests -------------------------------------------------------------
console.log('\n[6/6] tests');
if (quick) {
  console.log('  - skipped (--quick)');
} else {
  const testFiles = files.filter((f) => f.endsWith('.test.js')).sort();
  const res = spawnSync(process.execPath, ['--test', ...testFiles.map((f) => rel(f))], { cwd: root, encoding: 'utf8' });
  const tail = res.stdout
    .split('\n')
    .filter((l) => /^# (tests|suites|pass|fail|cancelled|skipped|todo|duration_ms) /.test(l))
    .join('  ');
  if (res.status !== 0) {
    fail(`test suite failed — ${tail}`);
    console.log(res.stdout.split('\n').filter((l) => l.startsWith('not ok')).join('\n'));
  } else console.log(`  ✓ ${tail || 'all tests passed'}`);
}

console.log(`\n${failures ? `✖ ${failures} problem(s)` : '✓ check passed'}`);
process.exit(failures ? 1 : 0);

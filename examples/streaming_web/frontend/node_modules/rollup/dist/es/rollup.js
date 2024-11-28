/*
  @license
	Rollup.js v4.27.3
	Mon, 18 Nov 2024 16:39:05 GMT - commit 7c0b1f8810013b5a351a976df30a6a5da4fa164b

	https://github.com/rollup/rollup

	Released under the MIT License.
*/
export { version as VERSION, defineConfig, rollup, watch } from './shared/node-entry.js';
import './shared/parseAst.js';
import '../native.js';
import 'node:path';
import 'path';
import 'node:process';
import 'node:perf_hooks';
import 'node:fs/promises';
import 'tty';

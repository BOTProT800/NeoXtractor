/*
 * Run the official Khronos glTF-Validator over glTF/GLB files.
 *
 *   NEOX_GLTF_VALIDATOR=<dir>/node_modules/gltf-validator node khronos_validate.js a.glb b.gltf
 *
 * The validator is the npm package ``gltf-validator``, published by Khronos
 * from https://github.com/KhronosGroup/glTF-Validator. It is not a project
 * dependency; install it anywhere with ``npm install gltf-validator`` and point
 * NEOX_GLTF_VALIDATOR at the package directory.
 *
 * Prints one JSON object per file: {"file", "validatorVersion", "numErrors",
 * "numWarnings", "numInfos", "numHints", "messages": [...]}. Exits 0 when it
 * ran, whatever the reports say; judging them is the caller's job.
 */

"use strict";

const fs = require("fs");
const path = require("path");

const location = process.env.NEOX_GLTF_VALIDATOR;
if (!location) {
  console.error("set NEOX_GLTF_VALIDATOR to the gltf-validator package directory");
  process.exit(2);
}
const validator = require(path.resolve(location));

async function report(file) {
  const bytes = new Uint8Array(fs.readFileSync(file));
  const result = await validator.validateBytes(bytes, {
    uri: path.basename(file),
    maxIssues: 0, // report everything
  });
  return {
    file,
    validatorVersion: result.validatorVersion,
    numErrors: result.issues.numErrors,
    numWarnings: result.issues.numWarnings,
    numInfos: result.issues.numInfos,
    numHints: result.issues.numHints,
    messages: result.issues.messages,
  };
}

(async () => {
  for (const file of process.argv.slice(2)) {
    console.log(JSON.stringify(await report(file)));
  }
})().catch((error) => {
  console.error(String((error && error.stack) || error));
  process.exit(1);
});

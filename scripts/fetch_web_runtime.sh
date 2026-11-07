#!/usr/bin/env bash
# Downloads transformers.js and its WebAssembly runtime from npm into web/vendor, so the dashboard's live demo
# serves them itself instead of depending on a third-party CDN.
set -euo pipefail
VERSION="4.3.1"
DEST="$(cd "$(dirname "$0")/.." && pwd)/${1:-web/vendor}"
mkdir -p "$DEST"
TMP="$(mktemp -d)"
cd "$TMP"
npm init -y > /dev/null
npm install --ignore-scripts --no-audit --no-fund "@huggingface/transformers@${VERSION}" > /dev/null
cp node_modules/@huggingface/transformers/dist/transformers.min.js "$DEST/"
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs "$DEST/"
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm "$DEST/"
ls -la "$DEST"

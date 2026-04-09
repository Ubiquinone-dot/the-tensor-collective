#!/usr/bin/env bash
# Convert BibTeX references to CSL-JSON for Hugo's data store.
#
# Edit the .bib file, then run:
#   ./scripts/build-bib.sh
#
# Hugo will pick up the regenerated data/bibliography.json automatically.
set -euo pipefail

SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/site"
BIB_IN="${SITE_DIR}/assets/bibliography/refs.bib"
JSON_OUT="${SITE_DIR}/data/bibliography.json"

if ! command -v pandoc >/dev/null 2>&1; then
    echo "error: pandoc is required (brew install pandoc)" >&2
    exit 1
fi

if [[ ! -f "${BIB_IN}" ]]; then
    echo "error: no .bib file at ${BIB_IN}" >&2
    exit 1
fi

mkdir -p "$(dirname "${JSON_OUT}")"
pandoc "${BIB_IN}" -t csljson -o "${JSON_OUT}"
echo "wrote ${JSON_OUT} ($(jq 'length' "${JSON_OUT}" 2>/dev/null || echo "?") entries)"

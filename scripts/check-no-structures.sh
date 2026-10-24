#!/usr/bin/env sh
# Fail the build if any raw structure file would be published.
# Usage: check-no-structures.sh <dir>   (defaults to site/public)
dir="${1:-site/public}"
found=$(find "$dir" -type f \( -iname '*.pdb' -o -iname '*.pdb.gz' -o -iname '*.ent' -o -iname '*.cif' \
  -o -iname '*.cif.gz' -o -iname '*.mmcif' -o -iname '*.bcif' -o -iname '*.mmtf' -o -iname '*.pse' \
  -o -iname '*.pze' -o -iname '*.pml' \))
if [ -n "$found" ]; then
  echo "Refusing to publish raw structure files:" >&2
  echo "$found" >&2
  exit 1
fi

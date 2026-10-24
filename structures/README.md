# structures/

Source structure files (PDB, mmCIF, PyMOL sessions) behind the homepage protein renders.
Some are unpublished designs, so they live here, outside `site/`, and are never served.

Netlify builds only `site/` and publishes only `site/public/` (see `netlify.toml`), so nothing
in this folder reaches the website. What does get published is derived from these files:
a rendered PNG and its dot data under `site/static/alive/data/`.

Rules:

- Keep `.pdb`, `.cif`, `.pse` and friends here, never under `site/`. `.gitignore` refuses them
  under `site/`, and `scripts/check-no-structures.sh` fails the Netlify build if one slips into
  the published output.
- Only commit files here while the GitHub repo is private. If it ever goes public again,
  move them out first; a public repo exposes its full git history.

"""Render the homepage designs from structures/ and convert each to dot data for the alive canvas.

    python scripts/alive/render_designs.py [name ...]

Needs PyMOL (`pip install pymol-open-source-whl`), numpy and Pillow. Writes the PNG renders to
structures/renders/ (kept out of the site) and the dot data to site/static/alive/data/.
Add the new file to PROTEINS in site/static/alive/index.html to put it on the homepage.
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from render_style import style, cmd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'structures')
# name: (session or structure file, protein selection, accent selection, extra side chains, outline)
JOBS = {
    'phosphotriesterase': ('seths_phosphotriesterase.pse', 'polymer.protein', 'not polymer and not solvent',
                           'byres (polymer.protein within 3.6 of (not polymer and not solvent))', True),
    'protease_tdp43': ('anqi-protease/tdp43.pse', 'rfd3 and chain A', 'rfd3 and (chain B+C or not polymer)',
                       'byres (rfd3 and chain A within 4.0 of (rfd3 and metals))', True),
    'crystal_unit_cell': ('h11_03_unit_cell.pse', 'unit_cell', None, None, False),
}

for name, (f, prot, acc, st, outline) in JOBS.items():
    if sys.argv[1:] and name not in sys.argv[1:]: continue
    png = os.path.join(SRC, 'renders', name + '.png'); os.makedirs(os.path.dirname(png), exist_ok=True)
    cmd.reinitialize(); cmd.load(os.path.join(SRC, f))
    style(prot, acc, st, out=png, outline=outline)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'img2alive.py'), png,
                    os.path.join(ROOT, 'site/static/alive/data', name + '.alive.json'), name], check=True)

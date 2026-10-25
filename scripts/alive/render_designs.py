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
SRC = os.environ.get('ALIVE_SRC', os.path.join(ROOT, 'structures'))   # override to render from elsewhere
# name: (session or structure file, protein selection, accent selection, extra side chains, outline[, orange part of protein])
JOBS = {
    'phosphotriesterase': ('seths_phosphotriesterase.pse', 'polymer.protein', 'not polymer and not solvent',
                           'byres (polymer.protein within 3.6 of (not polymer and not solvent))', True),
    'protease_tdp43': ('anqi-protease/tdp43.pse', 'rfd3 and chain A', 'rfd3 and (chain B+C or not polymer)',
                       'byres (rfd3 and chain A within 4.0 of (rfd3 and metals))', True),
    # each chain fuses a D3 motif (1-198) through a linker to a C2 motif (216-279): D3 teal, C2 orange
    'crystal_unit_cell': ('h11_03_unit_cell.pse', 'unit_cell', None, None, False, 'resi 216-279'),
}

for name, (f, prot, acc, st, outline, *paint) in JOBS.items():
    if sys.argv[1:] and name not in sys.argv[1:]: continue
    png = os.path.join(SRC, 'renders', name + '.png'); os.makedirs(os.path.dirname(png), exist_ok=True)
    cmd.reinitialize(); cmd.load(os.path.join(SRC, f))
    style(prot, acc, st, out=png, outline=outline, paint=paint[0] if paint else None)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'img2alive.py'), png,
                    os.path.join(ROOT, 'site/static/alive/data', name + '.alive.json'), name], check=True)

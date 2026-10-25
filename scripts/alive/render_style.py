"""Render a protein in the homepage hydrolase style: flat teal cartoon, orange ligand/site, outlined."""
import pymol; pymol.finish_launching(['pymol','-cq'])
from pymol import cmd

TEAL=[18/255,106/255,93/255]; ORANGE=[175/255,103/255,43/255]

def style(protein, accent=None, sticks=None, scene=None, width=1400, height=1200, out='out.png', outline=True):
    view=None
    if scene:
        cmd.scene(scene,'recall',animate=0); view=cmd.get_view()
    # Fresh copies drop the session's per-object/per-atom settings and colours.
    cmd.create('_p', protein)
    if accent: cmd.create('_a', f'({accent}) and not ({protein})')
    if sticks: cmd.create('_s', sticks)
    for o in cmd.get_names('objects'):
        if o not in ('_p','_a','_s'): cmd.delete(o)
    cmd.set_color('home_teal',TEAL); cmd.set_color('home_orange',ORANGE)
    cmd.hide('everything')
    cmd.show('cartoon','_p'); cmd.color('home_teal','_p')
    if accent:
        cmd.show('cartoon','_a'); cmd.show('sticks','_a and not hydro and not polymer.protein'); cmd.show('spheres','_a and metals')
        cmd.color('home_orange','_a')
    if sticks:
        cmd.show('sticks','_s and not hydro and not name N+C+O'); cmd.color('home_orange','_s')
    for k,v in dict(cartoon_transparency=0,stick_transparency=0,sphere_transparency=0,transparency=0,cartoon_color=-1,stick_color=-1,sphere_color=-1,cartoon_highlight_color=-1,sphere_scale=0.6,stick_radius=0.3,cartoon_side_chain_helper=1,cartoon_fancy_helices=1,
                    ambient=0.75,direct=0.35,reflect=0.0,specular=0,shininess=0,ray_shadows=0,
                    depth_cue=0,ray_trace_fog=0,fog=0,light_count=1,ray_trace_mode=1 if outline else 0,
                    ray_trace_gain=0.6,antialias=2,ray_opaque_background=0,orthoscopic=1).items():
        cmd.set(k,v)
    cmd.bg_color('white')
    if view: cmd.set_view(view)
    else:
        cmd.orient('_p'); cmd.zoom('all', 2, complete=1)
    cmd.png(out, width=width, height=height, ray=1)

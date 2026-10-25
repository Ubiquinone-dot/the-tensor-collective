"""Convert a transparent PNG render into an .alive.json dot hierarchy for site/static/alive.

Leaves are the opaque cells of a square grid, coloured by the cell's mean colour. Internal
nodes come from recursive 2-means bisection, so every node has two children and children
precede parents in the list. Each node carries:
  x, y  centroid      r  sqrt(sum child r^2) (area-preserving)      c  mean colour
  s     split level: 1 for leaves, falling with log(cluster size)  d  split transition width
"""
import json, math, sys
import numpy as np
from PIL import Image

def leaves_from_image(path, cell=5, target=1100, alpha_min=0.5):
    im = Image.open(path).convert('RGBA')
    bb = im.getchannel('A').point(lambda a: 255 if a > 20 else 0).getbbox()
    im = im.crop(bb)
    k = target / max(im.size)
    im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    a = np.asarray(im).astype(float)
    H, W = a.shape[0] // cell, a.shape[1] // cell
    a = a[:H * cell, :W * cell].reshape(H, cell, W, cell, 4).transpose(0, 2, 1, 3, 4).reshape(H, W, cell * cell, 4)
    alpha = a[..., 3] / 255
    keep = alpha.mean(-1) >= alpha_min
    wsum = alpha.sum(-1, keepdims=True).clip(1e-6)
    rgb = (a[..., :3] * alpha[..., None]).sum(-2) / wsum
    ys, xs = np.nonzero(keep)
    pts = np.stack([xs * cell + cell / 2, ys * cell + cell / 2], 1)
    return pts, rgb[ys, xs], (W * cell, H * cell)

def bisect(idx, pts, rng, iters=12):
    p = pts[idx]
    # seed the two centres along the principal axis, then a few Lloyd steps
    c = p.mean(0); u = np.linalg.svd(p - c, full_matrices=False)[2][0]
    proj = (p - c) @ u
    lab = proj > np.median(proj)
    for _ in range(iters):
        if lab.all() or (~lab).all(): break
        c0, c1 = p[~lab].mean(0), p[lab].mean(0)
        new = ((p - c1) ** 2).sum(1) < ((p - c0) ** 2).sum(1)
        if (new == lab).all(): break
        lab = new
    if lab.all() or (~lab).all():
        lab = np.zeros(len(idx), bool); lab[rng.permutation(len(idx))[: len(idx) // 2]] = True
    return idx[~lab], idx[lab]

def build(pts, cols, r_leaf, seed=0):
    rng = np.random.default_rng(seed)
    L = len(pts)
    nodes = [dict(x=round(float(x), 2), y=round(float(y), 2), r=r_leaf, c=[int(round(v)) for v in c], s=1.0)
             for (x, y), c in zip(pts, cols)]
    size = [1] * L
    logL = math.log2(L)
    sys.setrecursionlimit(100000)

    def rec(idx):
        if len(idx) == 1: return int(idx[0])
        a, b = bisect(idx, pts, rng)
        ia, ib = rec(a), rec(b)
        A, B = nodes[ia], nodes[ib]
        n = size[ia] + size[ib]
        wa, wb = size[ia] / n, size[ib] / n
        s = 1 - 0.95 * math.log2(n) / logL + rng.uniform(-0.03, 0.03)
        nodes.append(dict(
            x=round(A['x'] * wa + B['x'] * wb, 2), y=round(A['y'] * wa + B['y'] * wb, 2),
            r=round(math.sqrt(A['r'] ** 2 + B['r'] ** 2), 2),
            c=[int(round(A['c'][i] * wa + B['c'][i] * wb)) for i in range(3)],
            s=round(min(max(s, 0.03), 0.97), 5), ch=[ia, ib], d=round(float(rng.uniform(0.04, 0.07)), 4)))
        size.append(n)
        return len(nodes) - 1

    a, b = bisect(np.arange(L), pts, rng)
    roots = [rec(a), rec(b)]
    return nodes, roots

if __name__ == '__main__':
    src, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
    cell = 5
    pts, cols, (w, h) = leaves_from_image(src, cell)
    nodes, roots = build(pts, cols, r_leaf=round(cell * 0.724, 2))
    roots = sorted(roots, reverse=True)
    json.dump(dict(name=name, image=name + '.png', bounds=dict(w=w, h=h), cell=cell, n_leaves=len(pts),
                   n_nodes=len(nodes), roots=roots, nodes=nodes), open(out, 'w'), separators=(',', ':'))
    print(name, len(pts), 'leaves', len(nodes), 'nodes')

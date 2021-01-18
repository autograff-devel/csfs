# %%

import matplotlib.gridspec as gridspec
import os
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import pdb

import autograff.plut as plut
import autograff.geom as geom
import autograff.utils as utils
import autograff.svg as svg
import autograff.geom.euler_spiral as es
import csfs.path_sym as sym
import csfs.common as cmn

reload(plut)
reload(svg)
reload(sym)

sym.cfg.feature_saliency_thresh = 5e-3  # 5e-3
sym.cfg.refine_clothoid_fit = True
sym.cfg.transition_angle_subd = np.pi * 4/5
# <- total turning angle subdivision for computing transitions
sym.cfg.vma_thresh = 20.
# <- Smoothing (for spline) when computing minima
sym.cfg.minima_smooth_k = 1020
# <- Smoothing (for spline) when computing minima
sym.cfg.minima_smooth_sigma = 30

# test_arc_2.svg'] #, 'wiggle.svg']
paths = ['ellipses.svg', 'ellipses.svg', 'ellipses.svg']
closed = True
angle_subds = [np.pi*4/5]  # sym.COMPUTE_MINIMA] #, sym.COMPUTE_MINIMA]
shapes = []
shape_features = []

for path, params in zip(paths, [dict(noise=0, vma_thresh=0.5, size=150), dict(noise=0, vma_thresh=0.5, size=100), dict(noise=1, vma_thresh=20, size=150)]):
    for i in range(1):  # path, angle_subd in zip(paths, angle_subds):
        sym.cfg.vma_thresh = params['vma_thresh']
        # sym.cfg.transition_angle_subd = angle_subd
        S = cmn.load_svg('../../data/svgs/' + path,
                         params['size'], union=False, closed=closed)
        S = [P + np.random.uniform(-1, 1, size=P.shape)
             * params['noise'] for P in S]
        # , flags=0) #, flags=0) #, flags=0) #, flags=0) #3) #cfg.draw_steps)
        feature_list = sym.compute_shape_features(
            S, closed=closed, n_steps=2, draw_steps=0, flags=1)
        # pdb.set_trace
        # feature_list = sym.compute_transitions(feature_list, S, closed=closed)
        # features = sym.expand_all_anchors(Ps, features, closed)
        sym.compute_CSFs(feature_list, S, closed, compute_axis=False)
        shape_features.append(feature_list)
        shapes.append(S)

print('Computed CSFs')
# pdb.set_trace()
reload(sym)
pad = 20
mr = 12  # marker radius
cyan = plut.colors.cyan

labels = {}


def one_label(txt):
    if txt in labels:
        return ''
    labels[txt] = 1
    return txt


def s0s1_to_kappa(P, a, b, s0, s1):
    d = np.linalg.norm(P[:, a] - P[:, b])
    l = np.sqrt((es.C_(s1) - es.C_(s0))**2 + (es.S_(s1) - es.S_(s0))**2)
    return np.pi * s0 * (l/d),  np.pi * s1 * (l/d),  # abs(sall[1] - sall[0])


def to_kappa(P, f):
    return s0s1_to_kappa(P, *f.anchors, *f.data['s'])

    s0, s1 = f.data['s']
    d = np.linalg.norm(P[:, f.anchors[0]] - P[:, f.anchors[1]])
    l = np.sqrt((es.C_(s1) - es.C_(s0))**2 + (es.S_(s1) - es.S_(s0))**2)
    return np.pi * s0 * (l/d),  np.pi * s1 * (l/d),  # abs(sall[1] - sall[0])
    # return es.t_to_kappa(s0) * (l/d),  es.t_to_kappa(s1) * (l/d),  # abs(sall[1] - sall[0])
 #   return s*s*np.sign(s)*np.pi
    spio2 = np.sqrt(np.pi * .5)
    return s / spio2

    # return s
# %%
reload(plut)
plut.figure_points(300, 250, dpi=200)  # cfg.page_width, 3.)
m = len(shapes)
pad = 10
gs = gridspec.GridSpec(m, 1)

# Show two complete CSFs, one with tangents
for i in range(len(shapes)):
    feature_list = shape_features[i]
    S = shapes[i]
    #closed = shape_closed[i]

    plt.subplot(gs[i, 0])
    plut.figure_label(['a', 'b', 'c'][i], loc='left')
    # plut.boldtitle(titles[i])
    plut.stroke_shape(S, 'k', closed=closed, linewidth=0.5)

    k = 0
    for features, P in zip(feature_list, S):
        for f in features:
            if sym.is_endpoint(f):
                continue
            sym.draw_CSF(f, plut.default_color(
                0), linewidth=0.5, draw_axis=True)
            k += 1
    plut.setup(axis_limits=geom.bounding_box(S, pad))

plt.tight_layout()
# cfg.save_thesis_figure('ellipses.pdf','csfs')
# plut.save_tight(os.path.join(output_dir,'B.pdf'))
plt.show()


# %%


def figure_points(w, h, dpi):
    fig = plt.figure(dpi=dpi)
    fig.set_size_inches(w/72, h/72)


figure_points(360, 100, dpi=150)
plut.figure_label('a')
plt.plot(range(10), range(10))
plt.tight_layout()
plt.savefig('150.pdf')
plt.show()  # or 'plt.savefig('image.png')', depending on your matplotlib backend

figure_points(360, 100, dpi=200)
plut.figure_label('a')
plt.plot(range(10), range(10))
plt.tight_layout()
plt.savefig('200.pdf')
plt.show()  # or 'plt.savefig('image.png')', depending on your matplotlib backend

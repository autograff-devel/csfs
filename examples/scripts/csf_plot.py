# %%
'''
Plot CSFs for one or more inputs
usage
python csf_plot --input='path_to_file_or_directory
See below for additional flags
you can set these with the --flags arg
For example
--flags='km'
Will plot the curvature reconstruction together with estimated inflections (k)
and compute absolute minima CSFs (m)
'''
from matplotlib.gridspec import GridSpec
import os, sys
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import pdb
import argparse

import module_paths

import autograff.plut as plut
reload(plut)
import autograff.geom as geom
import autograff.utils as utils
import autograff.svg as svg
import autograff.geom.euler_spiral as es

import csfs.config as config
reload(config)

import csfs.path_sym as sym
reload(sym)

cfg = sym.cfg

import csfs.common as cmn
reload(cmn)

import global_cfg


print(sys.argv)
args = argparse.ArgumentParser(description='''CSF analysis scripts command line arguments''')
# hanzi_test_params_refined
args.add_argument('-i', '--input', type=str, default='./../../data/bsplines/closed/c03.pkl',
                 help='''Determines the input directory or file to process''')
args.add_argument('-o', '--output', type=str, default='./csf_plot.pdf',
                 help='''Determines output image location''')
args.add_argument('--chars', type=str, default='A', help='''Characters to process if input is a font''')
args.add_argument('--closed', type=bool, default=True,
                  help='''Determines if input is closed or not,
                  Note that if the input path contains the word "closed" or "open", this argument will have no effect ''')

args.add_argument('--flags', type=str, default='',
                 help='''Flags:
                 k: Reconstructs and plots curvature
                 i: Identifies inflections (if k is set)
                 m: Computes minima
                 ''')

args.add_argument('--draw', type=str, default='aOoecs',
                 help='''Flags:
                 a: Computes and plots CSF local axes
                 O: Draw osculating circle
                 o: Draw osculating circle center
                 e: Draw extremum
                 c: Draw contact region
                 s: Draw support segments
                 ''')

args.add_argument('--noise_sigma', type=float, default=0.,
                  help='''Optional artificial noise.
                  Note that this will require tweaking (increasing) --vma_thresh to get sensible results''')

args.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

#args.add_argument('--save_figures', type=bool, default=True,
#                  help='''If true saves figures''')
# Add config args
for key, val in cfg.__dict__.items():
    print('addding ' + '--' + key)
    docstr = key + ': custom parameter'
    if key in cfg.doc:
        docstr = cfg.doc[key]
    args.add_argument('--' + key, type=type(val), default=val, help=docstr)
# Parse
args_cfg = args.parse_args()

# Update parameters if specified
for key, val in args_cfg.__dict__.items():
    #if not key in cfg.__dict__:
    print('setting: ' + key + ': ' + str(val))
    cfg.__dict__[key] = val


if os.path.isdir(cfg.input):
    paths = utils.files_in_dir(cfg.input)
else:
    paths = [cfg.input]

#cfg.straightness_tolerance = 0.2
#cfg.minima_saliency_thresh = 1e-6  # <- Minimum CSF saliency
#cfg.smoothing = 3.5                 # <- input smoothing window
#cfg.curvature_smoothing = 3
# Load input shapes
input_shapes = []
closed_flags = []
for path in paths:
    # If specified in filename set closed or open flag, otherwise use user set default
    if 'closed' in path or '.jpg' in path or '.png' in path:
        closed_flags.append(True)
    elif 'open' in path:
        closed_flags.append(False)
    else:
        closed_flags.append(cfg.closed)
    print('Loading ' + path)
    S = cmn.load(path, cfg.shape_size, closed_flags[-1], cfg.chars)
    print('Done')
    if type(S[0]) == list:
        input_shapes += S
        for s in S[1:]:
            closed_flags.append(closed_flags[-1])
    else:
        input_shapes.append(S)

ncols = 1
rowh = 200
if 'k' in cfg.flags:
    ncols = 2
    rowh = 140
plut.set_theme()
plut.figure_points(360, rowh*len(input_shapes), dpi=200)

if 'k' in cfg.flags:
    gs = GridSpec(len(input_shapes), 2, width_ratios=[0.3,0.7])
else:
    gs = GridSpec(len(input_shapes), 1)

draw_flags_map = {
    'axis':'a',
    'osculating':'O',
    'osculating_center':'o',
    'extremum':'e',
    'contact':'c',
    'support':'s'
}
draw_flags = {}
for key,c in draw_flags_map.items():
    if c in cfg.draw:
        draw_flags[key] = True
    else:
        draw_flags[key] = False
        
for i, S in enumerate(input_shapes):
    S = [P + np.random.uniform(-1, 1, size=P.shape)*cfg.noise_sigma
            for P in S]

    if cfg.smoothing > 0:
        print('Applying smoothing')
        S = [geom.gaussian_smooth_contour(P, cfg.smoothing) for P in S]
        
    flags = 0
    if 'm' in cfg.flags:
        flags = 1
    print("Computing all features")
    feature_list = sym.compute_shape_features(
        S, closed=closed_flags[i], n_steps=10, draw_steps=0, flags=flags)
    if 'k' in cfg.flags:
        feature_list = sym.compute_transitions(feature_list, S, closed=closed_flags[i])
        plt.subplot(gs[i, 1])
        inflections = sym.reconstruct_curvature(S, feature_list, closed=closed_flags[i], plot=True, lw=0.5)
    else:
        inflections = []

    plt.subplot(gs[i, 0])
    
    print("Computing CSF data")
    sym.compute_CSFs(feature_list, S, closed_flags[i], compute_axis='a' in cfg.draw)
    print("Done")


    #plut.figure_label(['a', 'b', 'c'][i], loc='left')
    # plut.boldtitle(titles[i])
    plut.stroke_shape(S, 'k', closed=closed_flags[i], linewidth=0.5)

    if 'k' in cfg.flags:
        sym.draw_reconstruction(feature_list, S, closed_flags[i], linewidth=1.5)
    else:
        k = 0
        for features, P in zip(feature_list, S):
            sym.draw_CSFs(features, offset=1, draw_flags=draw_flags, only=[0, 3])
        # for f in features:
        #     if sym.is_en
        #     dpoint(f):
        #         continue
        #     sym.draw_CSF(f, plut.default_color(
        #         0), linewidth=0.5, draw_axis='a' in cfg.flags)
        #     k += 1
    for P, I in zip(S, inflections):
        for j, (i, Pp) in enumerate(I):
            plut.draw_marker(P[:,i], sym.feature_colors[sym.FEATURE_INFLECTION], markersize=3, marker='x')

    plut.setup(axis_limits=geom.bounding_box(S, 10))

plt.savefig('csf_plot.pdf')
plt.show()

        
# sym.cfg.feature_saliency_thresh = 5e-3  # 5e-3
# sym.cfg.refine_clothoid_fit = True
# sym.cfg.transition_angle_subd = np.pi * 4/5
# # <- total turning angle subdivision for computing transitions
# sym.cfg.vma_thresh = 20.
# # <- Smoothing (for spline) when computing minima
# sym.cfg.minima_smooth_k = 1020
# # <- Smoothing (for spline) when computing minima
# sym.cfg.minima_smooth_sigma = 30

# # test_arc_2.svg'] #, 'wiggle.svg']
# paths = ['ellipses.svg', 'ellipses.svg', 'ellipses-noise.svg']
# closed = True
# angle_subds = [np.pi*4/5]  # sym.COMPUTE_MINIMA] #, sym.COMPUTE_MINIMA]
# shapes = []
# shape_features = []

# for path, params in zip(paths, [dict(noise=0, vma_thresh=0.5, size=150), dict(noise=0, vma_thresh=0.5, size=100), dict(noise=1, vma_thresh=20, size=150)]):
#     for i in range(1):  # path, angle_subd in zip(paths, angle_subds):
#         sym.cfg.vma_thresh = params['vma_thresh']
#         # sym.cfg.transition_angle_subd = angle_subd
#         S = cmn.load_svg('../../data/svgs/' + path,
#                          params['size'], union=False, closed=closed)
#         S = [P + np.random.uniform(-1, 1, size=P.shape)
#              * params['noise'] for P in S]
#         # , flags=0) #, flags=0) #, flags=0) #, flags=0) #3) #cfg.draw_steps)
#         feature_list = sym.compute_shape_features(
#             S, closed=closed, n_steps=2, draw_steps=0, flags=1)
#         # pdb.set_trace
#         # feature_list = sym.compute_transitions(feature_list, S, closed=closed)
#         # features = sym.expand_all_anchors(Ps, features, closed)
#         sym.compute_CSFs(feature_list, S, closed, compute_axis=False)
#         shape_features.append(feature_list)
#         shapes.append(S)

# print('Computed CSFs')
# # pdb.set_trace()
# reload(sym)
# pad = 20
# mr = 12  # marker radius
# cyan = plut.colors.cyan

# labels = {}


# def one_label(txt):
#     if txt in labels:
#         return ''
#     labels[txt] = 1
#     return txt


# def s0s1_to_kappa(P, a, b, s0, s1):
#     d = np.linalg.norm(P[:, a] - P[:, b])
#     l = np.sqrt((es.C_(s1) - es.C_(s0))**2 + (es.S_(s1) - es.S_(s0))**2)
#     return np.pi * s0 * (l/d),  np.pi * s1 * (l/d),  # abs(sall[1] - sall[0])


# def to_kappa(P, f):
#     return s0s1_to_kappa(P, *f.anchors, *f.data['s'])

#     s0, s1 = f.data['s']
#     d = np.linalg.norm(P[:, f.anchors[0]] - P[:, f.anchors[1]])
#     l = np.sqrt((es.C_(s1) - es.C_(s0))**2 + (es.S_(s1) - es.S_(s0))**2)
#     return np.pi * s0 * (l/d),  np.pi * s1 * (l/d),  # abs(sall[1] - sall[0])
#     # return es.t_to_kappa(s0) * (l/d),  es.t_to_kappa(s1) * (l/d),  # abs(sall[1] - sall[0])
#  #   return s*s*np.sign(s)*np.pi
#     spio2 = np.sqrt(np.pi * .5)
#     return s / spio2

#     # return s
# # %%
# reload(plut)
# plut.figure_points(300, 250, dpi=200)  # cfg.page_width, 3.)
# m = len(shapes)
# pad = 10
# gs = gridspec.GridSpec(m, 1)

# # Show two complete CSFs, one with tangents
# for i in range(len(shapes)):
#     feature_list = shape_features[i]
#     S = shapes[i]
#     #closed = shape_closed[i]

#     plt.subplot(gs[i, 0])
#     plut.figure_label(['a', 'b', 'c'][i], loc='left')
#     # plut.boldtitle(titles[i])
#     plut.stroke_shape(S, 'k', closed=closed, linewidth=0.5)

#     k = 0
#     for features, P in zip(feature_list, S):
#         for f in features:
#             if sym.is_endpoint(f):
#                 continue
#             sym.draw_CSF(f, plut.default_color(
#                 0), linewidth=0.5, draw_axis=True)
#             k += 1
#     plut.setup(axis_limits=geom.bounding_box(S, pad))

# plt.tight_layout()
# # cfg.save_thesis_figure('ellipses.pdf','csfs')
# # plut.save_tight(os.path.join(output_dir,'B.pdf'))
# plt.show()


# # %%


# def figure_points(w, h, dpi):
#     fig = plt.figure(dpi=dpi)
#     fig.set_size_inches(w/72, h/72)


# figure_points(360, 100, dpi=150)
# plut.figure_label('a')
# plt.plot(range(10), range(10))
# plt.tight_layout()
# plt.savefig('150.pdf')
# plt.show()  # or 'plt.savefig('image.png')', depending on your matplotlib backend

# figure_points(360, 100, dpi=200)
# plut.figure_label('a')
# plt.plot(range(10), range(10))
# plt.tight_layout()
# plt.savefig('200.pdf')
# plt.show()  # or 'plt.savefig('image.png')', depending on your matplotlib backend

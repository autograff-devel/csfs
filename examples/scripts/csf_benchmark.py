#%%
import os
import time
from importlib import reload
import module_paths
import autograff.plut as plut
import autograff.geom as geom
import autograff.utils as utils
import autograff.svg as svg
reload(svg)
import autograff.geom.euler_spiral as es
import matplotlib.pyplot as plt
import numpy as np
import pdb
import csfs.path_sym as sym
reload(sym)
import csfs.common as cmn
import global_cfg as cfg # Global script configurations
reload(cfg)

reload(plut)

import os
import time

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


sym.cfg.feature_saliency_thresh = 5e-3 #5e-3
sym.cfg.refine_clothoid_fit = True
sym.cfg.transition_angle_subd = np.pi * 4/5 # <- total turning angle subdivision for computing transitions
sym.cfg.vma_thresh = 0.5
sym.cfg.minima_smooth_k = 1020           # <- Smoothing (for spline) when computing minima
sym.cfg.minima_smooth_sigma = 30           # <- Smoothing (for spline) when computing minima

paths = ['bsmall.svg', 'b-script.svg']
closeds= [True, False]
angle_subds = [np.pi*4/5] #sym.COMPUTE_MINIMA] #, sym.COMPUTE_MINIMA]
shapes = []
shape_features = []

sizes = 50 + np.linspace(0, 4, 5)*100 #np.linspace(0, 9, 5)*50

times_csfs = []
times_recons = []
times_total = []
npoints = []

shapes_saved = []
features_saved = []

for closed, path in zip(closeds, paths):
    times_csfs.append([])
    npoints.append([])
    print('Computing ' + path)
    for i, size in enumerate(sizes):#[150]: #sizes: #sizes:
        print('computing CSFs for ' + str(size))
        plut.figure_points(200,200)
        S = cmn.load_svg('../../data/svgs/' + path, size, union=False, closed=closed)
        plut.stroke_shape(S, 'k', closed=closed)
        with Timer() as tcsf:
            feature_list = sym.compute_shape_features(S, closed=closed, n_steps=2, draw_steps=2, flags=0) #, flags=0) #, flags=0) #, flags=0) #, flags=0) #3) #cfg.draw_steps)

        if i == 1:
            shapes_saved.append(S)
            features_saved.append(feature_list)
            
        times_csfs[-1].append(tcsf.interval)
        npoints[-1].append(np.sum([P.shape[1] for P in S]))

        # #pdb.set_trace
        # print('reconstructing')
        # with Timer() as trecons:
        #     feature_list = sym.compute_transitions(feature_list, S, closed=closed)
        # times_recons.append(trecons.interval)
        #times_total.append(times_recons[-1] + times_csfs[-1])
        #features = sym.expand_all_anchors(Ps, features, closed)
        sym.compute_CSFs(feature_list, S, closed, compute_axis=False)

        for features, P in zip(feature_list, S):
            for f in features:
                if sym.is_endpoint(f):
                    continue
                sym.draw_CSF(f, plut.default_color(0), linewidth=0.5, draw_axis=True)
                #k += 1
        plut.setup()
        plt.show()

#%%

def double_axis(ax1, i):
    ax1.set_xlabel('# Points', labelpad=1.5, fontsize=5)
    ax1.set_ylabel('Time (seconds)', labelpad=1)
    ax1.tick_params(axis='y', pad=-2)
    #ax1.legend()
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["bottom"].set_position(("axes", 0.0))
    ax1.tick_params(axis='x', labelsize=5 )
    ax1.tick_params(axis='y', labelsize=5 )
    ax1.tick_params(axis='x', pad=-2)

    ax2 = ax1.twiny()
    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("axes", -0.4))

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    #ax1.set_frame_on(True)
    #ax2.patch.set_visible(False)
    # for sp in ax2.spines.itervalues():
    #     sp.set_visible(False)
    #x2 = sizes[i] # list(range(0, 100, n))
    ax2.spines["bottom"].set_visible(True)
    ax2.set_xticks(npoints[i]) #[i]) #x2[:n][::2])
    #
    #ax2.set_xlim([x2[0], x2[-1]]) #ax1.get_xlim())
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticklabels(['%d'%int(j) for j in sizes]) #tick_function(new_tick_locations))
    ax2.set_xlabel('Height', labelpad=1.5, fontsize=5)
    ax2.tick_params(axis='x', labelsize=5 )

plut.figure_points(300,50)
from matplotlib.gridspec import GridSpec
w = 0.8
gs = GridSpec(1, 4, width_ratios=[w, 1, w, 1])

for i in range(2):
    plt.subplot(gs[0, i*2])
    plut.boldtitle(['(a)', '(b)'][i], loc='left')
    plut.stroke_shape(shapes_saved[i], 'k', closed=closeds[i], linewidth=0.25)
    for features, P in zip(features_saved[i], shapes_saved[i]):
        for f in features:
            if sym.is_endpoint(f):
                continue
            sym.draw_CSF(f, plut.default_color(0), linewidth=0.5, draw_axis=True)
    plut.setup()

    ax = plt.subplot(gs[0, i*2+1])
    plt.plot(npoints[i], times_csfs[i])
    double_axis(ax, i)
#plt.tight_layout()
cfg.save_thesis_figure('benchmark.pdf', 'csfs')
plt.show()
#%%
draw = True

if draw:
    mfig = plut.MultiFigure()
    mfig.begin()

rows = 2
test_cols = list(range(2, 15))
layering_times = []
softie_times = []
global_times = []
num_segments = []
num_points = []

for i, periodic in enumerate([False, True]):
    global_times.append([])
    softie_times.append([])
    layering_times.append([])
    num_segments.append([])
    num_points.append([])
    for cols in test_cols:
        set_seed(33)
        P, depths = compute_lattice(rows, cols)
        num_segments[-1].append(P.shape[1]-1)
        print('%d cols'%(cols))
        with Timer() as global_t:
            print('softie')
            with Timer() as softie_t:
                softie, softie_shape = compute_softie(P, periodic=periodic)

            if draw:
                mfig.add_subplot()
                plut.plot(P, 'r:')
                plut.stroke_shape(softie_shape, 'k', linewidth=0.25, alpha=0.4)

            softie_times[-1].append(np.array(softie_t.interval))
            num_points[-1].append(np.sum([sp.shape[1] for sp in softie_shape]))

            print('layering')
            with Timer() as layer_t:
                outlines, whole = compute_layering(softie, depths)
            layering_times[-1].append(np.array(layer_t.interval))

            if draw:
                #plut.plot(P, 'r:')
                plut.stroke_shape(outlines, 'k', closed=False)

        global_times[-1].append(np.array(global_t.interval))

if draw:
    mfig.end()
    
#%%
# https://pythonmatplotlibtips.blogspot.com/2018/01/add-second-x-axis-below-first-x-axis-python-matplotlib-pyplot.html
plut.set_theme()

plut.figure_points(360, 180) #(figsize=(5.6,4))
#
for i in range(2):
    ax1 = plt.subplot(1,2,i+1)
    plt.title(['Non periodic', 'Periodic'][i])

    n = len(num_segments[i])

    plot_segs = True
    if plot_segs:
        x = np.array(num_segments[i])
        xlab = 'Num Segments'
        x = np.array(num_points[i])
        xlab = '# Points'
    else:
        x = test_cols
        xlab = 'Num. columnns'

    ax1.plot(x[:n], np.array(softie_times[i][:n])+np.array(layering_times[i][:n]), label='Global')
    ax1.plot(x[:n], softie_times[i][:n], label='Curve Generation')
    ax1.plot(x[:n], layering_times[i][:n], label='Layering')

    ax1.set_xlabel(xlab)
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()

    ax2 = ax1.twiny()
    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("axes", -0.3))

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    #ax1.set_frame_on(True)
    #ax2.patch.set_visible(False)
    # for sp in ax2.spines.itervalues():
    #     sp.set_visible(False)
    x2 = num_segments # list(range(0, 100, n))
    ax2.spines["bottom"].set_visible(True)
    ax2.set_xticks(x[:n][::2]) #x2[:n][::2])
    #
    #ax2.set_xlim([x2[0], x2[-1]]) #ax1.get_xlim())
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticklabels(['%d'%i for i in num_segments[i][:n][::2]]) #tick_function(new_tick_locations))
    ax2.set_xlabel('# Spine Segments')

#ax2.plot(x2, np.zeros(n))       
plt.tight_layout()
import global_cfg as cfg
cfg.save_thesis_figure('timing.pdf', 'chunks')
#ax2.xticks(num_segments)
#plt.savefig('timing.pdf')
plt.show()

#%%

    
cols = 5


softie, softie_shape = compute_softie(P)
outlines, whole = compute_layering(softie, depths)

plut.figure(cols*2,rows*2)
plut.stroke_poly(P, 'r', closed=False, linestyle=':')
plut.stroke_shape(softie_shape, 'k', linewidth=0.5, linestyle=':')
plut.stroke_shape(outlines, 'k', closed=False, linewidth=2.)
plut.stroke_shape(whole, 'k', linewidth=2.)
plut.show()

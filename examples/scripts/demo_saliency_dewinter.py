#%%
import os
from importlib import reload
from collections import defaultdict
import autograff.plut as plut
import autograff.geom as geom
import autograff.utils as utils
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
import autograff.dataset as dataset
import os

import csfs.path_sym as sym
reload(sym)
reload(plut)

def normalize(x, lo=0.0, hi=1.):
    return x/np.max(x)
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    return lo + x * (hi-lo)

from scipy.ndimage.filters import gaussian_filter1d

def saliency_freq(salient, n, sigma, thresh=2): #3):
    ''' Smoothed saliency frequency as described in DeWinters and Wagemans'''
    #freq = np.histogram(v, n-1, density=False)[0].astype(float)
    freq = np.zeros(n)
    for v in salient:
        if v > -1:
            freq[v] += 1

    if sigma > 0.:
        freq = gaussian_filter1d(np.array(freq), sigma=sigma, mode='wrap')
    freq[np.where(freq<thresh)] = 0. #= np.max
    return freq

# 42 hat
# 43 camel -> problem with minima sign
# 258 - cup
# 49 cat
# 73 dog
# 119 heart
# 121 horse
# 246 vase
#
#

from matplotlib.colors import Normalize
import matplotlib.cm as cm


stimuli = [246, 49]
nstimuli = len(stimuli)
reload(sym)
sym.cfg.minima_smooth_k = 2
sym.cfg.minima_expansion_tol = 1
sym.cfg.minima_saliency_thresh = 1e-100
sym.cfg.vma_thresh = 0.3
sym.cfg.feature_saliency_thresh = 1e-8
sym.cfg.merge_thresh = 0.98
sym.cfg.straightness_tolerance = 0.001
sym.cfg.max_radius_height_ratio = 20
#sym.cfg.r_thresh = 11200. # maximum accepted disk radius  (gets overridden in CASA for relative size)
sym.cfg.minima_r_thresh = 12000
sym.cfg.only_simple_areas = False

closed = True

plut.set_theme()
plut.figure_points(360, 150, dpi=200)


import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, nstimuli*2+1, width_ratios = list(np.ones(nstimuli*2))+[0.1], height_ratios=[1.,0.3])

from autograff.algorithms import dtw

for i, stimulus_ind in enumerate(stimuli):
    #plt.subplot(1, , i*2+1)
    plt.subplot(gs[0,i*2])
    #plut.boldtitle(['(a)', '(b)'][i], loc='left')
    plt.title('Frequencies')
    #stimulus_ind = 246 #258 # Note not all stimuli from the DeWinters and Wagmeans dataset are present in the folder, these can be downloaded at
    stim = dataset.dewinter_load_stimulus('../../data/dewinters', stimulus_ind)
    print('loading stimulus ' + str(stimulus_ind))
    flip_flags = []
    P = geom.fix_shape_winding(stim['path'], cw=True, flip_flags=flip_flags)
    #[P], ratio = sym.preprocess_shape([P], closed, 100, smooth_sigma=0.8, get_ratio=True, pre_smooth=True, resample=True)
    P = geom.gaussian_smooth_contour(P, 2)
    [P], ratio = geom.rescale_and_sample_vertical([P], closed=closed, height=0, dest_height=150, get_ratio=True)

    features = sym.compute_features(P, closed=closed, n_steps=3, draw_steps=0, flags=sym.COMPUTE_MINIMA) #, flags=0) #3) #cfg.draw_steps)
    sym.compute_CSFs(features, P, closed=closed, compute_axis=False)
    # Wagemans stimulus saliency
    salient = stim['salient']
    Psalient = stim['path']*ratio
    n = Psalient.shape[1]

    if flip_flags[0]:
        salient = [n-1-k if k > -1 else k for k in salient]
        #salient = salient[::-1]
        Psalient = Psalient[:, ::-1]

    n = Psalient.shape[1]

    print('computed features')

    saliency = saliency_freq(salient, n, sigma=4, thresh=0.0)

    so = saliency
    saliency = normalize(saliency)

    colorbar = plut.ColorBar(saliency, vmin=0, vmax=1)

    scalefac = 30
    #scalefac = 30

    participant_color = [0.7, 0.7, 0.7]
    auto_color = [1., 0.0, 0.] #[0.1, 0.9, 1.]
    idx = saliency.argsort()

    saliency_unsorted = saliency
    Psalient_unsorted = Psalient

    Psalient = Psalient[:,idx]

    saliency = saliency[idx]
    
    #x, y, z = x[idx], y[idx], z[idx]

    # Align dataset trajectory and sampled one with DTW
    _, _, path = dtw(P, Psalient_unsorted)
    P_to_salient = {}
    for a, b in path:
        P_to_salient[a] = b

    plut.stroke_poly(P, 'k', closed=closed, alpha=1., linewidth=0.5)
    # NOTE scatter size must be squared.... just because.. wtf matplotlib
    plt.scatter(Psalient[0,:], Psalient[1,:], s=(saliency**2)*scalefac, c=saliency, cmap=colorbar.cmap, alpha=1., label='participant saliency')
    plut.setup(axis_limits=geom.bounding_box(P, 20))

    # CSFS
    plt.subplot(gs[0, i*2+1])
    plt.title('CSF saliency')
    plut.stroke_poly(P, 'k', closed=closed, alpha=1., linewidth=0.5)
    sym.draw_CSFs(features, clr=np.ones(3)*0.7, linewidth=0.5, draw_axis=False, count=0,
                  draw_flags={'support':False,
                              'contact':False})

    fsaliency = np.array([f.data['saliency'] for f in features])
    fsaliency = normalize(fsaliency)

    csf_saliency_map = []
    csf_pts = []
    for j, f in enumerate(features):
        fr = 4
        csf_pts.append(P[:,f.i])
        #plut.fill_poly(f.data['area'], np.ones(3)*0.5, alpha=0.5)
        csf_saliency_map.append([P_to_salient[f.i], fsaliency[j], int(sym.is_extremum(f))])

    csf_pts = np.array(csf_pts).T
    plt.scatter(csf_pts[0], csf_pts[1], s=(np.maximum(fsaliency, 0.3)**2)*scalefac, c=fsaliency, cmap=colorbar.cmap, alpha=1., label='participant saliency')
    plut.setup(axis_limits=geom.bounding_box(P, 20))

    plt.subplot(gs[1, i*2:i*2+2])
    #plut.colors.cyan
    #plt.plot(list(range(len(saliency_unsorted))), (saliency_unsorted)) #/(saliency_unsorted).max())
    plut.plot_filled_positive(list(range(len(saliency_unsorted))), (saliency_unsorted), clr=np.ones(3)*0.5, alpha=0.5) #, edgecolor=np.ones(3)*0.4)
    for k, sal, extr in csf_saliency_map:
        clr = 'r'
        if not extr:
            clr = [0, 0.4, 1.]
        plut.draw_marker([k, sal], color=clr, marker='o')
        plut.draw_line([k, sal], [k, 0], np.ones(3)*0.5, linewidth=0.25)

    plt.xlabel('Sample #')
plt.subplot(gs[0, -1])
plt.colorbar(colorbar.m, fraction=0.5, pad=0.05)
plt.axis('off')

plt.tight_layout()
cfg.save_thesis_figure('dewinter_2.pdf', 'csfs')
plt.show()
#%%

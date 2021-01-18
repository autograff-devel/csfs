#%%
import os
from importlib import reload
import module_paths
import autograff.plut as plut
import autograff.geom as geom
import autograff.utils as utils
import autograff.symmetry.voronoi_skeleton as vma
import autograff.symmetry.path_sym as sym

from autograff.numeric import gaussian_smooth

import matplotlib.pyplot as plt
import numpy as np
import common as com

import casa
import common as cmn
reload(cmn)
reload(casa)
import cfg # Global script configurations
reload(cfg)

import csfs.casa as casa
import strokestyles.common as cmn
reload(oa)
reload(plut)

casa.cfg.vma_thresh = 0.5
casa.cfg.compute_CSF_axes = True
casa.cfg.anchor_expansion_tol = 0.2 #5

import os

output_dir = utils.create_dir(os.path.join(cfg.figure_dir, 'demo_results'))

closed = True
pts = utils.load_pkl('../data/bsplines/closed/c04.pkl')
P = geom.bspline(400, pts, ds=2., closed=closed)


plut.figure_points(cfg.page_width, 100, dpi=200)
plut.stroke_poly(P, 'k')
plut.stroke_poly(Ps, 'r')

plut.show()
#%%
#pts = utils.load_pkl('./bsplines/02.pkl')
#closed = False
ds = 2.
n = 400


#P = geom.ellipse_points(np.linspace(0, np.pi*2, n), 100, 60)
#evo = geom.ellipse_evolute( np.linspace(0, np.pi*2, n), 100, 60)

P = geom.bspline(n, pts, ds=ds, closed=closed)
evo = geom.bspline_evolute(n, pts, ds=ds, closed=closed)

closed = True
evo = None
#P = load_contour('./pkls/201.pkl', closed=closed, smooth_sigma=0.6)
stimulus_ind = 258 #42 #258
stim = dataset.dewinter_load_stimulus('/Users/colormotor/Dropbox/data/databases/dewinters_contour', stimulus_ind)
P = geom.fix_shape_winding(stim['path'], cw=True)
[P], ratio = preprocess_shape([P], closed, 130, 0.8, get_ratio=True, pre_smooth=True, resample=True)

#      
plt.figure(figsize=(6,5))
plt.title('B-Spline')
if evo is not None:
    plt.plot(evo[0,:], evo[1,:], 'r:', linewidth=0.4)
plt.plot(P[0,:], P[1,:], 'k')
plut.draw_marker(P[:,0], 'ro')
plut.setup(axis=False)
plut.set_axis_limits(P, 5)


plt.show()

#%%

from sympy import *
init_printing()
n = symbols('n')

simplify((n*log(n))*(log(n)))

#%%

def O1(n):
    return n*np.log(n)

def O2(n):
    res = O1(n)
    if n > 1:
        res += O1(n/2) + O1(n/2)
    return res

def O3(n):
    return n*(np.log(n**2))

N = np.linspace(1, 20000, 100)
o = [O1(n) for n in N]
o2 = [O2(n) for n in N]
o3 = [O3(n) for n in N]
plut.figure_points(300,300)
plt.plot(N, o, label='o1')
plt.plot(N, o2, label='o2')
plt.plot(N, o3, label='o3')
plt.plot(N, N, label='N')
plt.legend()
plt.show()

#%%
from importlib import reload
import time, copy, os, sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MACOSX')

# Add module paths, modify module_paths.py to adjust 
import module_paths

import autograff.svg as svg
import autograff.plut as plut
import autograff.geom as geom
import autograff.utils as utils
import csfs.voronoi_skeleton as vma
reload(utils)

import csfs.config as config
reload(config)

import csfs.common as cmn
reload(cmn)
from csfs.common import *
import csfs.path_sym as sym
reload(sym)
import csfs.casa as casa
reload(casa)

# Common argument parsing
import global_args
reload(global_args)

def draw_CSFs(features, closed=True, only=[], draw_axis=False):

    signs = [-1, 1]
    for i, f in enumerate(features): #f in features:
        #vma.draw_skeleton(CSF['MA'])
        if only and i not in only:
            continue
        
        contact = f.contact
        offset = 0 #signs[i%2]*14.
        clr_ind = i
        if only:
            clr_ind = [0,2,1][i]
            
        clr = plut.default_color(clr_ind)
        casa.draw_CSF(f, clr, offset)
        
sym.cfg.vma_thresh = 0.5                # <- Voronoi skeleton reg. threshold
sym.cfg.straightness_tolerance = 1      # <- Straight segment tolerance (multiple of ds)
sym.cfg.merge_thresh = 0.99             # <- disk overlap threshold
sym.cfg.minima_smooth_k = 30           # <- Smoothing (for spline) when computing minima
sym.cfg.feature_saliency_thresh = 5e-3  # <- Minimum CSF saliency
sym.cfg.casa_has_minima = True
sym.cfg.minima_saliency_thresh = 5e-3  # <- Minimum CSF saliency
casa.cfg.max_radius_height_ratio = 2.
casa.cfg.draw_steps = 0
sym.cfg.straightness_tolerance = 0.5
sym.cfg.minima_expansion_tol = 0.01     # <- annulus for minima

plut.set_theme()

def run():
    cfg = config.cfg

    # output images
    img_output_dir = utils.create_dir(os.path.join(cfg.output, 'images/csfs')) 

    if cfg.type == 'svg':
        src = SvgIterator(cfg.input_path, cfg.flip_y)
        iterator = src.iterate(paths=cfg.items, union=True, size=cfg.shape_size)
    elif cfg.type == 'struct':
        src = ShapeStructureIterator(cfg.input_path, cfg.flip_y)
        iterator = src.iterate(paths=cfg.items, union=True, size=cfg.shape_size)
    else:
        src = FontIterator(cfg.input_path)
        iterator = src.iterate(fonts=cfg.items, chars=cfg.chars, char_map=cfg.char_map, size=cfg.shape_size)
        
    def plot_csfs(gp, S, name, char, glyph_data):
        ''' Callback from GridPlot'''
        
        MA, MA_exterior, features = casa.compute_skeleton_and_features(S)
        CASA = casa.compute_casa(MA, features)

        # # Draw CSFs
        gp.begin_draw('MA')
        casa.draw_shape(S)
        casa.draw_skeleton(MA, 'b', linewidth=0.5)
        casa.draw_skeleton(MA_exterior, 'r', linewidth=0.5)
        gp.end_draw(padding=5)

        gp.begin_draw('CSFs')
        casa.draw_shape(S)
        casa.draw_skeleton(CASA)
        draw_CSFs(CASA.graph['features'])
        gp.end_draw(padding=5)
    #endf

    print("Running draw csfs")
    gp = GridPlot(iterator, plot_csfs, figsize=(2.,2.))
    save_path = ''
    if cfg.save_figures:
        save_path = img_output_dir
    gp.run(save_path, cfg.start_from)

if __name__ == '__main__':
    global_args.parse()
    run()
    

# %%

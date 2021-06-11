'''Public parameter config'''
from importlib import reload
import autograff.utils as utils
reload(utils)
import numpy as np

#cfg = lambda : None
cfg = utils.Config()

cfg.add("tangent_sleeve_tolerance", 2., "tangent approximation tolerance, multiple of ds")
cfg.add("max_radius_height_ratio", 0.5, "maximum feature radius for abs maxima (relative to bbox)")
cfg.add("vma_thresh", 0.5, "Voronoi skeleton reg. threshold")
cfg.add("straightness_tolerance", 0.2, "Straight segment tolerance (multiple of ds)")
cfg.add("merge_thresh", 0.98, "disk overlap threshold")
cfg.add("minima_smooth_k", 100., "Smoothing (for spline) when computing minima")
cfg.add("minima_smooth_sigma", 1, "unused")
cfg.add("feature_saliency_thresh", 5e-3, "Minimum CSF saliency")
cfg.add("minima_saliency_thresh", 1e-6, "Minimum CSF saliency")
cfg.add("use_area_saliency", False, "If true, uses area based saliency")

cfg.add("transition_angle_subd", np.pi * 4/5, "total turning angle subdivision for computing transitions")
cfg.add("spiral_subdivision", 70, "Number of points per spiral segment (for transition fitting proc.)")

cfg.add("curvature_smoothing", 0., "Digital curvature smoothing when reconstructing.")
cfg.add("subdivision_smooth_sigma", 0.5, "Transition segment smoothing before subdivision")

cfg.add("corner_radius_thresh", 2., "corner radius threshold")
cfg.add("compute_CSF_axes", True, "If True, compute local axes")
cfg.add("num_sym_passes", 3, "Max passes for csf ")
cfg.add("debug_draw", False, "debug draw")

cfg.add("smoothing", 0.0, "input smoothing window")
cfg.add("anchor_expansion_tol", 0.0, "annulus tolerance for terminal disks (was 0.05). See path_sym.expand_all_anchors for notes. Should be zero, or need corner detection")
cfg.add("minima_expansion_tol", 0.05, "annulus for minima")

cfg.add("refine_clothoid_fit", True, "Least squares clothoid fit refinement")

cfg.add("draw_steps", False, "Debug draw CSF estimation steps")
cfg.add("debug_feature_support", True, "...")

cfg.add("shape_size", 150., "Reference shape size")


def setup_cfg(args_cfg, json_input=None):
    """Sets up configuration given parsed arguments and optional json input
    """    
    for key, val in args_cfg.__dict__.items():
        cfg.__dict__[key] = val

    cfg.items = []
    cfg.chars = ''
    cfg.char_map = {}

    if json_input is None and '.json' in cfg.input:
        json_input = cfg.input
    
    if json_input is not None: #'.json' in cfg.input:
        print('Loading json input')
        jdata = utils.load_json(json_input )
        # Read font list if present
        if 'input_path' in jdata:
            cfg.input_path = jdata['input_path']
            
        if 'params' in jdata:
            for key, val in jdata['params'].items():
                if key in cfg.__dict__:
                    cfg.__dict__[key] = val

        if 'chars' in jdata:
            cfg.chars = jdata['chars']

        if 'items' in jdata: 
            cfg.items = jdata['items']

        if 'char_map' in jdata:
            cfg.char_map = jdata['char_map']
    else:
        cfg.input_path = cfg.input
        
    if 'only' in cfg.__dict__ and cfg.only:
        cfg.items = [cfg.only]
    if 'with_chars' in cfg.__dict__ and cfg.with_chars:
        cfg.chars = cfg.with_chars
#endf



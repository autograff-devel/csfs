'''Public parameter config'''
import autograff.utils as utils

cfg = lambda : None


cfg.tangent_sleeve_tolerance = 0.5  # <- tangent approximation tolerance, multiple of ds
cfg.max_radius_height_ratio = 2.    # <- maximum feature radius for abs maxima (relative to bbox)
cfg.vma_thresh = 1                  # <- Voronoi skeleton reg. threshold
cfg.straightness_tolerance = 1      # <- Straight segment tolerance (multiple of ds)
cfg.merge_thresh = 0.98             # <- disk overlap threshold
cfg.minima_smooth_k = 100           # <- Smoothing (for spline) when computing minima
cfg.feature_saliency_thresh = 5e-3  # <- Minimum CSF saliency
cfg.minima_saliency_thresh = 1e-6  # <- Minimum CSF saliency

cfg.corner_radius_thresh = 1        # <- corner radius threshold
cfg.compute_CSF_axes = True         # <- If True, compute local axes
cfg.num_sym_passes = 3              # <- Max passes for csf 
cfg.debug_draw = False              

cfg.smoothing = 0.5                 # <- input smoothing window
cfg.anchor_expansion_tol = 0.0      # <- annulus tolerance for terminal disks
cfg.minima_expansion_tol = 0.05     # <- annulus for minima

cfg.refine_clothoid_fit = False

cfg.draw_steps = False
cfg.debug_feature_support = True

cfg.shape_size = 150



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



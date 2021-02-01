'''
Curvilinear Shape Features
Computes curvature extrema and reconstruct a contour using local symmetry axes
Currently in preparation:
Berio, Leymarie, Plamondon "Kinematics Reconstruction of Static Calligraphic Traces from Curvilinear Shape Features"
And
Berio, Asente, Echevarria and Leymarie "Stroke Styles:  Stroke-Based Segmentation and Stylization of Fonts"
'''

from __future__ import division
from importlib import reload

import time
import math
import numpy as np
from numpy.linalg import det, inv, norm
import os, sys
import matplotlib.pyplot as plt
import autograff.plut as plut
import autograff.geom as geom
import autograff.utils as utils
import autograff.geom.euler_spiral as es
from autograff.algorithms import UnionFind
from autograff.utils import perf_timer
from functools import cmp_to_key
import pdb
from autograff.graph import traverse_directed_nodes, traverse_directed_edges
from autograff.geom.tangent_cover import tangent_cover

import networkx as nx

from scipy.interpolate import interp1d
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.qhull import QhullError
from collections import namedtuple, defaultdict

from . import voronoi_skeleton as vma
from . import config as config
reload(config)

brk = pdb.set_trace

# Configuration (shared)
cfg = config.cfg 

# cfg.vma_thresh = 0.5 # VMA pruning threshold (chord residual)
# 
# cfg.merge_thresh = 0.98 # disk overlap threshold
# cfg.anchor_expansion_tol = 0.0 # adjustment factor for recomputing anchor points of disks (min distance to contour) 
# cfg.feature_saliency_thresh = 5e-3 # CSF saliency threshold
# cfg.straightness_tolerance = 1 # Multiple of ds
# cfg.minima_smooth_k = 100
# cfg.cusp_tolerance = 1. # Radius threshold for cusp detection

cfg.r_thresh = 1200. # maximum accepted disk radius  (gets overridden in CASA for relative size)
cfg.minima_r_thresh = 2000

cfg.discard_nested = True
cfg.discard_nested_postprocess = False

cfg.cw_winding = True

# Support types
SUPPORT_ALTERNATE = 0
SUPPORT_EXTREMA = 1
SUPPORT_INTERPOLATED = 2
SUPPORT_CONTACT = 3
SUPPORT_ALL = 4

# Support.
# Saliency support type is used for saliency computations
# Generator support type is used to generate local axes and for recursive computation
# The values set here are finalized. Do not change
cfg.saliency_support_type = SUPPORT_ALL #SUPPORT_ALL #SUPPORT_EXTREMA #SUPPORT_ALL # #ALTERNATE #EXTREMA
cfg.generator_support_type = SUPPORT_CONTACT
cfg.shortest_support_only = False # <- if True max saliency computation stops when it reaches the shortest segment
cfg.cut_axis_segments = False # <- cuts axis segments at first support end
cfg.clamp_support_to_shortest = False # <- deprecated
cfg.interpolation_exp_rise = 5 # Exp-rise paramter for SUPPORT_INTERPOLATED (deprecated)
cfg.support_uses_distance = False#True # <- If True, interpolate support segments relative to adjacent CSF radius

cfg.debug_draw = False #True
cfg.draw_steps = False # draws recursive local MA computations
cfg.save_steps = False # save steps to pdf
cfg.step_count = 0 # Internal for steps

cfg.verbose = False
cfg.draw_anchors = False
cfg.debug_draw_markersize = 4.
cfg.inf_radius = np.inf
cfg.clothoid_opt_xtol = 1e-1 #1e-2 # Clothoid fit termination thresh (for inflections only)
cfg.clothoid_debug_steps = False          
cfg.refine_spiral_curv = False 
cfg.clothoid_opt_method = 'lm' # Unused
cfg.clothoid_nofit_thresh = 1e-2 # Unused

cfg.vma_thresh_farthest = 0.002 # Pruning threshold for Farthest VMA computation (using object angle)
                                # UNUSED

cfg.only_simple_areas = True # Addition not present in thesis. CSF area must be simple

def debug_print(s):
    if cfg.verbose:
        print(s)

''' CSF types '''

Feature = namedtuple('Feature', 'i center extrema_pos, r anchors sign type data') 
Feature.__doc__='''\
    Curvilinear shape feature
    i (int): index of feature along contour
    center (ndarray): center of curvature
    extrema_pos (ndarray): position of extremum (since index may be shifted due to discretization)
    r (float): radius of curvature
    anchors (tuple of 2 ints): indices of anchor points (defines CSF contact region), given by Voronoi generating points
    sign (int): -1, 0, 1 sign of curvature, 0 is an inflection
    type (string): extremum type
    data (dict): additional data (writable)
    '''
FEATURE_POS_EXTREMUM = 'M+'         # Positive maximum (extremum)
FEATURE_NEG_EXTREMUM = 'm-'         # Negative minimum (extremum)
FEATURE_POS_MINIMUM = 'M-'          # Positive minimum (absolute minimum)
FEATURE_NEG_MINIMUM = 'm+'          # negative maximum (absolute minimum)
FEATURE_INFLECTION = 'inflection'   # Inflection
FEATURE_ENDPOINT = 'endpoint'       # Terminal point of an open contour
FEATURE_TRANSITION = 'transition'   # Clothoid segment connecting CSFs

# TODO rename extremaum to abs_maximum and minimum to abs_minimum
def is_extremum(f):
    ''' Returns true if feature is an absolute maximum of curvature'''
    return f.type == FEATURE_POS_EXTREMUM or f.type == FEATURE_NEG_EXTREMUM

def is_minimum(f):
    ''' Returns true if feature is an absolute minimum of curvature'''
    return f.type == FEATURE_POS_MINIMUM or f.type == FEATURE_NEG_MINIMUM

def is_endpoint(f):
 ''' Returns true if feature is an endpoint'''
 return f.type == FEATURE_ENDPOINT

def is_transition(f):
 ''' Returns true if feature is an endpoint'''
 return f.type == FEATURE_TRANSITION

def get_extrema(features):
    return [f.i for f in features if is_extremum(f) or f.type == FEATURE_ENDPOINT]
   
def get_simplified_shape(features, P):
    ''' Returns a simplified shape given by features points'''
    I = get_extrema(features)
    return P[:,I]

######################################################################
## MEAT 1: Compute features 
COMPUTE_MINIMA = 1
COMPUTE_INFLECTIONS = 2
COMPUTE_ALL = COMPUTE_MINIMA | COMPUTE_INFLECTIONS
CUMPUTE_MAXIMA = 0

def compute_features(P, closed, n_steps=2, draw_steps=0, force_static=False, flags=CUMPUTE_MAXIMA, full_output=False, vma_thresh=None, full_reconstruction=False, stats={}):
    """Compute CSFs or optionally fully reconstruct one or more input contours
       Two variants:
       - for closed shapes or with force_static=True, recursively computes CSFs starting from interior/exterior MA
       - for open contours, iteratively compute CSFs by travelling along contours. Handles self loops but slow.

        NOTE: Assumes shape is uniformely sampled, and assumes shape winding is correct, eventually call geom.fix_shape_winding before

    Args:
        P (ndaray or list): one or more contours
        closed (bool): true if contour is closed
        n_steps (int, optional): number of CSF refinement steps. Defaults to 2.
        draw_steps (int, optional): flags indicating wether to draw steps. Defaults to 0.
                                    if non zero:
                                     1 plots steps for abs maxima, 
                                     2 plots steps for abs minima
                                     3 plots steps for inflections
        force_static (bool, optional): forces recursive refinement also if contour is closed. Defaults to False.
        flags (int, optional):  bitwise combination of flags:
                                COMPUTE_MINIMA: activates minima computation step
                                COMPUTE_INFLECTIONS: activates inflection computation step
                                when 0, only absolute maxima are computed. Defaults to COMPUTE_ALL.
        full_output (bool, optional): If true also return MA, Voronoi and Delaunay. Defaults to False.
        vma_thresh (float, optional): Force Voronoi skeleton regularization threshold. Defaults to None (default in voronoi_skeleton.py).
        full_reconstruction (bool, optional): If true, attempt to reconstruct the whole trace also with Euler Spiral segments
                                               Results in a linear approximation of curvature. Defaults to False.
    
    Returns:
        list: list of Feature namedtuples or list of lists (for compound shapes)
    """    
    if type(P) == list:
        return compute_shape_features(P, closed, n_steps, draw_steps, force_static, flags, full_output, full_reconstruction, stats)

    # force vma thresh if requested
    vma_thresh_old = cfg.vma_thresh
    if vma_thresh is not None:
        cfg.vma_thresh = vma_thresh

    # print("Vma thresh is: %g"%cfg.vma_thresh)
    def test_features(features):
        n = P.shape[1]
        for f in features:
            a, b = f.anchors
            l1 = (b-a)%n
            l2 = (a-b)%n
            if l1 > l2:
                print('test_feature: corrupted feature')
                #raise ValueError

    ds = np.mean(geom.chord_lengths(P))

    terminals = []
    if closed or force_static:
        # Assumes no self intersections, faster
        features, MA, vor, delu = sym_extrema(P, ds, closed=closed, full_output=True, is_sat=True) #dynamic_symmetry_extrema
        self_intersections = []
        if MA is None:
            return []
        disks = MA.graph['disks']
    else:
        # Detect self intersections
        features, self_intersections = open_sym_extrema(P, ds, get_intersections=True)
        #features = dynamic_symmetry_extrema(P, closed=closed) 

    if len(features):
        sign0 = features[0].sign
        sign1 = features[-1].sign
    else:
        sign0 = 1
        sign1 = 1
    
    #pdb.set_trace()
    # Add entpoints if open
    if closed==False:
        # Add endpoints
        f0 = Feature(i=0,
                   center=P[:,0],
                   extrema_pos=P[:,0],
                   r=cfg.inf_radius, #1, #1000.,
                   anchors=(0,0),
                   type=FEATURE_ENDPOINT,
                   sign=sign0,
                   data={'branch_len':1000})
        f1 = Feature(i=P.shape[1]-1,
                   center=P[:,-1],
                   extrema_pos=P[:,-1],
                   r=cfg.inf_radius, #1, #1000.,
                   anchors=(P.shape[1]-1, P.shape[1]-1),
                   type=FEATURE_ENDPOINT,
                   sign=sign1,
                   data={'branch_len':1000})
        features = [f0] + features + [f1]

    test_features(features)
    features = merge_features(features, P, closed)
    features = discard_unsalient(features, P, closed)
    test_features(features)

    # Refinement steps    
    cfg.step_count = 0

    initial_count = len(features)

    # all maxima
    for i in range(n_steps):
        features = sort_features(P, features + compute_local_maxima(P, ds, features, closed=closed, draw_steps=draw_steps==1, self_intersections=self_intersections), closed)
        #pdb.set_trace()
        features = merge_features(features, P, closed)
        test_features(features)
        if len(features) != initial_count:
            stats['nested_CSFs'] = True
            initial_count = len(features)
        else:
            #print('Bailing')
            break
        
    features = discard_unsalient(features, P, closed)
    
    # minima 
    if flags&COMPUTE_MINIMA:
        minima = compute_local_minima(P, ds, features, closed=closed, draw_steps=draw_steps==2)    
        features = sort_features(P, features + minima, closed)
    
    test_features(features)
    features = discard_unsalient(features, P, closed, only_minima=True)

    # inflections
    if flags&COMPUTE_INFLECTIONS:
        inflections = compute_local_inflections(P, ds, features, closed=closed, draw_steps=draw_steps==3)    
        # # merge and sort
        features = sort_features(P, features + inflections, closed) #minima + inflections + features)
    
    test_features(features)
    
    # reset original thresh
    cfg.vma_thresh = vma_thresh_old
    
    if full_reconstruction: # approximate whole trace with euler spirals
        features = compute_internal_angles(features, P)
        features = compute_transitions(features, P, closed)

    if full_output:
        if not closed and not force_static:
            print('full_output flag only valid for static sym calculation')
            raise ValueError
        return features, MA, vor, delu 
    return features
    
def compute_shape_features(S, closed, n_steps=1, draw_steps=False, force_static=False, flags=COMPUTE_ALL, full_output=False, full_reconstruction=False, stats={}):
    feat_list = []
    for P in S:
        features = compute_features(P, closed, n_steps, draw_steps, force_static, flags, full_reconstruction=full_reconstruction, stats=stats)
        feat_list.append(features)
    return feat_list

def unscale_features(features, ratio):
    return [f._replace(center=f.center/ratio, r=f.r/ratio) for f in features]

def transform_features(mat, features):
    if type(features[0]) == list:
        feature_list = features
        feature_list_tsm = []
        for features in feature_list:
            feature_list_tsm += [transform_features(mat, features)]
        return feature_list_tsm

    scale = np.sqrt(np.abs(np.linalg.det(mat[:2,:2])))
    return [f._replace(center=geom.affine_mul(mat, f.center), r=f.r*scale) for f in features]


###############################################################
### Extension to open/self intersecting contours
def segment_path_intersection(pa, pb, segs):
    for i, (qa, qb) in enumerate(segs):
        res, ins = geom.segment_intersection(pa, pb, qa, qb)
        if res:
            #plut.fill_circle(ins, 7, 'r')
            return i
    return None

def segment_path_intersection_sweep(pa, pb, ctr):
    ins, inds = geom.segment_shape_intersections(pa, pb, [np.array(ctr).T], 0.01, get_indices=True)
    return ins, inds
        
def split_at_self_intersections(P):
    m = P.shape[1]
    #I = [0]
    I = []
    prev = []
    prev_ctr = []
    for i in range(m-1):
        pa, pb = P[:,i], P[:,i+1]
        if len(prev)>2:
            #if segment_path_intersection_sweep(pa, pb, prev_ctr): #segment_path_intersection(pa, pb, prev[:-1]):
            j = segment_path_intersection(pa, pb, prev[:-1])
            if j is not None:

                #pdb.set_trace()
                I.append((i,j)) #(current segment, intersecting segment)
                prev = [(pa, pb)]
                prev_ctr = [pa]
                continue
        prev.append((pa, pb))
        if not prev_ctr:
            prev_ctr.append(pa)
        prev_ctr.append(pb)
        
    #I.append(m)
    return I

def open_sym_extrema(P, ds, self_intersections=None, get_intersections=False):
    if self_intersections is None:
        #pdb.set_trace()
        self_intersections = split_at_self_intersections(P)
    # Get only intersection index
    # discard any self intersection that is not fully contained in segment
    I = [i for i,j in self_intersections if i>0 and i<P.shape[1] and j > 0 and j < P.shape[1]]
    # Add extremities
    I = [0] + I + [P.shape[1]]
    #pdb.set_trace()
    features = []
    for a, b in zip(I, I[1:]):
        #pdb.set_trace()
        local_features = sym_extrema(P[:,a:b], ds, closed=False)
        local_features = [shift_feature(a, f, P, closed=False) for f in local_features]
        features += local_features
    #features =  merge_features(features, P, False)
    if get_intersections:
        return features, self_intersections
    
    return features

def symmetry_axis(P, closed, farthest, full_output=False, terminals=None):
    if farthest:
        thresh = cfg.vma_thresh_farthest
        residual = vma.lambda_residual  
    else:
        thresh = cfg.vma_thresh
        residual = vma.chord_residual
    res = vma.voronoi_skeleton([P], thresh, 
                            residual=residual, 
                            closed=closed, 
                            farthest=farthest, 
                            get_voronoi=full_output,
                            terminal_branches=terminals)
    return res

###############################################################
### MEAT 2, features from symmetry axes 
def sym_extrema(P, ds, closed=True, farthest=False, full_output=False, vma_thresh=None, is_sat=False, draw_steps=False):
    ''' Find extrema/features from direct computation of medial axis
        Input:
            P: contour
            closed: open/closed shape
            farthest: if True the farthest Voronoi diagram is used for skeleton computation (will correspond to Leyton's ESAT)
            full_output: function will also return the nearest/farthest MA graph, Voronoi diagram and Delaunay triangulation 
            vma_thresh: residual threshold for MA computation, for nearest the chord residual is used, for farthest, an object angle based measure
        Output:
            features: list of Feature namedtuples (see above)
            optionally:
                MA, vor, delu: MA graph, Voronoi, Delaunay 
    '''
    E = []
    MA = None
    extrema = []
    extrema_pos = []
    anchors = []
    extremities = []
    vor = None
    
    thresh = cfg.vma_thresh
    residual = vma.chord_residual

    # Farthest voronoi settings
    if farthest:
        thresh = cfg.vma_thresh_farthest
        residual = vma.lambda_residual  
        #print thresh

    if vma_thresh is not None:
        thresh = vma_thresh

    try:    
        res = symmetry_axis(P, closed, farthest, full_output) #vma.voronoi_skeleton([P], thresh, residual=residual, closed=closed, farthest=farthest, get_voronoi=full_output)
        if full_output:
            E, MA, vor, delu = res
        else:
            E, MA = res
        #vma.draw_skeleton(MA)
        #ds = np.linalg.norm(P[:,1] - P[:,0])
        disks = MA.graph['disks']

        if draw_steps:
            vma.draw_skeleton(MA)
            
        for e in E:
            if not farthest and disks[e].r > cfg.r_thresh:
                continue
            p = sort_anchors(disks[e].anchors, P, closed)
            #p = expand_voronoi_anchors(disks[e].anchors, disks[e], P, closed)
            mid = get_anchor_midpoint_index(p, P, closed) 
            pm = get_anchor_midpoint_pos(p, P, closed)

            extrema.append(mid)
            anchors.append(p)
            extremities.append(e) 
            extrema_pos.append(pm)

        # endfor 

        # Sort along contour    
        I = np.argsort(extrema)
        extrema = [extrema[j] for j in I]
        anchors = [anchors[j] for j in I]
        E = [extremities[j] for j in I]
        extrema_pos = [extrema_pos[j] for j in I]

    except QhullError:
        print("ERROR: No MA")
        if full_output:
            return [], None, None, None
        else:
            return []
    
    def curv_sign(P, p, i, e):
        a, b = p
        refnode = e
        # Get neighboring vertex if present, which will give us a better distance to compute sign
        bors = list(MA.neighbors(refnode))
        if bors:
            refnode = bors[0]
        v = get_vertex_pos(refnode, MA)
        d1 = P[:,a] - v
        d2 = P[:,b] - v
        #pdb.set_trace()
        if abs(geom.angle_between(d1, d2)) > 3.13: # Hack, computing curvature sign for quite 'flat' features
            return curvature_sign(P[:,a], P[:,i], P[:,b])
        else:
            return curvature_sign(P[:,b], v, P[:,a])
        
    signs = [curv_sign(P, p, m, e) for p, m, e in zip(anchors, extrema, E)]
    radii = [disks[e].r for e in E]
    vertices = [disks[e].center for e in E]

    features = []
    for i,_ in enumerate(vertices):
        if signs[i] > 0:
            ftype=FEATURE_POS_EXTREMUM
        else:
            ftype=FEATURE_NEG_EXTREMUM

        features.append(Feature(i=extrema[i],
                                center=vertices[i],
                                extrema_pos=extrema_pos[i],
                                r=radii[i],
                                anchors=anchors[i],
                                sign = signs[i],
                                type=ftype,
                                data={'is_sat':is_sat, 'branch_len': MA.graph['data']['branch_lengths'][E[i]]}))

    features = sort_features(P, features, closed)
    # VMA sometimes creates overlapping contact regions, make sure this does not happen
    features = force_monotonic(P, features, closed)

    # MINIMA CASE
    if farthest and features:
        # Hack, we keep only one feature for the FVD with maximal radius
        # assuming analysis of a single codon
        imax = np.argmax([f.r for f in features])
        features = [features[imax]]
    # endif

    if full_output:
        return features, MA, vor, delu
    return features
    

##########################################
## Compute/store additional CSF info (axes, support-segments)

def compute_CSF_axis(Pv, center, farthest=False):
    """Compute the local symmetry axis of a CSF given (given its supporting contour)

    Args:
        Pv (contour): Union of support segments and contact region
        center (point): CSF center

    Returns:
        nx.Graph: Local symmetry axis
    """    
    terminals = []
    terminals = []
    if farthest:
        return None #linear_esat(Pv)

    npts = Pv.shape[1]
    E, MA = symmetry_axis(Pv, False, farthest, terminals=terminals)
    #E, MA = vma.voronoi_skeleton([Pv], cfg.vma_thresh, terminal_branches=terminals, closed=False) 
    disks = MA.graph['disks']
    if terminals:
        disks = MA.graph['disks']
        terminals = sorted(terminals, key=lambda branch:geom.distance(disks[branch[0]].center, center)) #P[:,f2.i]))
        #MA.clear()
        if len(terminals) > 1:
            for branch in terminals[1:]:
                MA.remove_nodes_from(branch[:-1])
        terminal = terminals[0]
        if cfg.cut_axis_segments: #len(terminal) > 1:
            edges = list(traverse_directed_edges(MA, terminal[0], terminal[1]))
            MA.remove_edges_from(list(MA.edges()))
            #pdb.set_trace()
            for a, b in edges: 
                MA.add_edge(a, b)
                anchors = disks[a].anchors
                if 0 in anchors or npts-1 in anchors:
                    break

    return MA

def compute_CSFs(features, P, closed, compute_saliency=True, compute_axis=True):
    """ Computes additional CSF info: contact region, support segments, (optiona) local axis, (optional) saliency
    """    

    if type(P)==list:
        feature_list = []
        for Pp, feats in zip(P, features):
            feature_list.append(compute_CSFs(feats, Pp, closed, compute_saliency, compute_axis))
        return feature_list

    # Avoid transitions
    features = [f for f in features if f.type != FEATURE_TRANSITION]

    n = len(features)  
    count = n
    start = 0
    if not closed:
        start = 1
        count = count-1
    CSFs = []
    
    def safe_clip(X,n):
        if len(X.shape)<2:
            return np.zeros((2,0))
        if X.shape[1] < 1:
            return X
        if n < 0:
            return X[:,:n]
        else:
            return X[:,n:]


    for i in range(start, count):
        f1 = features[(i-1)%n]
        f2 = features[i]
        f3 = features[(i+1)%n]
        # if i==3:
        #     pdb.set_trace()
        l_anchor, r_anchor = left_right_support_anchors(P, f1, f2, f3, closed, support_type=cfg.generator_support_type)

        # Left support
        a, b =  l_anchor, f2.anchors[0]+1
        left = get_contour_segment(P, a, b, closed=closed)
        left_inds = get_contour_segment_indices(P, a, b, closed=closed)

        # Contact
        contact = get_contour_segment(P, f2.anchors[0], f2.anchors[1]+1, closed=closed)
        contact_inds = get_contour_segment_indices(P, f2.anchors[0], f2.anchors[1]+1, closed=closed)

        # Right support
        a, b =  f2.anchors[1], r_anchor+1
        right = get_contour_segment(P, a, b, closed=closed)
        right_inds = get_contour_segment_indices(P, a, b, closed=closed)
        
        cenp = f2.center
        r = f2.r
        d1 = contact[:,0]-cenp
        d2 = contact[:,-1]-cenp
        extremum = P[:,f2.i] #cenp + geom.normalize(d1 + d2)*r
        
        Pv = np.hstack([safe_clip(left, -1), contact[:,:-1], safe_clip(right,1)])

        if compute_axis and Pv.shape[1] > 4:
            MA = compute_CSF_axis(Pv, f2.center, farthest=is_minimum(f2))
        else:
            MA = None
        
        if compute_saliency: # and not is_minimum(f2):
            if is_minimum(f2):
                print('minimum')
            saliency, area = compute_depth_saliency(P, f1, f2, f3, closed, get_area=True)
        else:
            saliency, area = 0, None

        # If left or right support is short, extend by one sample into adjacent contact region
        if left.shape[1] < 2:
            left = get_contour_segment(P, l_anchor, f2.anchors[0]+2, closed=closed)
            left_inds = get_contour_segment_indices(P, l_anchor, f2.anchors[0]+2, closed=closed)

        if right.shape[1] < 2:
            right = get_contour_segment(P, f2.anchors[1], r_anchor+2, closed=closed)
            right_inds = get_contour_segment_indices(P, f2.anchors[1], r_anchor+2, closed=closed)

        # flip left segment
        left = left[:,::-1]
        left_inds = left_inds[::-1]

        f2.data.update(dict(extremum=extremum,
                            support=[left, right],
                            support_inds=[left_inds, right_inds],
                            contact=contact,
                            contact_inds=contact_inds,
                            center=cenp,
                            saliency=saliency,
                            contour=Pv,
                            MA=MA,
                            area=area))
    return CSFs

#############################################
# Shape reconstruction from features
def compute_angles_and_transitions(features, S, closed=False):
    if type(features[0]) == list:
        feature_list = features
        feature_list_extended = []
        for features, P in zip(feature_list, S):
            feature_list_extended += [compute_angles_and_transitions(features, P)]
        return feature_list_extended
    else:
        features = compute_internal_angles(features, S)
        features = compute_transitions(features, S, closed)
        return features

def compute_internal_angles(features, P):
    ''' Computes contact region (circular arc) internal angles for all maxima and minima'''
    if type(P) == list:
        return [compute_internal_angles(F, Pp) for F, Pp in zip(features, P)]
    
    features_new = []
    for f in features:
        if is_extremum(f) or is_minimum(f):
            p1 = P[:,f.anchors[0]]
            p2 = P[:,f.anchors[1]]
            d1 = p1 - f.center
            d2 = p2 - f.center
        
            theta1 = np.arctan2(d1[1], d1[0])
            theta2 = np.arctan2(d2[1], d2[0])
            # This assures we don't get errors in arcsin
            # which can happen with noise in the contour
            pp1 = f.center + [np.cos(theta1)*f.r, np.sin(theta1)*f.r]
            pp2 = f.center + [np.cos(theta2)*f.r, np.sin(theta2)*f.r]
            c = np.linalg.norm(pp2 - pp1)
            s = np.sign( geom.triangle_area(p1, f.center, p2))

            #theta = np.arcsin(c / f.r)*s
            theta = np.arcsin(c / (f.r*2))*s*2

            f.data['theta'] = theta #= f._replace(data={'theta':theta}) # FIXME
        features_new.append(f)
    return features_new

def approximate_params_tangent(Pv, tol=2.):
    T = tangent_cover(Pv, tol)
    t1 = T[0]
    t2 = T[-1]
    th1 = np.arctan2(t1[1], t1[0])
    th2 = np.arctan2(t2[1], t2[0])

    p1 = Pv[:,0]
    p2 = Pv[:,-1]
    ts = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
    s0, s1 = es.fit_euler_spiral(th1-ts, th2-ts)
    return s0, s1

def compute_transition(P, Pv, a, b, sign, adjacent_info, straight=False):
    if cfg.refine_clothoid_fit:
        #print "FITTING TRANSITION %d, start with: %.3f %.3f"%(j, s0,s1)
        
        #pdb.set_trace()
        Pv_samp = geom.uniform_sample_n(Pv, cfg.spiral_subdivision, closed=False)
        with perf_timer('Tangent approx'):
            s0, s1 = approximate_params_tangent(Pv_samp)
        
        with perf_timer('Clothoid fit main'):
            #pdb.set_trace()
            (s0, s1), dx = es.fit_clothoid(Pv_samp, (s0, s1), tol=cfg.clothoid_opt_xtol, debug_draw=cfg.clothoid_debug_steps, get_err=True)
        # if f1.type == FEATURE_ENDPOINT:
        #     with perf_timer('Clothoid fit first endpoint'):
        #         ss, dx2 = es.fit_clothoid(Pv_samp, (-s0, s1), tol=cfg.clothoid_opt_xtol, debug_draw=cfg.clothoid_debug_steps, get_err=True)
        #         if np.mean(dx2) < np.mean(dx):
        #             s0, s1 = ss
        # if f2.type == FEATURE_ENDPOINT:
        #     with perf_timer('Clothoid fit second endpoint'):
        #         ss, dx2 = es.fit_clothoid(Pv_samp, (s0, -s1), tol=cfg.clothoid_opt_xtol, debug_draw=cfg.clothoid_debug_steps, get_err=True)
        #         if np.mean(dx2) < np.mean(dx):
        #             s0, s1 = ss



        # Adjust curvature if estimate is near to bounding features
        # if cfg.refine_spiral_curv:
        #     k0, k1, ld = to_kappa(P, a, b, s0, s1)
        #     khat_0 = 1. / f1.r * -f1.sign
        #     khat_1 = 1. / f2.r * -f2.sign
        #     shat_0, shat_1 = to_s(P, a, b, khat_0, khat_1, ld)

        #     if abs(k0 - khat_0) < 0.05:
        #         s0 = shat_0
        #     if abs(k1 - khat_1) < 0.05:
        #         s1 = shat_1
    else:
        T = tangent_cover(Pv, 1)
        t1 = T[0]
        t2 = T[-1]
        th1 = np.arctan2(t1[1], t1[0])
        th2 = np.arctan2(t2[1], t2[0])

        p1 = Pv[:,0]
        p2 = Pv[:,-1]
        ts = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        s0, s1 = es.fit_euler_spiral(th1-ts, th2-ts)

        if np.sign(s1) != np.sign(s0):
            infl_t = abs(s0) / (abs(s1) + abs(s0))
            #infl_t = abs(s0) / abs(s1 - s0)
            infl = a + int(infl_t * Pv.shape[1])
        else:
            # with same sign get mid point (hack?)
            infl = a + Pv.shape[1]//2

    if np.sign(s1) != np.sign(s0):
        infl_t = abs(s0) / (abs(s1) + abs(s0))
        #infl_t = abs(s0) / abs(s1 - s0)
        infl = a + int(infl_t * Pv.shape[1])
    else:
        # with same sign get mid point (hack?)
        infl = a + Pv.shape[1]//2
            
    f = Feature(i=infl%P.shape[1],
                center=P[:,infl%P.shape[1]],
                extrema_pos=P[:,infl%P.shape[1]],
                r=cfg.inf_radius,
                anchors=(a,b),
                type=FEATURE_TRANSITION,
                sign=sign,
                data={'branch_len':1e-5, 's':(s0, s1), 'is_straight':straight, 'adjacent_info':adjacent_info})
    return f


# def compute_transitions(features, P, closed, only_maxima=False):
#     ''' Compute remaining transition regions between CSFS
#         Optionally only between absolute maxima and endpoints
#     '''
#     if type(P) == list:
#         return [compute_transitions(F, Pp, closed) for F, Pp in zip(features, P)]
    
#     m = len(features)
    
#     transitions = []
#     j = 1

#     count = m
#     # if closed
#     closed = True
#     if features[0].type == FEATURE_ENDPOINT:
#         count = count-1
#         closed = False
    
#     features_2 = []

#     debug_print('Computing transitions\n')

#     for i in range(count):
#         f1 = features[i]
#         f2 = features[(i+1)%m]

#         if True: #cfg.verbose:
#             utils.progress_bar(float(i)/(count))
        
#         # We may want to compute transitions only between opposite sign maxima and endpoints
#         if only_maxima: 
#             if is_minimum(f1) or is_minimum(f2):
#                 features_2.append(f1)
#                 continue
            
#         if f1.type == FEATURE_INFLECTION or f2.type == FEATURE_INFLECTION:
#             features_2.append(f1)
#             continue
        
#         a, b = f1.anchors[1], f2.anchors[0]
#         #pdb.set_trace()
#         if not closed and b < a: # HACK, TODO need to sort this eventually (after lognormal)
#             #pdb.set_trace()
#             print('Flipped ordering')
#             a, b = b, a

#         Pv = get_contour_segment(P, a, b, closed)
#         # Skip very short segments (could use straight lines here)
#         if len(Pv.shape) < 2 or Pv.shape[1] < 5:
#             print("Short one: %s"%i)
#             infl = (a+b)//2
#             s0 = 0.01
#             s1 = s0 + 1e-8

#             features_2.append(f1)
#             features_2.append(Feature(i=infl,
#                                    center=P[:,infl],
#                                    extrema_pos=P[:,infl],
#                                    r=cfg.inf_radius,
#                                    anchors=(a,b),
#                                    type=FEATURE_TRANSITION,
#                                    sign=f1.sign,
#                                    data={'branch_len':1e-5, 's':(s0, s1), 'is_straight':True}))

#             debug_print("SHORT TRANSITION")
#             continue

#         if cfg.refine_clothoid_fit:
#             #print "FITTING TRANSITION %d, start with: %.3f %.3f"%(j, s0,s1)
#             j = j+1
            
#             Pv_samp = geom.uniform_sample_n(Pv, 30, closed=False)
#             with perf_timer('Tangent approx'):
#                 s0, s1 = approximate_params_tangent(Pv_samp)
            
#             with perf_timer('Clothoid fit main'):
#                 (s0, s1), dx = es.fit_clothoid(Pv_samp, (s0, s1), tol=cfg.clothoid_opt_xtol, debug_draw=cfg.clothoid_debug_steps, get_err=True)
#             # if f1.type == FEATURE_ENDPOINT:
#             #     with perf_timer('Clothoid fit first endpoint'):
#             #         ss, dx2 = es.fit_clothoid(Pv_samp, (-s0, s1), tol=cfg.clothoid_opt_xtol, debug_draw=cfg.clothoid_debug_steps, get_err=True)
#             #         if np.mean(dx2) < np.mean(dx):
#             #             s0, s1 = ss
#             # if f2.type == FEATURE_ENDPOINT:
#             #     with perf_timer('Clothoid fit second endpoint'):
#             #         ss, dx2 = es.fit_clothoid(Pv_samp, (s0, -s1), tol=cfg.clothoid_opt_xtol, debug_draw=cfg.clothoid_debug_steps, get_err=True)
#             #         if np.mean(dx2) < np.mean(dx):
#             #             s0, s1 = ss

#             if np.sign(s1) != np.sign(s0):
#                 infl_t = abs(s0) / abs(s1 - s0)
#                 infl = a + int(infl_t * Pv.shape[1])
#             else:
#                 # with same sign get mid point (hack?)
#                 infl = a + Pv.shape[1]//2

#             # Adjust curvature if estimate is near to bounding features
#             if cfg.refine_spiral_curv:
#                 k0, k1, ld = to_kappa(P, a, b, s0, s1)
#                 khat_0 = 1. / f1.r * -f1.sign
#                 khat_1 = 1. / f2.r * -f2.sign
#                 shat_0, shat_1 = to_s(P, a, b, khat_0, khat_1, ld)

#                 if abs(k0 - khat_0) < 0.05:
#                     s0 = shat_0
#                 if abs(k1 - khat_1) < 0.05:
#                     s1 = shat_1
#         else:
#             T = tangent_cover(Pv, 1)
#             t1 = T[0]
#             t2 = T[-1]
#             th1 = np.arctan2(t1[1], t1[0])
#             th2 = np.arctan2(t2[1], t2[0])

#             p1 = Pv[:,0]
#             p2 = Pv[:,-1]
#             ts = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
#             s0, s1 = es.fit_euler_spiral(th1-ts, th2-ts)

#             if np.sign(s1) != np.sign(s0):
#                 infl_t = abs(s0) / abs(s1 - s0)
#                 infl = a + int(infl_t * Pv.shape[1])
#             else:
#                 # with same sign get mid point (hack?)
#                 infl = a + Pv.shape[1]//2

#         #mid = (a + b)//2
#         features_2.append(f1)
#         features_2.append(Feature(i=infl%P.shape[1],
#                                    center=P[:,infl%P.shape[1]],
#                                    extrema_pos=P[:,infl%P.shape[1]],
#                                    r=cfg.inf_radius,
#                                    anchors=(a,b),
#                                    type=FEATURE_TRANSITION,
#                                    sign=f1.sign,
#                                    data={'branch_len':1e-5, 's':(s0, s1), 'is_straight':False}))
                   
#     if not closed:
#         features_2.append(f2)

#     debug_print('Finished transitions')
#     return features_2 #sort_features(transitions + features, remove_duplicates=False)

# def subdivide_curved_segments(Pv, a, b):
#     phi = geom.turning_angles(Pv)
#     k = np.abs(np.sum(phi))
#     maxang = np.pi/2 #np.pi*3/4
#     #pdb.set_trace()
#     if k > maxang:
#         #pdb.set_trace()
#         nsubd =  int(k/maxang)
#         print('Subdividing %d times'%(nsubd))
#         print(geom.degrees(k))

#         subd = np.linspace(a, b, 2 + nsubd).astype(int)
#         I = subd-a
#         m = len(I)
#         spans = [[Pv[:,I[i]:I[i+1]], subd[i], subd[i+1]] for i in range(m-1)]
#         return spans
#     return [[Pv, a, b]]

def subdivide_curved_segments(Pv, a, b):
    phi = geom.turning_angles(Pv)
    k = np.abs(np.sum(np.abs(phi)))
    #k = np.abs(np.sum(phi))
    maxang = cfg.transition_angle_subd #np.pi * 4/5 ##*3/4
    #print("maxang")
    #print(maxang)
    #print(k)
    if k > maxang and Pv.shape[1] > 4:
        #pdb.set_trace()
        m = Pv.shape[1]//2
        print('Subdividing: %d -> [%d, %d], [%d, %d]'%(Pv.shape[1], a, a+m+1, a+m, b))
        
        return (subdivide_curved_segments(Pv[:,:m], a, a+m) +
                subdivide_curved_segments(Pv[:,m:], a+m, b))
    return [[Pv, a, b]]

def compute_transitions(features, P, closed, only_maxima=False):
    ''' Compute remaining transition regions between CSFS
        Optionally only between absolute maxima and endpoints
    '''
    if type(P) == list:
        return [compute_transitions(F, Pp, closed) for F, Pp in zip(features, P)]
    
    m = len(features)
    if not m:
        return features
    
    transitions = []
    j = 1

    count = m
    # if closed
    closed = True
    if features[0].type == FEATURE_ENDPOINT:
        count = count-1
        closed = False
    
    features_2 = []

    debug_print('Computing transitions\n')

    ds = np.mean(geom.chord_lengths(P))

    for i in range(count):
        f1 = features[i]
        f2 = features[(i+1)%m]
        adjacent_info = {'sign':(f1.sign, f2.sign),
                         'type':(f1.type, f2.type)}

        if True: #cfg.verbose:
            utils.progress_bar(float(i)/(count))
        
        # We may want to compute transitions only between opposite sign maxima and endpoints
        if only_maxima: 
            if is_minimum(f1) or is_minimum(f2):
                features_2.append(f1)
                continue
            
        if f1.type == FEATURE_INFLECTION or f2.type == FEATURE_INFLECTION:
            features_2.append(f1)
            continue
        
        a, b = f1.anchors[1], f2.anchors[0]
        #pdb.set_trace()
        if not closed and b < a: # HACK, TODO need to sort this eventually (after lognormal)
            #pdb.set_trace()
            print('Flipped ordering')
            a, b = b, a

        Pv = get_contour_segment(P, a, b, closed)
        # Skip very short segments (could use straight lines here)
        is_straight = is_segment_straight(Pv, ds)
        print('straight: ' + str(is_straight))
        if len(Pv.shape) < 2 or Pv.shape[1] < 5: # or is_segment_straight(Pv, ds):
            print("Short one: %s"%i)
            infl = (a+b)//2
            s0 = 0.01
            s1 = s0 + 1e-8

            features_2.append(f1)
            features_2.append(Feature(i=infl,
                                   center=P[:,infl],
                                   extrema_pos=P[:,infl],
                                   r=cfg.inf_radius,
                                   anchors=(a,b),
                                   type=FEATURE_TRANSITION,
                                   sign=f1.sign,
                                   data={'branch_len':1e-5, 's':(s0, s1),
                                         'is_straight':True,
                                         'adjacent_info':adjacent_info}))

            debug_print("Straight or short transition")
            continue

    
        features_2.append(f1)
        Pv_smooth = geom.gaussian_smooth_contour(Pv, cfg.subdivision_smooth_sigma*ds, False)
        #pdb.set_trace()

        spans = subdivide_curved_segments(Pv_smooth, a, b)
        #pdb.set_trace()
        for i, (Ppv, ia, ib) in enumerate(spans):
            if i > 0:
                ftype = FEATURE_POS_MINIMUM
                
                if f1.sign < 0:
                    ftype = FEATURE_NEG_MINIMUM

                if closed:
                    ia = ia%P.shape[1]
                    ib = ib%P.shape[1]
                features_2.append(Feature(i=ia,
                                   center=P[:,ia],
                                   extrema_pos=P[:,ia],
                                   r=cfg.inf_radius,
                                   anchors=(ia,ia),
                                   type=ftype,
                                   sign=f1.sign,
                                   data={'branch_len':1e-5, 'is_straight':is_straight}))
                
            features_2.append(compute_transition(P, Ppv, ia, ib, f1.sign, adjacent_info, straight=is_straight))

    if not closed:
        features_2.append(f2)
        # make sure that any transition next to end-point is labelled
        for f in [features_2[1], features_2[-1]]:
            if f.type == FEATURE_TRANSITION:
                f.data['end_transition'] = True

    
    debug_print('Finished transitions')
    return features_2 #sort_features(transitions + features, remove_duplicates=False)

###############################################################
### Helpers 
def in_circular_interval(v, interval, n):
    ''' Returns wether v is in an interval (integer)'''
    a, b = interval
    if a <= b:
        return v >= a and v <= b
    
    return v >= a or v <= b
    
def circular_interval_union(s1, s2, n):
    ''' Union of two intervals on a circle'''
    w1 = min((s2[0]-s1[1])%n, (s1[1]-s2[0])%n)
    w2 = min((s1[0]-s2[1])%n, (s2[1]-s1[0])%n)
    if w1 <= w2:
        return (s1[0], s2[1])
    return (s2[0], s1[1])
    
def get_anchor_midpoint_index(p, P, closed):
    if not closed:
        return p[0] + (p[1] - p[0])//2 # (p[0] + p[1])//2
    a, b = p
    n = P.shape[1]
    # wrapped midpoint
    m = (a + (((b-a)%n)//2)) % n
    return m

def get_anchor_midpoint_pos(p, P, closed):
    n = P.shape[1]
    Pv = get_contour_segment(P, p[0], p[1]+1)
    if Pv.shape[1] < 2:
        return Pv[:,0]
    return geom.path_subset_of_length(Pv, geom.chord_length(Pv)/2)[:,-1]

def get_contour_midpoint_index(a, b, P, closed):
    # TODO: Check +1 issues here and elsewhere
    if not closed:
        return a + (b - a)//2 # (p[0] + p[1])//2
    n = P.shape[1]
    # wrapped midpoint
    m = (a + (((b-a)%n)//2)) % n
    return m

# Need to take care of wrapped anchor points
def sort_anchors(p, P, closed):
    if not closed:
        a, b = p
        if b < a:
            a, b = b, a
        return a, b
    # endif
    a, b = p
    n = P.shape[1]
    if (a-b)%n < (b-a)%n:
        a, b = b, a
    return a, b
# end func  

#def force_monotonic_anchors(f1, f2, )
def arg_sort_indices(I, P, closed):
    if not closed:
        return np.argsort(I)
    n = P.shape[1]
    def cmp(a, b):

        if (I[a]-I[b])%n < (I[b]-I[a])%n:
            return 1
        return -1
    return sorted(range(len(I)), key=cmp_to_key(cmp))

# def sort_anchors(p):
#     ''' Sort anchors in increasing order, TODO 'sort' this out (for closed ctrs)!'''
#     if p[0] > p[1]:
#         return p[1], p[0]
#     return p[0], p[1]
    
def get_extremity_features(P, MA, E, closed):
    ''' Get extremity features'''
    mid_pts = []
    proj_pts = []
    exts = []
    extrema_pos = []
    disks = MA.graph['disks']
    
    for e in E:
        p = sort_anchors(disks[e].anchors, P, closed) #expand_anchors(P, disks[e].anchors, disks[e].center, disks[e].r, False)
        if p[1] < p[0]:
            p = (p[1], p[0])
        mid = get_anchor_midpoint_index(p, P, closed) #(p[0] + p[1]) // 2
        mp = get_anchor_midpoint_pos(p, P, closed)
        mid_pts.append(mid)
        proj_pts.append(p)
        exts.append(e) 
        extrema_pos.append(mp)

    # Sort along contour    
    I = np.argsort(mid_pts)
    mid_pts = [mid_pts[j] for j in I]
    proj_pts = [proj_pts[j] for j in I]
    exts = [exts[j] for j in I]
    E = [E[j] for j in I]
    extrema_pos = [extrema_pos[j] for j in I]

    return exts, mid_pts, extrema_pos, proj_pts, E
 
def preprocess_shape(S, closed, size=100, smooth_sigma=0., get_ratio=False, vertical_scale=False, pre_smooth=False, resample=True, draw=False):
    ''' Utility function to rescale and uniformely sample a shape before path-symmetry transform
        Input:
            S: list of matrices, each matrix represents a contour and is 2 X n where n is the number of points
            closed: True if the contour is closed, False otherwise
            size: reference size for shape scaling, the shape is scaled so that\
                  the maximum extent of its oriented bounding box matches this value
            smooth_sigma: standard deviation of the Gaussian used for smoothing the contour, if 0 no smoothing is applied
            get_ratio: if True also return the scaling ratio
            resample: if True contour is uniformely sampled at steps of distance 1
            pre_smooth: if True smoothing is applied before resampling
            draw: debug draw shape with matplotlib 
        Output:
            S: scaled shape
            optionally the scaling ratio
    ''' 
    if type(S) != list:
        is_path = True
        S = [S]
    else:
        is_path = False

    if pre_smooth:
        if smooth_sigma > 0:
            S = [geom.gaussian_smooth_contour(P, smooth_sigma, closed=closed) for P in S]
    if resample:
        S, ratio = geom.rescale_and_sample(S, closed, size, get_ratio=True, vertical_scale=vertical_scale)
    else:
        S, ratio = S, 1.
    if not pre_smooth:
        if smooth_sigma > 0:
            S = [geom.gaussian_smooth_contour(P, smooth_sigma, closed=closed) for P in S]

    if draw:
        plt.figure()
        plut.stroke_shape(S, 'k', closed=closed)
        for P in S:
            plut.draw_marker(P[:,0], 'go')
        plut.plt_setup(axis=True)
        plt.show()

    if is_path:
        S = S[0]

    if get_ratio:
        return S, ratio
    return S

def get_shape_index(i, start_inds, shape_inds):
    si = shape_inds[i]
    return si, i - start_inds[si]

def get_flat_index(si, start_inds):
    return start_inds[si[0]] + si[1]
    
def curvature_sign(a, b, c):
    s = np.sign(geom.triangle_area(a, b, c)) #TODO sort me out, need to define handedness in config
    if not cfg.cw_winding:
        return s
    return -s

def get_vertex_pos(i, MA):
        verts = MA.graph['vma'][0]
        return verts[i]

def sort_features(P, features, closed, remove_duplicates=True):
    ''' Sort features along contour and removes duplicates'''
    features = sorted(features, key=lambda v: v.i)
    n = P.shape[1]
    if len(features) > 1 and closed:
        # Check if wrapping 
        if ((features[-1].i - features[-2].i)%n > 
            (features[0].i - features[-1].i)%n):
            f = features.pop()
            features = [f] + features
    # remove duplicates
    if remove_duplicates:
        I = np.where(np.diff(np.array([f.i for f in features]))==0)[0]
        features = [f for i,f in enumerate(features) if not i in I]
    return features

def contact_regions_overlap(f1, f2, P, closed):
    if not closed:
        a, b = f1.anchors[1], f2.anchors[0]
        if b < a:
            return True
        return False
    # endif
    a, b = f1.anchors[1], f2.anchors[0]
    n = P.shape[1]
    # Test if second anchor of first feature is closer to second extrama
    if (f2.i - a)%n < (f2.i - b)%n:
        return True
    return False
#endf

def contact_region_length(f, P, closed):
    if closed:
        return f.anchors[1] - f.anchors[0]
    n = P.shape[1]
    return (f.anchors[1] - f.anchors[0])%n
#endf

def force_monotonic(P, features, closed):
    ''' Removes potentially overlapping contact regions
    Check consecutive feature pairs, if the first contact region overlaps with the second,
    shrink the longest contact region and recompute midpoint'''
    if not features:
        return features
    I = list(range(len(features)))
    if closed:
        I = I + [0]
    features_replace = {}
    for i, j in zip(I, I[1:]):
        fi = features[i]
        fj = features[j]
        if contact_regions_overlap(fi, fj, P, closed):
            if contact_region_length(fi, P, closed) > contact_region_length(fj, P, closed):
                #pdb.set_trace()
                #contact_regions_overlap(fi, fj, P, closed)
                a, b = fi.anchors[0], fj.anchors[0]
                fi = fi._replace(anchors=(a, b), i=get_anchor_midpoint_index((a,b), P, closed))
                features_replace[i] = fi
            else:
                #pdb.set_trace()
                #contact_regions_overlap(fi, fj, P, closed)
                a, b = fi.anchors[1], fj.anchors[1]
                fj = fj._replace(anchors=(a, b), i=get_anchor_midpoint_index((a,b), P, closed))
                features_replace[j] = fj

    features = [f for f in features]
    for i, f in features_replace.items():
        features[i] = f

    return features

#######################################
# ABSOLUTE MAXIMA

def compute_local_maxima(P, ds, features, closed=True, draw_steps=False, P_whole=None, self_intersections=[]):
    ''' Compute extrema for each part defined between two features'''
    if P_whole is None:
        P_whole = P
        
    if not features:
        return []

    if closed:
        features_loc = features + [features[0]]
    else:
        features_loc = features
        
    res = []
    for f1, f2 in zip(features_loc, features_loc[1:]):
        # pdb.set_trace()
        if draw_steps:
            plt.figure(figsize=(5,5))
            # if (f1.anchors[1]==143 and f2.anchors[0]==257):
            #     pdb.set_trace()
            plt.title(str((f1.anchors[1], f2.anchors[0])))
            plut.stroke_poly(P_whole, 'k', closed=closed)
        local, Pv, zone = compute_segment_maxima(P, ds, f1, f2, closed=closed, farthest=False, draw_steps=draw_steps, self_intersections=self_intersections)
        if draw_steps:
            plut.plt_setup()
            plut.set_axis_limits(P_whole, 7)
            if cfg.save_steps:
                plt.savefig('step_%d.pdf'%(cfg.step_count))
            plt.show()
        if local:
            # Recurse here?
            local = [f1] + [shift_feature(zone[0], f, P, closed) for f in local]
            res += local
    
    return res 

def discard_nested_feature(P, f, f1, f2):
    # pdb.set_trace()
    o1 = geom.circle_overlap(f.center, f.r, f1.center, f1.r)
    o2 = geom.circle_overlap(f.center, f.r, f2.center, f2.r)
    if o1 > cfg.merge_thresh and f.r > f1.r and f1.sign == f.sign:
        return True
    
    if o2 > cfg.merge_thresh and f.r > f2.r and f2.sign == f.sign:
        return True
    
    return False

def select_most_salient_feature(P, start_feature, end_feature, features, closed, start, draw=False):
    if not features:
        return []
    
    if cfg.discard_nested: 
        features = [f for f in features if not discard_nested_feature(P, f, start_feature, end_feature)]
        if not features:
            return []
        
    saliency = [compute_depth_saliency(P, start_feature, shift_feature(start, f, P, closed), end_feature, closed, debug_draw=draw) for f in features]
    i = np.argmax(saliency)
    f = features[i]
    
    # if cfg.discard_nested_postprocess:
    #     if discard_nested_feature(P, f, start_feature, end_feature):
    #         return []        
    if np.max(saliency) > cfg.feature_saliency_thresh: #1e-8: #0.001:
        if draw:
            plut.fill_circle(f.center, f.r, 'r', alpha=0.3)
            plt.text(*f.center, str([ff.i for ff in features]))
        return [f]
    else:
        if draw:
            plut.fill_circle(f.center, f.r, 'g', alpha=0.3)
            plt.text(*f.center, '%s %.4f'%(str([ff.i for ff in features]),saliency[i]))
    return []

def compute_segment_maxima(P, ds, start_feature, end_feature, closed, farthest=False, draw_steps=False, self_intersections=[]):
    ''' Compute absolute maxima for a part of the contour segment defined between two features'''
    if farthest:
        clr = 'b'
        
    if draw_steps:
        #print (start_feature.anchors[1], end_feature.anchors[0])
        plut.stroke_circle(start_feature.center, start_feature.r, 'c', linewidth=0.5)
        plut.stroke_circle(end_feature.center, end_feature.r, 'c', linewidth=0.5)
    
    n = P.shape[1]
    
    a, b = start_feature.anchors[1],  end_feature.anchors[0]
    
    if not closed and b-a < 4:
        return [], P, (0,0)

    if (b-a)%n < 4:
        return [], P, (0,0)
    
    Pv = get_contour_segment(P, a, b, closed=closed) #i, end_feature.i+1) #

    # Shift self intersections to local segment
    self_intersections = [(i-a, j-a) for i,j in self_intersections]

    if len(Pv.shape) < 2:
        print("Short segment!")
        print((a, b))
        print(n)
        
    features = open_sym_extrema(Pv, ds, self_intersections)
    # features = sym_extrema(Pv, ds, closed=False, farthest=farthest, draw_steps=draw_steps) #, full_output=True)
    if draw_steps:
        plut.stroke_poly(Pv, 'b', linewidth=2., closed=False)
        #vma.draw_voronoi(vor, 'k', alpha=0.3)
        
        #if features:
        draw_features(features, Pv, 'r', markersize=cfg.debug_draw_markersize) #draw_disks=True)

    #features = select_most_salient_feature(P, Pv, features, draw=draw_steps)
    #pdb.set_trace()
    #if (a,b)==(143,257):
    #    pdb.set_trace()
    #pdb.set_trace()
    features = select_most_salient_feature(P, start_feature, end_feature, features, closed, start=a, draw=draw_steps)

    if features and draw_steps:
        pass
    return features, Pv, (a,b)
    
#######################################
# HELPERS

def merge_features(features, P, closed, iteration=0): 
    ''' Merge features based on given disk overlap metric'''
    # DEPRECATED
    return features

def roll_list(L, i):
    n = len(L)
    return [L[(i+j)%n] for j in range(n)]

def feature_list_length(P, features, group):
    if len(group)<=1:
        return 0
    n = P.shape[1]
    l = 0
    for a, b in zip(group, group[1:]):
        l += (features[b].i - features[a].i)%n
    return l

def circular_sort_group(P, features, group):
    ''' Sort features along contour, finds shortest sequence'''
    if len(group)<=1:
        return group
    group = sorted(group, key=lambda i: features[i].i)
    m = len(group)
    rolled_groups = []
    lengths = []
    for i in range(m):
        rolled_groups.append(roll_list(group, i))
        lengths.append(feature_list_length(P, features, rolled_groups[-1]))
    return rolled_groups[np.argmin(lengths)]

def get_contour_segment_indices(X, a, b, closed=True):
    n = X.shape[1]

    if not closed:
        return list(range(max(0,a), min(b,n)))

    #if a==b:
    #    m = X.shape[1]
    #else:
    m = (b-a)%n
    if m == 0:
        m = X.shape[1]
    return [(a+i)%n for i in range(m)]

def get_contour_segment(X, a, b, closed=True):
    ''' Gets a (wrapped) contour segment between a and b
    TODO use indices from above
    '''
    n = X.shape[1]

    if not closed:
        return X[:, max(0, a):min(b, n)]

    #if a==b:
    #    m = X.shape[1]
    #else:
    m = (b-a)%n
    if m == 0: # Loop case
        m = X.shape[1]
    Xp = np.array([X[:,(a+i)%n] for i in range(m)]).T
    return Xp

def get_surrounding_contour_segment(X, a, b, c, closed=True):
    ''' Gets a (wrapped) contour segment between a and c
    and containing b.
    This is useful for longer segments that could break when specifying only two points.
    TODO: test if this is useful/necessary in rest of code
    '''
    n = X.shape[1]

    if not closed:
        return X[:, max(0, a):min(c, n)]

    Xl = get_contour_segment(X, a, b)
    Xr = get_contour_segment(X, b, (c+1)%n)
    return np.hstack([Xl, Xr])
    # Xp = []
    # m = (b-a)%n
    # Xp += [X[:,(a+i)%n] for i in range(m)]
    # m = (c-b)%n
    # Xp += [X[:,(b+i)%n] for i in range(m)]

    # return np.array(Xp).T


def shift_feature(offset, f, P, closed):
    ''' Shifts indices of a feature (wrapped)'''
    n = P.shape[1]
    if closed:
        return f._replace(i=(f.i+offset)%n,
                        anchors=((f.anchors[0]+offset)%n, (f.anchors[1]+offset)%n))
    else:
        return f._replace(i=np.clip(f.i+offset, 0, n),
                        anchors=(np.clip(f.anchors[0]+offset, 0, n),
                                 np.clip(f.anchors[1]+offset, 0, n)))

def expand_feature_anchors(P, f, closed, thresh=None):
    ''' Expands anchor points to a given tolerance from contour'''
    if thresh==None:
        thresh = cfg.anchor_expansion_tol
    if thresh <= 0.:
        return f
    n = P.shape[1]
    anchors = f.anchors
    if closed:
        a1 = anchors[0]%n
        while abs(np.linalg.norm(f.center - P[:,a1]) - f.r) < thresh:
            a1 = (a1 - 1)%n
        a2 = anchors[1]%n
        while abs(np.linalg.norm(f.center - P[:,a2]) - f.r) < thresh:
            a2 = (a2 + 1)%n

    else:
        a1 = min(anchors[0], n-1)
        a2 = min(anchors[1], n-1)
        while abs(np.linalg.norm(f.center - P[:,a1]) - f.r) < thresh and a1 > 0:
            a1 = (a1 - 1)
        while abs(np.linalg.norm(f.center - P[:,a2]) - f.r) < thresh and a2 < n-1:
            a2 = (a2 + 1)
            
    return f._replace(i=(a1+a2)//2,
                      anchors=(a1, a2))

def expand_feature_anchor_at(features, P, i, closed, thresh=None):
    ''' Expands anchor points to a given tolerance from contour, 
        indexed version, checks if we hit the contact for adjacent features'''
    m = len(features)
    f = features[i]

    if closed:
        f_prev = features[(i-1)%m]
        f_next = features[(i+1)%m]
    else:
        f_prev = features[i-1] if i > 0 else None
        f_next = features[i+1] if i < m-1 else None
            
    n = P.shape[1]
    anchors = f.anchors
    # if i==3: # or i==4:
    #     pdb.set_trace()
    # hanzi 37276 is a good corner case for anchor expansion
    if closed:
        a1 = anchors[0]%n
        while abs(np.linalg.norm(f.center - P[:,a1]) - f.r) < thresh:
            if a1==f_prev.anchors[1]:
                break
            a1 = (a1 - 1)%n

        a2 = anchors[1]%n
        while abs(np.linalg.norm(f.center - P[:,a2]) - f.r) < thresh:
            if a2==f_next.anchors[0]:
                break
            a2 = (a2 + 1)%n

    else:
        a1 = anchors[0] #, n-1)
        a2 = anchors[1] #, n-1)
        while abs(np.linalg.norm(f.center - P[:,a1]) - f.r) < thresh and a1 > 0:
            if a1 == 0:
                break
            if f_prev is not None and a2==f_prev.anchors[1]:
                break
            a1 = a1 - 1
        while abs(np.linalg.norm(f.center - P[:,a2]) - f.r) < thresh and a2 < n-1:
            if a2 >= n-1:
                break
            if f_next is not None and a2==f_next.anchors[0]:
                break
            a2 = a2 + 1

    d = ((a2 - a1)%n) // 2    
    mid = (a1 + d)%n
    return f._replace(i=mid,
                      anchors=(a1, a2))

def expand_all_anchors(P, features, closed, thresh=None):
    if thresh==None:
        thresh = cfg.anchor_expansion_tol
    if thresh <= 0.:
        features

    R = [f.r for f in features]
    I = np.argsort(R)
    m = len(features)
    for i in I:
        if not closed and i == 0 or i == m-1:
            continue
        features[i] = expand_feature_anchor_at(features, P, i, closed, thresh)
    return features
    
def expand_and_recompute_midpoint(P, f, closed, thresh=None, limits=None):
    ''' Expands anchor points to a given tolerance from contour'''
    if thresh==None:
        thresh = cfg.anchor_expansion_tol
    if thresh <= 0.:
        return f
    n = P.shape[1]
    anchors = f.anchors

    if closed:
        a1 = anchors[0]%n
        while abs(np.linalg.norm(f.center - P[:,a1]) - f.r) < thresh:
            a1 = (a1 - 1)%n
        a2 = anchors[1]%n
        while abs(np.linalg.norm(f.center - P[:,a2]) - f.r) < thresh:
            a2 = (a2 + 1)%n

    else:
        a1 = min(anchors[0], n-1)
        a2 = min(anchors[1], n-1)
        while abs(np.linalg.norm(f.center - P[:,a1]) - f.r) < thresh and a1 > 0:
            a1 = (a1 - 1)
        while abs(np.linalg.norm(f.center - P[:,a2]) - f.r) < thresh and a2 < n-1:
            a2 = (a2 + 1)
            
    if limits is not None:
        a1 = max(a1, limits[0])
        a2 = min(a2, limits[1])

    return f._replace(i=(a1+a2)//2, anchors=(a1, a2))

def plot_curvature_reconstruction(P, features, closed, lw=0.5):
    reconstruct_curvature(P, features, closed, True, lw)

def s0s1_to_kappa(P, a, b, s0, s1):
    d = np.linalg.norm(P[:,a] - P[:,b])
    l = np.sqrt((es.C_(s1) - es.C_(s0))**2 + (es.S_(s1) - es.S_(s0))**2)
    return np.pi * s0 * (l/d),  np.pi * s1 * (l/d),  # abs(sall[1] - sall[0])

def f_to_kappa(P, f):
    return s0s1_to_kappa(P, *f.anchors, *f.data['s'])

    s0,s1 = f.data['s']
    d = np.linalg.norm(P[:,f.anchors[0]] - P[:,f.anchors[1]])
    l = np.sqrt((es.C_(s1) - es.C_(s0))**2 + (es.S_(s1) - es.S_(s0))**2)
    return np.pi * s0 * (l/d),  np.pi * s1 * (l/d),  # abs(sall[1] - sall[0])
    #return es.t_to_kappa(s0) * (l/d),  es.t_to_kappa(s1) * (l/d),  # abs(sall[1] - sall[0])
 #   return s*s*np.sign(s)*np.pi
    spio2 = np.sqrt(np.pi * .5)
    return s / spio2
    #return s

labels = {}

def one_label(txt=''):
    if txt=='':
        labels.clear()
    if txt in labels:
        return ''
    labels[txt] = 1
    return txt

def is_potential_inflection(f):
    sign = f.data['adjacent_info']['sign']
    type = f.data['adjacent_info']['type']
    if sign[0] == sign[1]:
        return False
    if type[0] != FEATURE_POS_EXTREMUM and type[0] != FEATURE_NEG_EXTREMUM:
        return False
    if type[1] != FEATURE_POS_EXTREMUM and type[1] != FEATURE_NEG_EXTREMUM:
        return False
    return True
    
def reconstruct_curvature(P, features, closed, plot=False, lw=0.5, start=0, get_start=False):
    one_label()
    
    if type(P)==list: # compound shape case
        inflections = []
        start = 0
        for Pp, Ff in zip(P, features):
            I, start = reconstruct_curvature(Pp, Ff, closed, plot, lw, start, get_start=True)
            inflections.append(I)
        return inflections

    kappa = geom.curvature(P, closed=closed)
    if cfg.curvature_smoothing > 0:
        kappa = utils.gaussian_filter1d(kappa, cfg.curvature_smoothing)
        
    x = np.linspace(start, start+kappa.size-1, kappa.size)
    if plot:
        plt.plot(x, kappa, 'k', alpha=0.5, label='curvature', linewidth=0.5)
    ds = np.mean(geom.chord_lengths(P))

    is_trans = False
    inflections = []
    transitions = []

    for i, f in enumerate(features):
        clr = feature_colors[f.type]

        if f.type==FEATURE_INFLECTION or f.type==FEATURE_TRANSITION:
            lbl = 'transition spirals'
            if f.type==FEATURE_INFLECTION:
                lbl = 'inflections'
                if plot:
                    plut.draw_marker([x[f.i], 0.], clr, label=one_label(lbl), markersize=4)
                #inflections.append(f.i)
            else:
                #pdb.set_trace()
                transitions.append(f)
                # if is_trans:
                #     s0, s1 = f.data['s']
                #     a, b = f.anchors
                # else:
                #     s1 = f.data['s'][1]
                #     b = f.anchors[1]
                # is_trans = True

            xspan = x[f.anchors[0]:f.anchors[1]]
            if f.anchors[1] < f.anchors[0]:
                xspan = x[f.anchors[0]:]

            print(f.anchors)
            s = f.data['s']

            kk = f_to_kappa(P, f)
            k = np.linspace(kk[0], kk[1], len(xspan))

            if plot:
                if f.type==FEATURE_TRANSITION:
                    clr = plut.default_color(i)
                plt.plot(xspan, k, color=clr, label=one_label(lbl), linewidth=lw) #, label=one_label(label))

        elif is_extremum(f) or is_minimum(f):
            #continue
            k = 1. / f.r * f.sign
            xspan = x[f.anchors[0]:f.anchors[1]+1]

            label = 'abs maxima'
            #clr = 'r'
            if is_minimum(f):
                label = 'abs minima'
                #clr = 'b'

            if plot:
                plt.plot(xspan, np.ones(len(xspan))*k, color=clr, label=one_label(label), linewidth=lw)

        if f.type != FEATURE_TRANSITION and f.r != np.inf:
            if (transitions and is_potential_inflection(transitions[0])):
                infl = None
                for ft in transitions:
                    s0, s1 = ft.data['s']
                    a, b = ft.anchors
                    if np.sign(s1) != np.sign(s0):
                        infl_t = abs(s0) / abs(s1 - s0)
                        infl = a + int(infl_t * ((b-a)%P.shape[1]))
                        #pdb.set_trace()
                        break
                if infl is None and len(transitions) > 1:
                    for ft1, ft2 in zip(transitions, transitions[1:]):
                        s0 = ft1.data['s'][0]
                        s1 = ft2.data['s'][1]
                        a = ft1.anchors[0]
                        b = ft2.anchors[1]

                        if np.sign(s1) != np.sign(s0):
                            infl = ft1.anchors[1]
                            #infl_t = abs(s0) / abs(s1 - s0)
                            #infl = int(infl_t * (b-a))
                            break
                if infl is not None:
                    if closed:
                        infl = infl%P.shape[1]
                    P1 = get_contour_segment(P, transitions[0].anchors[0], infl, closed)
                    P2 = get_contour_segment(P, infl, transitions[-1].anchors[1], closed)
                    Pp = np.hstack([P1, P2])
                    L1 = geom.chord_length(P1)
                    L2 = geom.chord_length(P2)
                    if min(L1, L2)/(L1 + L2) > 0.2 and not  is_segment_straight(Pp, ds, tol=0.1):
                        if plot:
                            plut.draw_marker([x[infl], 0], 'r', markersize=3, marker='x')
                        inflections.append((infl, Pp))
                transitions = []
                # if np.sign(s1) != np.sign(s0):
                #     infl_t = abs(s0) / abs(s1 - s0)
                #     infl = int(infl_t * (b-a))
                #     #k0,k1 = s0s1_to_kappa(P, a, b, s0, s1)

                #     plut.draw_marker([x[infl], 0], 'r')
                #     print('infl')
                #     inflections.append(infl)
            is_trans = False
    if get_start:
        return inflections, kappa.size
    return inflections


def estimate_inflections(P, features, closed, plot=False, lw=0.5):
    kappa = geom.curvature(P, closed=closed)
    x = np.linspace(0, kappa.size-1, kappa.size)
    if plot:
        plt.plot(x, kappa, 'k:', alpha=0.5, label='curvature', linewidth=0.5)
    ds = np.mean(geom.chord_lengths(P))

    is_trans = False
    inflections = []
    transitions = []

    potential_inflections = []
    m = len(features)
    count = m
    if closed:
        count = count+1
    prev_f = None
    transition_span = []
    for i in range(count):
        f = features[i%m]
        if prev_f is None and is_extremum(f):
            prev_f = f
            continue
        if f.type == FEATURE_TRANSITION and prev_f is not None:
            transition_span.append(f)
            continue
        if is_minimum(f):
            transition_span = []
            prev_f = None
            continue
        if (is_extremum(f) and
            prev_f is not None):
            if prev_f.sign != f.sign and transition_span:
                potential_inflections.append((transition_span, prev_f, f))
            transition_span = []
            prev_f = f

    for transitions, f1, f2 in potential_inflections:
        infl = None
                #pdb.set_trace()
        for ft in transitions:
            s0, s1 = ft.data['s']
            a, b = ft.anchors
            if np.sign(s1) != np.sign(s0):
                #pdb.set_trace()
                infl_t = abs(s0) / (abs(s1) + abs(s0))
                infl = a + int(infl_t * (b-a))
                break
        if infl is None and len(transitions) > 1:
            for ft0, ft1 in zip(transitions, transitions[1:]):
                s0 = ft0.data['s'][0]
                s1 = ft1.data['s'][1]

                if np.sign(s0) != np.sign(s10):
                    infl = ft0.anchors[1]
                    #infl_t = abs(s0) / abs(s1 - s0)
                    #infl = int(infl_t * (b-a))
                    break
        if infl is not None:
            P1 = get_contour_segment(P, f1.i, infl, closed)
            P2 = get_contour_segment(P, infl, f2.i, closed)
            Pp = np.hstack([P1, P2])
            L1 = geom.chord_length(P1)
            L2 = geom.chord_length(P2)
            if min(L1, L2)/(L1 + L2) > 0.1 and not is_segment_straight(Pp, ds, tol=0.5, debug_draw=False):
                if plot:
                    plut.draw_marker([x[infl], 0], 'r')
                inflections.append(infl)
    return inflections
##################################################
# SALIENCY

def discard_unsalient(features, P, closed, only_minima=False): 
    ''' Merge features based on given disk overlap metric'''
    n = len(features)  
    start = 0 
    end = n
    if n < 3:
        return features
    
    salient_features = []
    
    if not closed:
        start = 1
        end = n-1
        salient_features.append(features[0])

    for i in range(start, end):
        fprev = features[(i-1)%n]
        fnext = features[(i+1)%n]
        f = features[i]
        # if is_minimum(f):
        #     pdb.set_trace()
        d = compute_depth_saliency(P, fprev, f, fnext, closed)
        if np.isnan(d):
            d = 10
            #continue
        
        thresh = cfg.feature_saliency_thresh
        
        if only_minima and not is_minimum(f):
            salient_features.append(f)
            continue

        r_thresh = cfg.r_thresh

        if is_minimum(f):
            r_thresh = cfg.minima_r_thresh
            thresh = cfg.minima_saliency_thresh
            
            print('Min saliency = ' + str(d))
            
        if d >= thresh and f.r < r_thresh: #cfg.feature_saliency_thresh:
            salient_features.append(f)
    
    if not closed:
        salient_features.append(features[-1])

    return salient_features

def consolidate_support(P, a, b, f0, f1, f2, closed, support_type):
    n = P.shape[1]

    # For minima always use support up to extremum
    if support_type != SUPPORT_ALL:
        if closed:
            b = b%n
        else:
            b = min(b, n-1)

        if is_minimum(f0):
            a = f0.i
        if is_minimum(f2):
            b = f2.i

        # TODO: a bug happens where f1.anchor[0] < f0.anchor[1] second anchor of f0 extends a bit beyond
        # the first anchor of f1 (the extrema for which we compute saliency)
        # This is a ugly work around but this is probably due to a +1 in contact region commputation
        if closed:
            if (f1.i - f0.i)%n < (f1.anchors[0] - a)%n: # (f1.anchors[0] - a)%n > (a - f1.anchors[0])%n:
                #pdb.set_trace()
                a = f1.anchors[0]
        else:
            if (a > f1.anchors[0]):
                #pdb.set_trace()
                a = f1.anchors[0]

        # Support around minimum goes up to extremum
        if is_minimum(f1):
            return f0.i, f2.i

    return a, b

def left_right_support_anchors(P, f0, f1, f2, closed, support_type=None):
    a = f0.anchors[1]
    b = f2.anchors[0] 
    n = P.shape[1]
    #pdb.set_trace() 
    if support_type is None:
        support_type = cfg.saliency_support_type

    #pdb.set_trace()
    if support_type==SUPPORT_EXTREMA: # cfg.support_until_extrema:
        a = f0.i
        b = f2.i
    elif support_type==SUPPORT_CONTACT:
        a = f0.anchors[1]
        b = f2.anchors[0]
    elif support_type==SUPPORT_ALL:
        if not closed:
            a = 0
            b = n-1
        else:
            # We usually traverse half contour on each side of feature
            # but there are cases in which there are only two adjacent features and these are asymmetric
            # the following co/mpensates for that
            lim_a = max(n//2-1, (f2.i - f1.i)%n)
            lim_b = max(n//2-1, (f1.i - f0.i)%n)
            a = (f1.i - lim_a)%n
            b = (f1.i + lim_b)%n
            # if f1.i == 0:
            #     pdb.set_trace()

    elif support_type==SUPPORT_INTERPOLATED: # cfg.interpolate_support:
        if cfg.support_uses_distance:
            speed = cfg.interpolation_exp_rise
            d0 = geom.distance(P[:,f0.i], f1.center)
            if d0 > f1.r:
                #speed = max(f0.r, 1)
                t = 1 - np.exp(speed - speed*d0/f1.r) #f1.r / d0
                d = (f0.anchors[1] - f0.i)%n
                a = (f0.i + int((1.-t)*d))%n
            d2 = geom.distance(P[:,f2.i], f1.center)
            if d2 > f1.r: # or f2.sign != f1.sign:
                #speed = max(f2.r,+1e-5
                t = 1 - np.exp(speed - speed*d2/f1.r) #1. - f1.r / d2
                d = (f2.i - f2.anchors[0])%n
                b = (f2.anchors[0] + int(t*d))%n
        else:
            # Radius based version
            if f0.r > f1.r: # or f0.sign != f1.sign:
                t = f1.r / f0.r
                d = (f0.anchors[1] - f0.i)%n
                a = f0.i + int(t*d)
                #a = f0.i
            if f2.r > f1.r: # or f2.sign != f1.sign:
                t = 1. - f1.r / f2.r
                d = (f2.i - f2.anchors[0])%n
                b = f2.anchors[0] + int(t*d)

            #b = f2.i
    elif support_type==SUPPORT_ALTERNATE:
        if f0.sign != f1.sign:
            a = f0.i
        else:
            a = get_contour_midpoint_index(f0.i, f1.i, P, closed) #anchors[1]
        if f2.sign != f1.sign:
            b = f2.i
        else:
            b = get_contour_midpoint_index(f1.i, f2.i, P, closed) #f2.anchors[0]

    return consolidate_support(P, a, b, f0, f1, f2, closed, support_type)

def CSF_contour_segment_and_extreum(P, f0, f1, f2, closed):
    """Get support contour segment for a CSF given its neighboring CSFs  
    
    Args:
        P (ndarray): Contour
        f0 (CSF): left neighboring CSF
        f1 (CSF): CSF
        f2 (CSF): right neighboring CSF
        closed (bool): True if contour is closed
    
    Returns:
        ndarray, CSF: contour segment and shifted CSF
    """
    n = P.shape[1]

    # if f1.i == 0:
    #     brk()

    if f0.i == f2.i: # Loop case
        Pv = get_contour_segment(P, f0.i, f2.i, closed)
        fs0 = shift_feature(-f0.i, f0, P, closed)
        fs1 = shift_feature(-f0.i, f1, P, closed)
        fs2 = shift_feature(-f0.i, f2, P, closed)

        return Pv, fs0, fs1, fs2
    #endif

    a, b = left_right_support_anchors(P, f0, f1, f2, closed)

    Pv = get_surrounding_contour_segment(P, a, f1.i, b, closed=closed)

    fs0 = shift_feature(-a, f0, P, closed)
    fs1 = shift_feature(-a, f1, P, closed)
    fs2 = shift_feature(-a, f2, P, closed)

    if len(Pv.shape) < 2:
        return np.zeros((2,0)), fs0, fs1, fs2 #Invalid
    # wat?
    #f = f._replace(anchors=(f.anchors[0], min(f.anchors[1], Pv.shape[1]-1)))
    return Pv, fs0, fs1, fs2
#endf

def compute_depth_saliency(P, f0, f1, f2, closed=True, debug_draw=False, get_area=False):
    #if is_minimum(f1):
    #    debug_draw=True
    Pv, fs0, fs1, fs2 = CSF_contour_segment_and_extreum(P, f0, f1, f2, closed)
    if Pv.shape[1] < 3:
        if get_area:
            return 0, Pv
        return 0

    #return compute_depth_saliency_contour_max_h(P, Pv, f1, debug_draw, get_area)
    return compute_depth_saliency_contour_all(P, Pv, fs0, fs1, fs2, debug_draw, get_area)

def angle_bisector(A, B, C):
    dc = B - A
    db = C - A
    c = norm(dc)
    b = norm(db)
    a = norm(B-C)
    d = np.sqrt(((b*c)/(b + c)**2)*
                ((b+c)**2 - a**2))
    return geom.normalize((dc/c + db/b))*d


def compute_depth_saliency_contour_all(P, Pv, fl, f, fr, debug_draw=False, get_area=False):

    # traverse both sides of feature
    #L = Pv[:, :f.anchors[0]+1][:,::-1]
    #R = Pv[:, f.anchors[1]:]
    a = fl.anchors[1]+1
    b = fr.anchors[0]
    L = Pv[:, :a][:,::-1]
    R = Pv[:, b:]

    H = []
    segs = []
    p = f.extrema_pos

    # if L.shape[1] < 1:
    #     L = Pv[:,f.anchors[0]].reshape(-1,1)
    # if R.shape[1] < 1:
    #     R = Pv[:,f.anchors[1]].reshape(-1,1)

    extremities = []
    endpoints = []
    # Simple polygon heuristic
    # each new segment connecting points must be longer than the previous
    # and it should not intersect it
    prev = None
    maxl = L.shape[1]-1
    maxr = R.shape[1]-1

    m = max(L.shape[1], R.shape[1])

    for i in range(m):
        il = min(i, maxl)
        ir = min(i, maxr)
        pl = L[:,il]
        pr = R[:,ir]

        #pdb.set_trace()
        extremities.append((il, ir))
        endpoints.append((pl, pr))

        if prev is not None:
            if il < maxl and geom.segment_intersection(pl, pr, prev[0], p)[0]:
                break
            if ir < maxr and geom.segment_intersection(pl, pr, prev[1], p)[0]:
                break
            if geom.distance(*prev) > geom.distance(pl, pr):
                break
        prev = (pl, pr)


        # #h = geom.point_segment_distance(p, pl, pr)
        # h = norm(angle_bisector(p, pl, pr)) # TODO consider using bisector here
        # if curvature_sign(pl, p, pr) != f.sign:
        #     h = -h
        # if np.isnan(h):
        #     h = -10
        # H.append(h)
        # segs.append((pl, pr))

    # Construct the contour area for saliency
    start_area = a - extremities[-1][0]
    end_area = b + extremities[-1][1]
    area_poly = Pv[:,start_area:end_area+1]

    if cfg.use_area_saliency:
        area = abs(geom.polygon_area(area_poly))
        disk_area = np.pi * f.r**2
        w = np.exp(-disk_area/(2*area))
    else:
        pl, pr = endpoints[-1]
        b = angle_bisector(p, pl, pr)
        h = norm(b)
        #plut.draw_line(p, p + b, 'b', linewidth=2.)
        w = np.exp(-f.r/h)

    if debug_draw and f.sign < 0:
        #pl, pr = segs[i]
        #plut.draw_line(pl, pr, 'b', linewidth=0.5, linestyle=':') #[0,0.3,0,7])

        #ph = p + angle_bisector(p, pl, pr)
        #ph = geom.project(p, pl, pr)
        #plut.draw_line(ph, p, 'r', linewidth=0.5)

        plut.fill_poly(area_poly, 'c', alpha=0.3)
        # plut.stroke_poly(P1o, plut.colors.cyan, closed=False, linewidth=1.5, alpha=alpha)
        # plut.stroke_poly(P2o, 'm', closed=False, linewidth=1.5, alpha=alpha)

    if get_area:
        return w, area_poly
    else:
        return w

### Curvature and 'depth' based saliency, different variants below
def saliency_depth(P, features, closed=True, debug_draw=False):
    m = len(features)
    saliency = []
    pts = []
    #pdb.set_trace()
    for i in range(m):
        f0 = features[(i-1)%m]
        f1 = features[i]
        f2 = features[(i+1)%m]

        x = compute_depth_saliency(P, f0, f1, f2, closed, debug_draw)

        #x = (depth/geom.distance(a, b)) * (s1/r) * (s2/r)
        saliency.append(x)
    return np.array(pts).T, np.array(saliency)


##########################################
## Drawing/reconstruction
def draw_feature_list(feature_list, S, clr='r', draw_types=False, labels=True, markersize=5., draw_disks=True, center_radius=0., disk_scale=1.):
    for features, P in zip(feature_list, S):
        draw_features(features, P, clr=clr, draw_types=draw_types, labels=labels, markersize=markersize, draw_disks=draw_disks, center_radius=center_radius, disk_scale=disk_scale)

def is_corner(f, P):
    #print('Cusp')
    #print((f.r, ds))
    #print(f.r/ds)
    ds = np.linalg.norm(P[:,1] - P[:,0])
    r = (f.r / ds)
    #if f.r == 0.01:
    #    plut.fill_circle(f.center, 3, 'r')
    #print('%s cusp ratio (%g / %g): %g'%(f.type, f.r, ds, r))
    return r < cfg.corner_radius_thresh #cfg.cusp_tolerance

def draw_features(features, S, clr='r', draw_types=False, labels=True, markersize=5., draw_disks=True, center_radius=1., disk_scale=1.):
    ''' Draw a list of features
        Input:
            features: feature list
            S: shape or contour
    '''

    if features and type(features[0]) == list:
        feature_list = features
        for features, P in zip(feature_list, S):
            draw_features(features, P, clr=clr, draw_types=draw_types, labels=labels, markersize=markersize, draw_disks=draw_disks, center_radius=center_radius, disk_scale=disk_scale)
        return

    if type(S) == list:
        S, _ = vma.flatten_shape(S)
    if not features:
        print("draw_features: empty!")
        return

    extrema = [f.i for f in features if f.type==FEATURE_POS_EXTREMUM or f.type==FEATURE_NEG_EXTREMUM]
    extremaneg = [f.i for f in features if f.type==FEATURE_NEG_EXTREMUM]
    extremapos = [f.i for f in features if f.type==FEATURE_POS_EXTREMUM]
    minima = [f.i for f in features if  f.type==FEATURE_POS_MINIMUM or f.type==FEATURE_NEG_MINIMUM]
    inflections = [f.i for f in features if f.type==FEATURE_INFLECTION]
    
    ds = np.linalg.norm(S[:,1] - S[:,0])
    
    if inflections:
        plut.draw_markers(S[:,inflections], 'g', markersize=markersize, marker='x', label='inflections' if labels else '')
        #plt.plot(S[0,inflections], S[1,inflections], 'gx', markersize=markersize, label='inflections' if labels else '')
    
    if extremaneg:
        cusps = [f.i for f in features if f.type==FEATURE_NEG_EXTREMUM if is_corner(f, S)]
        if cusps:
            plut.draw_markers(S[:,cusps], 'r', markersize=markersize, marker='v', label='cusp m-' if labels else '')
            #plt.plot(S[0,cusps], S[1,cusps], 'rv', markersize=markersize, label='cusp m-' if labels else '')
        extr = [f.i for f in features if f.type==FEATURE_NEG_EXTREMUM if not is_corner(f, S)] 
        if extr:
            plut.draw_markers(S[:,extr], 'r', markersize=markersize,  marker='o', label='m-' if labels else '')
            #plt.plot(S[0,extr], S[1,extr], 'ro', markersize=markersize, label='m-' if labels else '')
    if extremapos:
        cusps = [f.i for f in features if f.type==FEATURE_POS_EXTREMUM if is_corner(f, S)]
        if cusps:
            plut.draw_markers(S[:,cusps], 'c', markersize=markersize, marker='v', label='cusp M+' if labels else '')
            #plt.plot(S[0,cusps], S[1,cusps], 'cv', markersize=markersize, label='cusp M+' if labels else '')
        extr = [f.i for f in features if f.type==FEATURE_POS_EXTREMUM if not is_corner(f, S)] 
        if extr:
            plut.draw_markers(S[:,extr], 'c', markersize=markersize,  marker='o', label='M+' if labels else '')
            #plt.plot(S[0,extr], S[1,extr], 'co', markersize=markersize, label='M+' if labels else '')

        #plt.plot(S[0,extremapos], S[1,extremapos], 'co', markersize=markersize, label='M+' if labels else '')
    #plt.plot(S[0,extrema], S[1,extrema], 'ro', markersize=4., label='extrema (M+,m-)' if labels else '')
    if minima:
        plut.draw_markers(S[:,minima], 'b', markersize=markersize,  marker='o', label='minima (M-,m+)' if labels else '')
        #plt.plot(S[0,minima], S[1,minima], 'bo', markersize=markersize, label='minima (M-,m+)' if labels else '')
    
    if draw_disks:
        for i, f in enumerate(features):
            pos = S[:,f.i]
            if draw_types:

                plt.text(pos[0], pos[1], f.type + ' ' + str(f.sign), fontsize=9)
            #plt.text(pos[0], pos[1], str(i) + ' ' + f.type + ' ' + str(f.r), fontsize=10)
            #print f
            if f.r != cfg.inf_radius: #np.inf:
                if f.type==FEATURE_POS_MINIMUM or f.type==FEATURE_NEG_MINIMUM:
                    clr = 'b'
                else:
                    clr = 'r'
                clr = 'k'
                plut.stroke_circle(f.center, f.r, 'k', alpha=0.25) #, label='disks')
                if center_radius > 0.:
                    plut.fill_circle(f.center, center_radius, clr)

                plut.draw_line(f.center, S[:,f.anchors[0]], [0.5, 0.5, 0.5], linestyle=':')
                plut.draw_line(f.center, S[:,f.anchors[1]], [0.5, 0.5, 0.5], linestyle=':')


def draw_feature_arcs(features, P, clr, linewidth=2., label=''):
    for f in features:
        if not 'theta' in f.data:
            continue
        theta = f.data['theta']
        pts = arc_points(P[:,f.anchors[0]], P[:,f.anchors[1]], theta)
        plt.plot(pts[0,:], pts[1,:], clr, linewidth=linewidth, label=label)
        label=''
    
cyan = plut.colors.cyan
feature_colors = {'M+':'b','m-':'r', 'm+': cyan, 'M-':cyan, 'inflection':'k', 'transition':np.ones(3)*0.5, 'endpoint':'m'}
    
def draw_reconstruction(features, P, closed=False, labels=True, inflections=True, skip_types=[], linewidth=1.):
    ''' Draw reconstruction of a shape as a sequence of circular arcs and spiral segments
        Input:
            features: feature list
            P: contour (2 X n matrix)
    '''
    #print('drawing recons')
    #print(type(features[0]))
    if not features:
        return

    if type(features[0]) == list:
        S = P
        feature_list = features
        feature_list_extended = []
        for features, P in zip(feature_list, S):
            #print(type(features[0]))

            feature_list_extended += [draw_reconstruction(features, P, labels=labels, inflections=inflections, skip_types=skip_types, linewidth=linewidth)]
        return feature_list_extended



    # Check if we need the additional info
    needs_features = True
    needs_angles = True
    for f in features:
        if 'theta' in f.data:
            needs_angles = False
        
        if f.type==FEATURE_TRANSITION:
            needs_features = False
            #break

    if P.shape[1] < 3:
        return

    if needs_angles:
        features = compute_internal_angles(features, P)
        
    if needs_features:
        features = compute_transitions(features, P, closed)
        
    m = len(features)
    
    labels_rec={}
    linewidths={'M+':linewidth,'m-':linewidth, 'm+':linewidth, 'M-':linewidth, 'inflection':linewidth/2, 'transition':linewidth/2}
    
    debug_feats = [(i,f) for i,f in enumerate(features) if f.type == FEATURE_INFLECTION]
    
    for i, f in enumerate(features):
        pts = None
        if f.type in skip_types:
            continue

        if f.type == FEATURE_INFLECTION or f.type == FEATURE_TRANSITION:
            if not inflections:
                continue

            # Need to set data first
            if not 's' in f.data:
                continue

            #f1 = features[(i-1)%m]
            #f2 = features[(i+1)%m]
            s0, s1 = f.data['s']
            pts = es.euler_spiral(P[:,f.anchors[0]], P[:,f.anchors[1]], s0, s1, 100)
            if f.type == FEATURE_INFLECTION:
                plut.draw_marker(P[:,f.i], 'g', markersize=linewidth*3, marker='o') #, label='inflections' if labels else '')
            
            
            #pts = np.vstack([P[:,f1.anchors[1]], P[:,f2.anchors[0]]]).T
        elif f.type != FEATURE_ENDPOINT:
            # Need to set data first (compute_internal_angles)
            if 'theta' in f.data:
                theta = f.data['theta']
            else:
                theta = 0.
            if is_corner(f, P):
                theta = 0
            pts = arc_points(P[:,f.anchors[0]], P[:,f.anchors[1]], theta)
            #plut.stroke_circle(f.center, f.r, 'k', alpha=0.2, linestyle=':', linewidth=1.)
            clr = 'r'
            if is_minimum(f):
                clr = 'b'
        
        if pts is None:
            continue
        
        label = f.type
        if f.type in labels_rec or not labels:
            label = ''
        labels_rec[f.type] = 1
        clr = feature_colors[f.type]
        if f.type == FEATURE_TRANSITION:
            clr = plut.default_color(i)
        lw = 1.
        if f.type in linewidths:
            lw = linewidths[f.type]
        
        plt.plot(pts[0,:], pts[1,:], color=clr, linewidth=lw, label=label)
        # if not (f.type == FEATURE_INFLECTION or f.type == FEATURE_TRANSITION):
        #     plut.draw_marker(pts[:, pts.shape[1]//2], clr)
    # endfor
    return features
    

#######################################
# ABSOLUTE MINIMA
# Note: This code is not needed for font segmentation

def compute_local_minima(P, ds, features, closed=True, draw_steps=False, P_whole=None):
    ''' Compute extrema for each part defined between two extrema with the same sign'''
    if P_whole is None:
        P_whole = P
        
    if closed:
        features_loc = features + [features[0]]
    else:
        features_loc = features
        
    res = []
    for f1, f2 in zip(features_loc, features_loc[1:]):
        if f1.sign != f2.sign:
            continue #continue #pass #
        
        if draw_steps:
            plt.figure(figsize=(5,5))
            plt.title(str((f1.anchors[1], f2.anchors[0])))
            plut.stroke_poly(P_whole, 'k', closed=closed)
        local, Pv, zone = compute_segment_minima(P, ds, f1, f2, closed=closed, draw_steps=draw_steps)
        if draw_steps:
            plut.plt_setup()
            plut.set_axis_limits(P_whole, 7) #P_whole, 103) #3)
            plt.show()
            
            plt.figure(figsize=(5,3))
            plt.title('$\kappa$')
            K = geom.curvature(Pv)
            plt.plot(K)
            I = [f.i for f in local]
            plt.plot(I, K[I], 'ro')
            plt.plot()
        if local:
            #pdb.set_trace()
            local = [shift_feature(zone[0], f, P, closed) for f in local]

            res += local
    
    return res 
    
def get_pisa_point(P, a, b, center, r, neg=False):
    ''' Get PISA point given a disk and its generating points'''
    #http://mathnews.uwaterloo.ca/wp-content/uploads/2014/08/v111i6.pdf
    # From Dot product finds arc midpoint
    pa = P[:,a]
    pb = P[:,b]
    d1 = pa - center
    d2 = pb - center
    l = 1. + np.dot(d1,d2)/r**2
    m = (d1 + d2) / np.sqrt(2.*l)
    if neg:
        m = -m
    return center + m


def straight_line_mse(xy, reg=1e-5, debug_draw=False):
    ''' Returns mean squared error for a straight line fit'''
    from numpy.linalg import inv

    flip = False
    # Flip so largest std-dev is on x
    std = np.std(xy, axis=1)
    if std[1] > std[0]:
        flip = True
        xy = np.flipud(xy)
    n = xy.shape[1]
    X = np.vstack([np.ones(n), xy[0,:]]).T
    y = xy[1,:].reshape(-1,1)
    # Ridge regressionn
    n = xy.shape[1]
    X = np.vstack([np.ones(n), xy[0,:]]).T
    y = xy[1,:].reshape(-1,1)
    Theta = inv(X.T@X + np.eye(X.shape[1])*reg)@X.T@y
    
    if debug_draw:
        xy_hat = np.vstack([xy[0,:], (X@Theta).T])
        if flip:
            xy_hat = np.flipud(xy_hat)
        plt.plot(xy_hat[0,:], xy_hat[1,:], 'b')
        
    mse = (1./n)*((y - X@Theta).T @ (y - X@Theta))
    return mse


def linear_esat_disks(Pv):
    ''' Linear approximation of Layton's ESAT. '''

    Pv = geom.smoothing_spline(Pv.shape[1], Pv, smooth_k=cfg.minima_smooth_k)
    #Pv = geom.gaussian_smooth_contour(Pv, cfg.minima_smooth_sigma, closed=False)

    # Farthest Voronoi
    vor = vma.voronoi_diagram(Pv, True)
    #vma.draw_voronoi(vor, np.ones(3)*0.5)

    def at_infinity(v):
        return v[0] < 0 or v[1] < 0

    V = list(vor.vertices)

    # Only consider finite vertices
    disks = []
    
    for p, v in zip(vor.ridge_points, vor.ridge_vertices):
        v = np.asarray(v)

        if not at_infinity(v):
            center = (V[v[0]] + V[v[1]])/2
            pt = Pv[:,p[0]]
            r = geom.distance(center, pt)
            #plut.stroke_circle(center, r, np.ones(3)*0.5, linewidth=0.25, alpha=0.5)
            p = sort_anchors(p, Pv, False) 
            disks.append(vma.Disk(center, r, p, 0, False))

    if not disks:
        return (None, None), Pv_smooth

    radii = [d.r for d in disks]

    a = np.argmin(radii)
    b = np.argmax(radii)
    
    return [disks[a], disks[b]], Pv

def linear_esat(Pv):
    (da, db), Pv_smooth = linear_esat_disks(Pv)

    G = nx.Graph()
    if da is None:
        return G

    G.add_edge(0, 1)
    G.graph['disks'] = [da, db]
    G.graph['vpos'] = [da.center, db.center]
    G.graph['points'] = np.array(Pv.T)
    return G

def is_segment_straight(Pv, ds, tol=None, debug_draw=False):
    ''' Tests straightness of a contour segment with a least squares line fit'''
    if Pv.shape[1] < 4:
        return True
    if tol is None:
        tol = cfg.straightness_tolerance
    mse = straight_line_mse(Pv, debug_draw=debug_draw)
    straight = mse < (tol*ds)**2

    if straight:
        print((mse, (tol*ds)**2))
        #plut.stroke_poly(Pv, 'r', linewidth=3., closed=False)
    return straight

def is_contour_segment_straight(P, ds, start_feature, end_feature, closed, debug_draw=False):
    ''' Tests straightness of support segment'''
    a, b = start_feature.anchors[1],  end_feature.anchors[0]
    Pv = get_contour_segment(P, a, b, closed=closed) 
    return is_segment_straight(Pv, ds, debug_draw=debug_draw)

def compute_segment_minima(P, ds, start_feature, end_feature, closed, draw_steps=False):
    ''' Compute absolute minima for a part of the contour defined between two features'''
    clr = 'b'
    
    if draw_steps:
        #print (start_feature.anchors[1], end_feature.anchors[0])
        plut.stroke_circle(start_feature.center, start_feature.r, 'c', linewidth=0.5)
        plut.stroke_circle(end_feature.center, end_feature.r, 'c', linewidth=0.5)
    
    if is_endpoint(start_feature) or is_endpoint(end_feature):
        return [], P, (0,0)
    
    if is_contour_segment_straight(P, ds, start_feature, end_feature, closed, debug_draw=draw_steps):
        print('Straight segment')
        
        return [], P, (0,0)
    
    a, b = start_feature.i,  end_feature.i
    a, b = start_feature.anchors[1],  end_feature.anchors[0]

    #a, b = start_feature.anchors[0],  end_feature.anchors[1]
    
    n = P.shape[1]
    
    if not closed and (b-a) < 3:
        print('Bailing for open segment: ', a, b)
        return [], P, (0,0)
    
    #if closed and (a-b)%n < (b-a)%n:
    #    a, b = b, a
    if (b-a)%n < 3:
        print('Bailing for closed segment: ', a, b)
        return [], P, (0,0)
    
    Pv = get_contour_segment(P, a, b, closed=closed) #i, end_feature.i+1) #
    (da, db), Pv_smooth = linear_esat_disks(Pv)

    if draw_steps:
        plut.plot(Pv, 'c')
        plut.plot(Pv_smooth, 'm', linewidth=2.)

    if da is None:
        print('No ESAT disks')
        return [], P, (0,0)
    
    i = get_anchor_midpoint_index(db.anchors, Pv, False) 
    ftype = FEATURE_POS_MINIMUM
    if start_feature.sign < 0:
        ftype = FEATURE_NEG_MINIMUM
    f = Feature(i=i,
                   center=db.center, 
                   extrema_pos=Pv[:,i],
                   r=db.r,
                   anchors=db.anchors,
                   type=ftype,
                   sign=start_feature.sign,
                   data={})
    #pdb.set_trace()
    # Compute limits to avoid contact region overlaps
    limit_a = (start_feature.anchors[1] - start_feature.i)%n
    limit_b = (end_feature.i - end_feature.anchors[0])%n
    #limit_a = (start_feature.anchors[1] - start_feature.i)%n
    #pdb.set_trace()
    limits = (limit_a, Pv_smooth.shape[1]-limit_b)
    f = expand_and_recompute_midpoint(Pv_smooth, f, closed=False, thresh=ds*cfg.minima_expansion_tol, limits=limits) #ds*2)
    if draw_steps:
        plut.fill_circle(Pv_smooth[:,f.i], 4, 'r')
    return [f], Pv, (a,b)


def compute_segment_minima_OLDDD(P, ds, start_feature, end_feature, closed, draw_steps=False):
    ''' Compute absolute minima for a part of the contour defined between two features'''
    clr = 'b'
    
    if draw_steps:
        #print (start_feature.anchors[1], end_feature.anchors[0])
        plut.stroke_circle(start_feature.center, start_feature.r, 'c', linewidth=0.5)
        plut.stroke_circle(end_feature.center, end_feature.r, 'c', linewidth=0.5)
    
    if is_contour_segment_straight(P, ds, start_feature, end_feature, closed):
        return [], P, (0,0)

    a, b = start_feature.i+1,  end_feature.i-1 
    #a, b = start_feature.i,  end_feature.i #start_feature.anchors[1],  end_feature.anchors[0]

    #a, b = start_feature.anchors[0],  end_feature.anchors[1]
    
    n = P.shape[1]
    
    if not closed and (b-a) < 3:
        print('Bailing for open segment: ', a, b)
        return [], P, (0,0)
    
    #if closed and (a-b)%n < (b-a)%n:
    #    a, b = b, a
    if (b-a)%n < 3:
        print('Bailing for closed segment: ', a, b)
        return [], P, (0,0)
    
    Pv = get_contour_segment(P, a, b, closed=closed) #i, end_feature.i+1) #
    try:
        Pv_samp = geom.uniform_sample_spline_n(Pv,40) # geom.chord_length(Pv)/40) #geom.uniform_sample_n(Pv, 40, closed=False)
        Pv_smooth = geom.gaussian_smooth_contour(Pv_samp, cfg.part_smooth_sigma, closed=False)
    except TypeError:
        print('Failed computing uniform sampling for ')
        print(Pv)
        return [], P, (0,0)
    
    # Scale threshold depending on sampling step
    ds_local = np.mean(geom.chord_length(Pv_samp)) 
    #print("thresh farthest * ds: " + str(cfg.vma_thresh_farthest*ds_local))
    # compute features
    features, MA, vor, delu = sym_extrema(Pv_smooth, ds_local, closed=False, farthest=True, full_output=True, vma_thresh=cfg.vma_thresh_farthest*ds_local)
    
    if draw_steps:    
        plut.stroke_poly(Pv_smooth, 'r', linewidth=2., closed=False)
        #vma.draw_voronoi(vor, 'k', alpha=0.3, farthest=True)
        #vma.draw_pruned_delaunay(MA, delu, 'c', alpha=0.2)
        vma.draw_skeleton(MA)
        #plut.draw_line(Pv_smooth[:,0], Pv_smooth[:,-1], 'k', alpha=0.7)
        
    radii = [f.r for f in features]
    if not radii:
        print('No features')
        return [], P, (0,0)
    
    i = np.argmax(radii)
    f = features[i]
    # This seems to be important. Even though the (farthest) Delaunay triangle
    # provides with a good fit of the curvature radius at the minima, its points do not 
    # necessarily indicate well the projection on the contour of the osculating circle.
    # With a small threshold it is possible to fix this by expanding the anchor points to 
    # fully support the circle.
    f = expand_and_recompute_midpoint(Pv_smooth, f, closed=False)

    
    # if start_feature.sign != end_feature.sign:
    #     f = f._replace(type=FEATURE_INFLECTION,
    #                anchors=(start_feature.anchors[1], end_feature.anchors[0]))
    if True:
        if True: #f.r > 10: # need a threshold here for almost straight segments that can't be maxima
            if start_feature.sign < 0:
                f = f._replace(type=FEATURE_NEG_MINIMUM, sign=start_feature.sign)
            else:
                f = f._replace(type=FEATURE_POS_MINIMUM, sign=start_feature.sign)

    # Get point along contour cloest to pisa extremity
    pisap = get_pisa_point(Pv_smooth, f.anchors[0], f.anchors[1], f.center, f.r)

    ratio = float(Pv.shape[1]) / Pv_samp.shape[1]
    f = f._replace(i=int(ratio*f.i), anchors=sort_anchors((int(ratio*f.anchors[0]), int(ratio*f.anchors[1])), Pv, closed=False))
    dists = [np.linalg.norm(p - pisap) for p in Pv.T]
    f = f._replace(i=np.argmin(dists))    

    r_thresh = cfg.r_thresh

    if is_minimum(f):
        r_thresh = cfg.minima_r_thresh

    if f.r < r_thresh:
        features_ok = [f]
    
    if features_ok and draw_steps:
        plt.plot(Pv_smooth[0,:], Pv_smooth[1,:], 'b')
        draw_features(features_ok, Pv, clr, markersize=cfg.debug_draw_markersize)
    
    if get_smoothed:
        Pv = Pv_smooth
    
    return features_ok, Pv, (a,b)


#######################################
# INFLECTIONS

def compute_local_inflections(P, ds, features, closed=True, draw_steps=False, P_whole=None):
    ''' Compute inflection for each part defined between two extrema with different sign'''
    if P_whole is None:
        P_whole = P
        
    if closed:
        features_loc = features + [features[0]]
    else:
        features_loc = features
        
    res = []
    for f1, f2 in zip(features_loc, features_loc[1:]):
        if f1.sign != f2.sign and is_extremum(f1) and is_extremum(f2):
        
            if draw_steps:
                plt.figure(figsize=(5,5))
                plt.title('infl ' + str((f1.anchors[1], f2.anchors[0])))
                plut.stroke_poly(P_whole, 'k', closed=closed)
            local, Pv, zone = compute_segment_inflection(P, ds, f1, f2, closed=closed, draw_steps=draw_steps)
            if draw_steps:
                plut.plt_setup()
                plut.set_axis_limits(P_whole, 3) #3)
                plt.show()
                
                plt.figure(figsize=(5,3))
                plt.title('$\kappa$')
                K = geom.curvature(Pv)
                plt.plot(K)
                I = [f.i for f in local]
                plt.plot(I, K[I], 'ro')
                plt.plot()
            if local:                      
                local = [shift_feature(zone[0], f, P, closed) for f in local]
                res += local
    
    return res 


def to_kappa(P, a, b, s0, s1):
    ''' Curvature given Euler Spiral parameters'''
    d = np.linalg.norm(P[:,a] - P[:,b])
    l = np.sqrt((es.C_(s1) - es.C_(s0))**2 + (es.S_(s1) - es.S_(s0))**2)
    return np.pi * s0 * (l/d),  np.pi * s1 * (l/d), l/d # abs(sall[1] - sall[0])
    
def to_s(P, a, b, k0, k1, ld):
    ''' Euler spiral parameters given curvature values'''
    return k0 / (np.pi * ld), k1 / (np.pi * ld)

def compute_segment_inflection(P, ds, start_feature, end_feature, closed, draw_steps=False):
    ''' Compute inflections for a part of the contour defined between two features'''
    clr = 'g'
    
    if draw_steps:
        plut.stroke_circle(start_feature.center, start_feature.r, 'm')
        plut.stroke_circle(end_feature.center, end_feature.r, 'm')

    #a, b = start_feature.i,  end_feature.i #start_feature.anchors[1],  end_feature.anchors[0]
    a, b = start_feature.anchors[1],  end_feature.anchors[0]
    #a, b = start_feature.i,  end_feature.i
    #a, b = start_feature.anchors[0],  end_feature.anchors[1]
    
    n = P.shape[1]
    # Need to check this! but I guess it may happen that we have an overlap
    if not closed and b-a < 4:
        return [], P, (0,0)
    
    #if closed and (a-b)%n < (b-a)%n:
    #    a, b = b, a
    if (b-a)%n < 4:
        s0 = -start_feature.sign*0.001
        s1 = -end_feature.sign*0.001
        infl = abs(b-a)//2
        f = Feature(i=int(infl),
                   center=P[:,(a+infl)%n],
                   extrema_pos=P[:,(a+infl)%n],
                   r=np.inf,
                   anchors=(infl,infl+1),
                   type=FEATURE_INFLECTION,
                   sign=0,
                   data={'branch_len':1e-5, 's':(s0,s1)})
        return [f], np.zeros((2,0)), (a,b)
    
    Pv = get_contour_segment(P, a, b)
    k0 = 1. / start_feature.r*start_feature.sign
    k1 = 1. / end_feature.r*end_feature.sign
    d = geom.chord_length(Pv, closed=False)
    B = np.sqrt(d / (abs(k1 - k0)))
    spio2 = np.sqrt(np.pi * 0.5)
    s0, s1 = np.clip((-k0*B*spio2, -k1*B*spio2), -1., 1.)

    #if abs(s1 - s0) > cfg.clothoid_nofit_thresh:        
    s0, s1 = es.fit_clothoid(geom.uniform_sample_n(Pv, 40, closed=False), (s0, s1), tol=cfg.clothoid_opt_xtol, debug_draw=cfg.clothoid_debug_steps)
    
    s0, s1 = np.clip((s0, s1), -np.pi, np.pi)   
    if abs(s0 - s1) < 1e-8:
        s1 = s0 + 1e-8
    
    # Adjust curvature if estimate is near to bounding features
    if cfg.refine_spiral_curv:
        k0, k1, ld = to_kappa(P, a, b, s0, s1)
        khat_0 = 1. / start_feature.r * -start_feature.sign 
        khat_1 = 1. / end_feature.r * -end_feature.sign 
        shat_0, shat_1 = to_s(P, a, b, khat_0, khat_1, ld)

        if abs(k0 - khat_0) < 0.05:
            s0 = shat_0
        if abs(k1 - khat_1) < 0.05:
            s1 = shat_1

    #s0, s1 = es.fit_clothoid(Pv, (-0.8*start_feature.sign, -0.8*end_feature.sign))
    if np.sign(s1) != np.sign(s0):
        infl_t = abs(s0) / abs(s1 - s0)
        infl = int(infl_t * Pv.shape[1])
    else:
        # with same sign get mid point (hack?)
        infl = Pv.shape[1]//2 
            
    f = Feature(i=int(infl)%P.shape[1],
                   center=Pv[:,infl%P.shape[1]],
                   extrema_pos=Pv[:,infl%P.shape[1]],
                   r=np.inf,
                   anchors=(0, Pv.shape[1]),
                   type=FEATURE_INFLECTION,
                   sign=0,
                   data={'branch_len':1e-5, 's':(s0,s1)})
                   
    if draw_steps:    
        plut.stroke_poly(Pv, 'r', linewidth=2., closed=False)
        
    features_ok = [f]
    
    if features_ok and draw_steps:
        draw_features(features_ok, Pv, clr, markersize=cfg.debug_draw_markersize)
        pts = es.euler_spiral(Pv[:,0], Pv[:,-1], s0, s1, 100)
        plt.plot(pts[0,:], pts[1,:], 'g')
    return features_ok, Pv, (a,b)
    

##########################################
## ALTERNATIVE SALIENCY MEASURES, for test purposes 

### Turning-angle surprisal based
# Feldman and Singh (2005) Information along contours and object boundaries
def saliency_surprisal(P, features, closed=True, scale=10.):
    m = len(features)
    saliency = []
    pts = []
    for i in range(m):
        f = features[i]
        if f.type == FEATURE_INFLECTION: #not (is_extremum(f) or is_minimum(f)):
            continue

        pts.append(P[:,f.i])

        a, b = f.anchors

        n1 = f.center - P[:,a]
        n2 = f.center - P[:,b]
        t1 = geom.perp(n1)
        t2 = geom.perp(n2)
        theta = geom.angle_between(n1, n2)
        #v = object_angle_importance(theta, scale) #10. / (np.cos(theta/2)) - 10

        v = 1. - np.cos(theta)
        if not (is_extremum(f) or is_minimum(f)):
            v = 0

        saliency.append( v ) #aof ) #f1.data['branch_len'] / f1.r ) #1. - np.cos(geom.angle_between(d1, d2) ) )#d / d2)

    return np.array(pts).T, np.array(saliency)

### Branch length based
def saliency_branch_length(P, features, closed=True, scale=10.):
    m = len(features)
    saliency = []
    pts = []
    W = geom.cum_chord_lengths(P, closed=closed)

    for i in range(m):
        f0 = features[(i-1)%m]
        f = features[i]
        f2 = features[(i+1)%m]

        f = features[i]
        if not 'branch_len' in f.data:
            continue
        #if f.type == FEATURE_INFLECTION: #not (is_extremum(f) or is_minimum(f)):
        #    continue

        pts.append(P[:,f.i])

        a, b = f0.anchors[1], f2.anchors[0]
        d = min(abs(W[a] - W[b]), abs(W[b] - W[a]))
        print(('d', d, 'bl', f.data['branch_len']))
        saliency.append(f.data['branch_len']/(d+1e-5))

    return np.array(pts).T, np.array(saliency)

### Stickout based
# DeWinters and Wagemans (2008) Perceptual saliency of points along the contour of everyday objects: A large-scale study
# Zusne
def saliency_compactness(P, features, closed=True):
    m = len(features)
    saliency = []
    pts = []
    for i in range(m):
        f0 = features[(i-1)%m]
        f1 = features[i]
        f2 = features[(i+1)%m]

        if is_extremum(f1):
            X, f0, f, f2 = CSF_contour_segment_and_extreum(P, f0, f1, f2, closed)
            pts.append(P[:,f1.i])
            saliency.append( isoperimetric_quotient(X) ) #abs(geom.angle_between(d1, d2) ) )#d / d2)
    return np.array(pts).T, np.array(saliency)

### Stickout based
# DeWinters and Wagemans (2008) Perceptual saliency of points along the contour of everyday objects: A large-scale study
# Hoffman
def saliency_stickout(P, features, closed=True):
    m = len(features)
    saliency = []
    pts = []
    n = P.shape[1]
    for i in range(m):
        f0 = features[(i-1)%m]
        f1 = features[i]
        f2 = features[(i+1)%m]

        if True: #is_extremum(f1):
            X, f0, f, f2 = CSF_contour_segment_and_extreum(P, f0, f1, f2, closed)
            pts.append(P[:,f1.i])
            #d = np.linalg.norm(P[:,f0.anchors[1]] - P[:,f2.anchors[0]])
            #d2 = np.linalg.norm(P[:,f1.anchors[0]] - P[:,f1.anchors[1]])
            #d1 = geom.perp(P[:,f1.i] - P[:,f0.anchors[1]])
            #d2 = geom.perp(P[:,f2.anchors[0]] - P[:,f1.i])
            d1 = geom.perp(P[:,f1.i] - P[:,f0.i])
            d2 = geom.perp(P[:,f2.i] - P[:,f1.i])
            s = geom.chord_length(X) / np.linalg.norm(X[:,0] - X[:,-1])
            if f1.type == FEATURE_INFLECTION:
                s = 0.
            saliency.append(s) #isoperimetric_quotient(X) ) #abs(geom.angle_between(d1, d2) ) )#d / d2)
    return np.array(pts).T, np.array(saliency)

### Turning angle based
def saliency_turning_angle(P, features, closed=True):
    m = len(features)
    saliency = []
    pts = []
    for i in range(m):
        f0 = features[(i-1)%m]
        f1 = features[i]
        f2 = features[(i+1)%m]

        X, f0, f, f2 = CSF_contour_segment_and_extreum(P, f0, f1, f2, closed)

        pts.append(P[:,f1.i])
        #d = np.linalg.norm(P[:,f0.anchors[1]] - P[:,f2.anchors[0]])
        #d2 = np.linalg.norm(P[:,f1.anchors[0]] - P[:,f1.anchors[1]])
        #d1 = geom.perp(P[:,f1.i] - P[:,f0.anchors[1]])
        #d2 = geom.perp(P[:,f2.anchors[0]] - P[:,f1.i])
        d1 = geom.perp(P[:,f1.i] - P[:,f0.i])
        d2 = geom.perp(P[:,f2.i] - P[:,f1.i])
        s = 1. - np.cos(geom.angle_between(d1, d2) )

        saliency.append(s)#d / d2)
    return np.array(pts).T, np.array(saliency)

### Shape ratio based
# Leymarie, F. and Levine, Martin D. (2007) Curvature Morphology
def saliency_shape_ratio(P, features, closed=True):
    m = len(features)
    saliency = []
    pts = []
    for i in range(m):
        f0 = features[(i-1)%m]
        f1 = features[i]
        f2 = features[(i+1)%m]

        shape_f = 1. / shape_factor(P, features, i, closed)
        saliency.append( shape_f ) #aof ) #f1.data['branch_len'] / f1.r ) #1. - np.cos(geom.angle_between(d1, d2) ) )#d / d2)
    return np.array(pts).T, np.array(saliency)

def shape_factor(P, features, i, closed):
    m = len(features)
    f1 = features[i]
    f0 = features[(i-1)%m]
    f2 = features[(i+1)%m]
    r = features[i].r
    X, f0, f, f2 = CSF_contour_segment_and_extreum(P, f0, f1, f2, closed)
    #print (f0.i, f1.i, f2.i), Pv.shape
    dists = [geom.point_line_distance(p, X[:,0], X[:,-1]) for p in X.T]
    #maxdist = np.max(dists)
    halfh = np.mean(dists)
    #print (f0.i, f2.i, P.shape)
    area = abs(geom.polygon_area(X)) #P[:,f0.i:f2.i]))
    #plut.stroke_poly(Pv, 'b')
    area_avg = area / np.linalg.norm(X[:,0] - X[:,-1]) # This actually corresponds to half height for the triangle
    #print (maxdist/2, halfh, area_avg)
    return   r / halfh


# DEPRECATED "DEPTH" SALIENCY METHODS
def compute_depth_saliency_contour_angle(P, Pv, f, debug_draw=False):
    # Angle based variant of the above
    a, b = Pv[:,0], Pv[:,-1]
    r = f.r
    # <- THIS
    P1 = Pv[:,:f.i+1]
    P2 = Pv[:,f.i:]
    
    if debug_draw and f.sign < 0:
        clr = [0, 0.6, 0.9]
        alpha = 1.
        plut.stroke_poly(Pv, np.ones(3)*0.5, closed=False, linewidth=2)
        
    #pdb.set_trace()
    s1 = geom.chord_length(P1)
    s2 = geom.chord_length(P2)
    eps = 1e-05
    s = max(min(s1, s2), 1e-05)

    use_perpendicular = False
    if s > 1e-05 and f.sign < 0:
        P1 = geom.path_subset_of_length(P1[:,::-1], s)
        P2 = geom.path_subset_of_length(P2, s)
        a, b = P1[:,-1], P2[:,-1]
        p = P2[:,0]
        clr = [1., 0.5, 0]
        alpha = 1
            
        if use_perpendicular:
            h = geom.point_segment_distance(Pv[:,f.i], a, b)
        else:
            da = a - p
            db = b - p
            beta = abs(geom.angle_between(da, db)) #geom.angle_between(da, db)) #b - p, p - a))
            h = np.cos(beta/2)*s
            w = np.exp(-r/h)

        if debug_draw and f.sign < 0:
            plut.stroke_poly(P1, 'r', closed=False, linewidth=0.75, alpha=alpha)
            plut.stroke_poly(P2, 'b', closed=False, linewidth=0.75, alpha=alpha)
            
            plut.draw_line(a, b, np.ones(3)*0.5, linewidth=0.5, linestyle=':') #[0,0.3,0,7])
            ph = geom.project(Pv[:,f.i], a, b)
            #ph = p + geom.normalize(da + db)*h
            plut.draw_line(ph, Pv[:,f.i], 'r', linewidth=0.5) #[0.5,0.6,0.3])
            # if w < 1e-4:
            #     plt.text(*Pv[:,f.i], '%e'%(w), fontsize=7)
            # else:
            #     plt.text(*Pv[:,f.i], '$%.2f, %.2f, %.2f$'%(w, h, f.r), fontsize=7)

        return w
    else:
        return 0.
#endf


#######################################
# DEPRECATED code below

def intersect_last(P):
    ''' dynamic_symmetry_extrema helper: Utility function to test self intersections'''
    n = P.shape[1]
    if n < 6:
        return -1
    a, b = P[:,-2], P[:,-1]
    for i in range(n-3):
        p1, p2 = P[:,i], P[:,i+1]
        res = geom.intersect_proper(a, b, p1, p2)
        res, ins = geom.segment_intersection(a, b, p1, p2)
        if res:
            # plt.figure(figsize=(10,10))
            # plut.stroke_poly(P, 'k', closed=False)
            # #plut.stroke_poly(P[:, i:j], 'r', closed=False)
            # plut.draw_line(a, b, 'r', linewidth=2.)
            # plut.draw_line(p1, p2, 'b', linewidth=2.)
            # plut.show()
            
            return i+1
    return -1

def dynamic_symmetry_extrema(P, closed=False):
    ''' DEPRECATED. Path based extrema/features computation. 
        Iterative version of the above, especially useful for open and self intersecting contours.
        In general for non-self intersecting contours the sym_extrema function will be MUCH faster and produce the same results.
        Input:
            P: a contour in (2 X n) matrix format
            closed: open/closed contour
        Output:
            features: list of Feature namedtuples (see above)
    '''

    if P.shape[1] < 4:
        return []
    
    start_j = 0
    j = start_j
    i = 0
    
    evt = None
    evt_count = 0
    
    n_pts = P.shape[1]

    features = []
    
    loop_ind = None

    MA_prev = None
    E_prev = []
    MA_stable = None
    E_stable = []
    
    P_prev = Pv = []
    
    def feq(a, b, eps=0.0001):
        if abs(a-b) < eps:
            return True
        return False
    # endfunc

    P_orig = P
    if closed:
        P = np.hstack([P, P])
        lastp = None
    else:
        lastp = n_pts
    
    
    dists = []
    debug_print("Computing extrema for %d points\n"%(P.shape[1]))
    finished = False
    while not finished: 
        j += 1
        
        if True: #cfg.verbose:
            utils.progress_bar(float(j)/(P.shape[1]-1))

        # exit if we reach last point (or loop to the first extrema for closed contours)
        if lastp is not None and j > lastp:
            j = lastp
            finished = True
            print("Finished")

        # Contour section
        P_prev = Pv
        Pv = P[:,i:j]

        # VMA and extremities 
        try:
            E, MA = vma.voronoi_skeleton([Pv], cfg.vma_thresh) 
        except QhullError:
            continue
        
        # No extrema? next iteration      
        if not E:
            continue

        # Check self intersection
        if not loop_ind:
            ins = intersect_last(P[:, i:j])
            if ins > -1:
                loop_ind = j - i
                loop_ind2 = ins
                
        
        # Check stability
        stable = False 
        if len(E) != 1: 
            stable = False
        else: # E and E_prev:
            stable = True
#                    
        if cfg.draw_steps > 1: 
            plt.figure(figsize=(10,10))
            plt.title('step')
            debug_draw_part(P_orig, Pv, MA, E, get_extrema(features))
            plut.plt_setup()
            plut.set_axis_limits(P_orig)
            plt.show()       
                
        if stable and not loop_ind:
            E_stable = E
            MA_stable = MA        
        if loop_ind:
            MA_stable = MA_prev
            E_stable = E_prev
        
        E_prev = E
        MA_prev = MA
            
        # check for new events        
        evt = None
        if j == lastp:
            debug_print('evt: END')
            finished = True
            evt = 'END'
        elif loop_ind is not None:
            evt = 'LOOP'
            print('Loop')
            evt_count += 1
            debug_print('evt: LOOP')
        elif not stable: # and E_stable: # and E_stable: 
            evt = 'NEW'
            debug_print('evt: ' + evt)
            evt_count += 1
            
        # perhaps way to go is to remove axes one at the time ......... and recompute MA
        # (for "cervo" missing points)
           
        
        def compute_sign_and_type(P, f):
            refnode = f.data['node']
            MA = f.data['MA']
            bors = list(MA.neighbors(refnode))
            if bors:
                refnode = bors[0]
            v = get_vertex_pos(refnode, MA)
            #sign = curvature_sign(P[:,f.anchors[0]], P[:,f.i], P[:,f.anchors[1]])
            sign = curvature_sign(P[:,f.anchors[0]], v, P[:,f.anchors[1]])
            if sign>0:
                ftype = FEATURE_POS_EXTREMUM
            else:
                ftype = FEATURE_NEG_EXTREMUM
            return f._replace(sign=sign, type=ftype)
            
        def compute_degenerate_sign(P, f):
            sign = -curvature_sign(P[:,f.anchors[0]], P[:,f.i], P[:,f.anchors[1]])
            if sign>0:
                ftype = FEATURE_POS_EXTREMUM
            else:
                ftype = FEATURE_NEG_EXTREMUM
            return f._replace(sign=sign, type=ftype)   

        def add_features(s, Pv_s, MA_s, E_s):    
            if E_s:                    
                disks = MA_s.graph['disks']
                exts, mid_pts, extrema_pos, proj_pts, E_s = get_extremity_features(Pv_s, MA_s, E_s, closed=False)
                for e, mp, pp, ep in zip(exts, mid_pts, proj_pts, extrema_pos):
                    p1, m, p2 = P[:, pp[0] + s], P[:, mp + s], P[:, pp[1] + s] #P[:, mp + s] #data[e]['center']
                    ok = True
                    c = disks[e].center
                    r = disks[e].r
                    
                    if r > cfg.r_thresh:
                        continue

                    p = sort_anchors(disks[e].anchors, Pv_s, closed=False)

                    features.append(compute_sign_and_type(P_orig, Feature(i=(mp + s)%n_pts,
                                    center=disks[e].center,
                                    extrema_pos = ep,
                                    r=disks[e].r,
                                    anchors=((p[0]+s)%n_pts, (p[1]+s)%n_pts),
                                    type=' ',
                                    sign=0,
                                    data={'branch_len': MA_s.graph['data']['branch_lengths'][e], 'MA':MA, 'node':e})))
                if cfg.draw_steps:
                    plt.figure(figsize=(10,10))
                    plt.title('Stable  %d'%(s))
                    debug_draw_part(P_orig, Pv_s[:,:-1], MA_s, E_s, get_extrema(features))
                    plut.plt_setup()
                    plut.set_axis_limits(P_orig)
                    plt.show()   
               
                            
        if evt: # and E_stable: # and E_prev: #stable:
            if cfg.draw_steps:
                plt.figure(figsize=(10,10))
                plt.title('Unstable for evt=' + evt)
                debug_draw_part(P_orig, Pv, MA, E, get_extrema(features))
                plut.plt_setup()
                plut.set_axis_limits(P_orig)
                plt.show()        
                
            add_features(i, Pv, MA_stable, E_stable)
            
            # if degenerate loop force an extrema at it:
            #if evt == 'LOOP':
            #    print ('Found loop, dist is: ',abs(loop_ind - loop_ind2))
            if evt == 'LOOP' and abs(loop_ind - loop_ind2) < 9:
                debug_print('Degenerate loop')
                il = i + (loop_ind + loop_ind2)//2
                if True:
                    features.append(compute_degenerate_sign(P_orig, Feature(i=il%n_pts,
                                            center=P[:,il],
                                            extrema_pos = P[:,il],
                                            r=0.01,
                                            anchors=(il, il),
                                            type=' ',
                                            sign=0,
                                            data={'branch_len': 1000., 'MA':MA_stable, 'node':node})))
                
            exts, mid_pts, extrema_pos, proj_pts, E = get_extremity_features(Pv, MA, E, closed)
            
            # For closed contours, compute wrapped end point
            if lastp is None and len(features) >= 2:
                lastp = n_pts + np.min(features[-1].anchors)
                
#            if cfg.draw_steps:
#                plt.figure(figsize=(10,10))
#                plt.title('Stable')
#                draw_part(P_orig, Pv, MA_s, E_s, extrema)
#                plut.plt_setup()
#                plt_set_extent(P_orig)
#                plt.show()   
                
            # Select next start point
            if evt == 'END':
                pass
            if evt == 'LOOP':
                #print('loop')
                i = loop_ind + i
            else:
                i = max(proj_pts[-1][0] + i, i) #max(pp_new[0] - 1 + s, 0) #max(pp_new[1] + s, 0) #max(pp_new[0] - 1 + s, 0)
                #j = j-1 # This was here but not sure why. Need to comment this kinda stuff next time
                # But it ended up creating an infinite loop. 
           
            loop_ind = None
            
            MA_stable = None
            E_stable = []
            
    debug_print("Finished computing MA"+str((i,j)))
    debug_print("s original: %d, loop_pt: %d"%(P_orig.shape[1], lastp))
    debug_print("Sorting " + str(len(features)) + " features")
    features = sort_features(P, features, closed)

    return merge_features(features, P, closed)
    #return features

def metric_euclidean(a, b):
    return np.linalg.norm(b - a)

def metric_hyperbolic(a, b):
    #return np.linalg.norm(b[:2] - a[:2]) - a[2] - b[2]
    return max(0., np.linalg.norm(b[:2] - a[:2]) - (a[2] - b[2]))

def distance_between_disks(fi, fj, P, closed):
    pi = np.concatenate([fi.center, [fi.r]])
    pj = np.concatenate([fj.center, [fj.r]])
    return metric_euclidean(pi, pj)/max(pi[-1], pj[-1])

def distance_along_outline(fi, fj, P, closed):
    n = P.shape[1]
    if closed:
        if (fi.i - fj.i)%n < (fj.i - fi.i)%n:
            fi, fj = fj, fi
    
    X = get_contour_segment(P, fi.i, fj.i, closed)
    r = min(fi.r, fj.r)
    if len(X.shape) < 2 or X.shape[1] < 2 or geom.chord_length(X) < r*0.5:
        return (1. - cfg.merge_thresh)*0.5
    return 1 - cfg.merge_thresh + 1.

def distance_max_overlap(fi, fj, P, closed):
    return geom.circle_overlap_max(fi.center, fi.r, fj.center, fj.r)

def safe_asin(x):
    return np.arcsin(np.clip(x, -1, 1))

def distance_overlap_and_vicinity(fi, fj, P, closed):
    a = fi.extrema_pos
    b = fj.extrema_pos
    theta = 2. * safe_asin( geom.distance(a, b) / (2*min(fi.r, fj.r)))
    theta = 2. * safe_asin( geom.distance(a, b) / (2*max(fi.r, fj.r)))
    
    if abs(theta) > geom.radians(45):
        #pdb.set_trace()
        return 0.
   
    return geom.circle_overlap(fi.center, fi.r, fj.center, fj.r)

def distance_IoU(fi, fj, P, closed): 
    ''' Intersection over union distance between disks'''
    return (geom.circle_intersection_area(fi.center, fi.r, fj.center, fj.r) /
            geom.circle_union_area(fi.center, fi.r, fj.center, fj.r))


## Deprecated saliency variants
# Variants
def compute_depth_saliency_contour_max_h(P, Pv, f, debug_draw=False, get_area=False):
    a, b = Pv[:,0], Pv[:,-1]
    r = f.r

    # traverse both CSF support segments
    L = Pv[:, :f.anchors[0]+1][:,::-1]
    R = Pv[:, f.anchors[1]:]
    H = []
    segs = []
    p = f.extrema_pos

    if L.shape[1] < 1:
        L = Pv[:,f.anchors[0]].reshape(-1,1)
    if R.shape[1] < 1:
        R = Pv[:,f.anchors[1]].reshape(-1,1)

    if cfg.shortest_support_only:
        m = min(L.shape[1], R.shape[1])
    else:
        m = max(L.shape[1], R.shape[1])

    extremities = []

    # Simple polygon heuristic
    prev = None
    maxl = L.shape[1]-1
    maxr = R.shape[1]-1
    for i in range(m):
        il = min(i, maxl)
        ir = min(i, maxr)
        pl = L[:,il]
        pr = R[:,ir]

        if prev is not None and cfg.only_simple_areas:
            if il < maxl and geom.segment_intersection(pl, pr, prev[0], p)[0]:
                break
            if ir < maxr and geom.segment_intersection(pl, pr, prev[1], p)[0]:
                break
        prev = (pl, pr)

        extremities.append((il, ir))
        #h = geom.point_segment_distance(p, pl, pr)
        h = norm(angle_bisector(p, pl, pr)) # TODO consider using bisector here
        if curvature_sign(pl, p, pr) != f.sign:
            h = -h
        if np.isnan(h):
            h = -10
        H.append(h)
        segs.append((pl, pr))


    #pdb.set_trace()
    #if is_minimum(f):
    #    pdb.set_trace()

    i = np.argmax(H)

    # Construct the contour area for saliency
    start_area = f.anchors[0]-extremities[i][0]
    end_area = f.anchors[1]+extremities[i][1]
    area_poly = Pv[:,start_area:end_area+1]

    if i > 0 and debug_draw: # and f.sign < 0:
        for j in range(m): #i):
            if j%2:
                continue
            pl, pr = segs[j]
            plut.draw_line(pl, pr, np.ones(3)*0.7, alpha=0.5, linewidth=0.25)

    h = H[i]
    if h < 0:
        return 0

    if debug_draw and f.sign < 0:
        pl, pr = segs[i]
        plut.draw_line(pl, pr, 'b', linewidth=0.5, linestyle=':') #[0,0.3,0,7])

        ph = p + angle_bisector(p, pl, pr)
        #ph = geom.project(p, pl, pr)
        plut.draw_line(ph, p, 'r', linewidth=0.5)

        plut.fill_poly(area_poly, 'c', alpha=0.3)
        # plut.stroke_poly(P1o, plut.colors.cyan, closed=False, linewidth=1.5, alpha=alpha)
        # plut.stroke_poly(P2o, 'm', closed=False, linewidth=1.5, alpha=alpha)

    w = np.exp(-r/h)
    if get_area:
        return w, area_poly
    else:
        return w


def compute_depth_saliency_contour_simple(P, Pv, f, debug_draw=False):
    a, b = Pv[:,0], Pv[:,-1]
    r = f.r
    # <- THIS
    P1 = Pv[:,:f.i+1]
    P2 = Pv[:,f.i:]
    P1o, P2o = P1, P2
    if debug_draw and f.sign < 0:
        clr = [0, 0.6, 0.9]
        alpha = 1.
        #plut.stroke_poly(Pv, np.ones(3)*0.5, closed=False, linewidth=2)
        
    #pdb.set_trace()
    s1 = geom.chord_length(P1)
    s2 = geom.chord_length(P2)
    eps = 1e-05
    s = max(min(s1, s2), 1e-05)

    if s > 1e-05: # and f.sign < 0:
        P1 = geom.path_subset_of_length(P1[:,::-1], s)
        P2 = geom.path_subset_of_length(P2, s)
        a, b = P1[:,-1], P2[:,-1]
        p = P2[:,0]
        clr = [1., 0.5, 0]
        alpha = 1
            
        h = geom.point_segment_distance(Pv[:,f.i], a, b)
        w = np.exp(-r/h) #<- Looks like exponential decay works nicer than Gaussian
        #w = np.exp(-r**2 / (2*h**2))
        if debug_draw and f.sign > 0:
            plut.stroke_poly(P1o, plut.colors.cyan, closed=False, linewidth=1.5, alpha=alpha)
            plut.stroke_poly(P2o, 'm', closed=False, linewidth=1.5, alpha=alpha)
            
            plut.draw_line(a, b, np.ones(3)*0.5, linewidth=0.5, linestyle=':') #[0,0.3,0,7])
            ph = geom.project(Pv[:,f.i], a, b)
            #ph = p + geom.normalize(da + db)*h
            plut.draw_line(ph, Pv[:,f.i], 'r', linewidth=0.5) #[0.5,0.6,0.3])
            # if w < 1e-4:
            #plt.text(*Pv[:,f.i], '%e'%(w), fontsize=7)
            # else:
            #     plt.text(*Pv[:,f.i], '$%.2f, %.2f, %.2f$'%(w, h, f.r), fontsize=7)
        return w
    else:
        return 0.
#endf


def compute_depth_saliency_simple(P, Pv, f, debug_draw=False):
    a, b = Pv[:,0], Pv[:,-1]
    r = f.r
    p = f.extrema_pos
    h = norm(angle_bisector(p, a, b))
    if debug_draw and f.sign < 0:
        plut.draw_line(a, b, 'b', linewidth=0.5, linestyle=':') #[0,0.3,0,7])
        ph = p + angle_bisector(p, a, b)
        plut.draw_line(ph, p, 'r', linewidth=0.5)

    w = np.exp(-r/h)
    return w

def isoperimetric_quotient(X):
    P = geom.chord_length(X, closed=True)
    A = abs(geom.polygon_area(X))
    return (A * np.pi * 4) / (P**2)

def object_angle_importance(theta, scale=10.):
    return scale / (np.cos(theta/2)) - scale

def parallel_offset_open(P, o):
    if P.shape[1] < 2:
        return P
    N = geom.normals_2d(P, vertex=True)
    return P + N*o

def draw_CSF(f, clr, offset=0, linewidth=1., draw_axis=False, draw_flags={}):
    #from autograff.geom.shapely_wrap import parallel_offset
    def draw_flag(f):
        if not f in draw_flags:
            return True
        return draw_flags[f]

    if not is_minimum(f) and not is_extremum(f):
        #if f.type == FEATURE_INFLECTION:
        #    plut.draw_marker(f.pos, 'ro')
        return
    if f.r > 1000:
        return
    contact = f.data['contact']
    support = f.data['support']
    pos = f.data['extremum']

    #offset = 0 #signs[i%2]*14.
    #pdb.set_trace()
    draw_axis = draw_axis or ('axis' in draw_flags and draw_flags['axis'])
    if draw_axis:
        if f.data['MA'] is not None:
            vma.draw_skeleton(f.data['MA'], clr=clr)
        plut.draw_line(f.center, pos, clr, linewidth=linewidth*0.5, linestyle=':')

    if draw_flag('osculating'):
        plut.fill_circle(f.center, f.r, clr, alpha=0.15, zorder=-100)
        plut.stroke_circle(f.center, f.r, clr, linewidth=linewidth*0.25, zorder=-20)

    if draw_axis:
        plut.draw_line(f.center, contact[:,0], np.ones(3)*0.5, linewidth=linewidth*0.5, linestyle=':')
        plut.draw_line(f.center, contact[:,-1], np.ones(3)*0.5, linewidth=linewidth*0.5, linestyle=':')

    if draw_flag('osculating_center'):
        plut.fill_circle(f.center, linewidth, 'k')

    # There is no need for an offset if we don't have overlapping support segs.
    if not draw_flag('support'):
        offset = 0

    try:
        left = parallel_offset_open(support[0], -offset)
        right = parallel_offset_open(support[1], offset)
        contact = parallel_offset_open(contact, offset)
    except ValueError:
        return

    if draw_flag('extremum'):
        if is_extremum(f): #f.sign < 0:
            plut.fill_circle(pos, linewidth*2, 'r', zorder=1000)
        else:
            plut.fill_circle(pos, linewidth*2, 'b', zorder=1000)
    if type(left)==list:
        left = np.hstack(left)
    if type(right)==list:
        right = np.hstack(right)
    if type(contact)==list:
        contact = np.hstack(contact)

    if draw_flag('support'):
        plut.stroke_poly(left, clr, closed=False, linewidth=linewidth*1.5, alpha=0.5)
        plut.stroke_poly(right, clr, closed=False, linewidth=linewidth*1.5, alpha=0.5)
        plut.stroke_poly(contact, clr, closed=False, linewidth=linewidth*1., linestyle=':')
    if draw_flag('contact'):
        plut.stroke_poly(contact, 'k', closed=False, linewidth=linewidth*2.)

def draw_CSFs(features,
            clr=None,
            linewidth=1,
            draw_axis=False,
            count=0,
            offset=0,
            draw_flags={},
            only=None):
    if features and type(features[0])==list:
        k = 0
        for fts in features:
            k = draw_CSFs(fts, clr, linewidth, draw_axis, k, offset, draw_flags)
        return k
    
    k = count
    o = [offset, -offset]
    for i, f in enumerate(features):
        if is_endpoint(f):
            continue
        if only is not None and not i in only:
            continue
        c = clr
        if c is None:
            c = plut.default_color(k)
        draw_CSF(f, c, linewidth=linewidth, draw_axis=draw_axis, offset=o[k%2], draw_flags=draw_flags)
        k += 1
    return k

def arc_points(a, b, theta, subd=100):
    ''' Get points of an arc between a and b, 
        with internal angle theta'''
  
    a = np.array(a)
    b = np.array(b)
    mp = a + (b-a)*0.5
    
    if abs(theta) < 1e-9:
        theta = 1e-9
    
    d = b - a #b-a
    l = np.linalg.norm(d)
    r = l / (-np.sin(theta/2)*2)
    
    h = (1-np.cos(theta/2))*r
    h2 = r-h
    p = np.dot([[0,-1],[1, 0]], d)
    p = p / np.linalg.norm(p)
    
    cenp = mp-p*h2
    theta_start = np.arctan2(p[1], p[0])
    A = np.linspace(theta_start+theta/2, theta_start-theta/2, subd)
    arc = np.tile(cenp.reshape(-1,1), (1,subd)) + np.vstack([np.cos(A), np.sin(A)]) * r
    return arc

##########################################
# Internal utilities for debugging
def debug_skeleton(MA):
    plt.cla()
    plt.figure(figsize=(8,8))
    plut.stroke_shape(MA.graph['shape'], 'k')
    vma.draw_skeleton(MA)
    plut.show(axis_limits=geom.bounding_box(MA.graph['shape'], 100))

def debug_forks(MA, forks):
    plt.cla()
    plt.figure(figsize=(8,8))
    plut.stroke_shape(MA.graph['shape'], 'k')
    vma.draw_skeleton(MA)
    disks = MA.graph['disks']
    for n in MA.nodes():
        if MA.degree(n) > 2:
            plut.stroke_circle(disks[n].center, disks[n].r, 'r')
    plut.show(axis_limits=geom.bounding_box(MA.graph['shape'], 100))

def debug_draw_part(P, Pv, MA, E, I):
    ''' debug draw part of the local MA'''
    if E:
        vma.draw_skeleton(MA)
        disks = MA.graph['disks']
        for e in E:
            #plut.fill_circle(disks[e].center, 2, 'c', alpha=1.)
            plut.stroke_circle(disks[e].center, disks[e].r, 'c', alpha=1.)

    if I:
        plt.plot(P[0,I], P[1,I], 'ro', markersize=5)
    plut.stroke_poly(P, [0.4, 0.4, 0.4, 1.], linewidth=0.5, linestyle=':')
    plut.stroke_poly(Pv, 'k', linewidth=2, closed=False)

def debug_features(P, features):
    plt.cla()
    plt.figure(figsize=(6,6))
    plt.plot(P[0,:], P[1,:], 'k')
    for f in features:
        if is_extremum(f):
            plut.stroke_circle(f.center, f.r, 'r' if f.sign < 0 else 'b')
            plut.fill_circle(P[:,f.anchors[0]], 0.25, 'r')
            plut.fill_circle(P[:,f.anchors[1]], 0.25, 'g')
        else:
            plut.fill_circle(f.center, 0.5, 'r') # if f.sign < 0 else 'b')
    plt.axis('equal')
    plt.show()


def debug_feature(P, Pv, f):
    plt.cla()
    plt.figure(figsize=(6,6))
    plut.stroke_poly(P, 'k', closed=False, alpha=0.4)
    plut.stroke_poly(Pv, 'r', closed=False, linewidth=2.)
    plut.stroke_circle(f.center, f.r, 'c')
    plut.draw_marker(Pv[:,f.i], 'g')
    plut.show()
#endf

def debug_feature_local(P, Pv, f):
    plt.cla()
    plt.figure(figsize=(6,6))
    #plut.stroke_poly(P, 'k', closed=False, alpha=0.4)
    plut.stroke_poly(Pv, 'r', closed=False, linewidth=2.)
    plut.stroke_circle(f.center, f.r, 'c')
    plut.draw_marker(Pv[:,f.i], 'go')

    #plut.stroke_poly(Pv, 'r', closed=False, linewidth=2.)
    plut.show()
#endf

def debug_stroke_poly(P, Pv):
    plt.cla()
    plt.figure(figsize=(6,6))
    plut.stroke_poly(P, 'k', closed=False, alpha=0.4)
    plut.stroke_poly(Pv, 'r', closed=False, linewidth=2.)
    plut.show()
#endf

def debug_point(P, p):
    plt.cla()
    plt.figure(figsize=(6,6))
    plut.stroke_poly(P, 'k', closed=False, alpha=0.4)
    plut.fill_circle(p, 10, 'r')
    #plut.stroke_poly(Pv, 'r', closed=False, linewidth=2.)
    plut.show()
#endf

#%%

''' 
Curvilinear Augmented Symmetry Axis, 
assumes the parent directory of autograff is in the Python path
'''

from __future__ import division
from importlib import reload
import time, math, copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import pdb
brk = pdb.set_trace

import autograff.geom as geom
import autograff.utils as utils
import autograff.plut as plut
from autograff.geom.tangent_cover import tangent_cover

from autograff.graph import (graph_branches,
                             remove_node_safe,
                             substroke_graph, 
                             contiguous_strokes, 
                             branch_contour, 
                             peripheral_branches,
                             incident_branches)
from autograff.numeric import gaussian_smooth


from autograff.algorithms import UnionFind
from . import voronoi_skeleton as vma
reload(vma)
from . import path_sym as sym
reload(sym)

from itertools import combinations
from collections import defaultdict, namedtuple
import autograff.graph as graph
reload(graph)

from autograff.utils import perf_timer

pair = lambda a, b: frozenset([a,b])

draw_skeleton = vma.draw_skeleton

# NB main areparamters are defined in config.py
# cfg = lambda: None
from . import config as config

# Configuration
cfg = config.cfg 

# cfg.anchor_expansion_tol = 0.0 #0.5
# cfg.corner_radius_thresh = 2 
# cfg.tangent_sleeve_tolerance = 0.5
# cfg.max_radius_height_ratio = 2.
# cfg.vma_thresh = 1
# cfg.smoothing = 0.5
# cfg.feature_saliency_thresh = 5e-3 
# cfg.compute_CSF_axes = True
# cfg.num_sym_passes = 3
cfg.debug_draw = False


cfg.draw_steps = False
cfg.debug_feature_support = True
cfg.casa_has_minima = False

# Local CSF struct. bit redundant. CSFS in path_sym are computed for each path separately. 
# Since we are interested in computing CSFs for compound shapes (with holes), and only considering
# absolute maxima of curvature, the following abstracts the information for a CSF along a flattened version of the shape contours
# Also simplifies the structure in path_sym, which is overly-complex because of all the iterations in its development
CSFLocal = namedtuple('CSFLocal', 'i pos anchors center r sign saliency is_sat support support_inds contact contact_inds MA area')

#%%
class FlatShape:
    ''' Utility to index a compound shape flatten into a single array'''
    def __init__(self, S, axis=1):
        if axis==1:
            lenf = lambda P: P.shape[1]
            self.flat = np.hstack(S)
        
        else:
            lenf = lambda P: len(P)
            self.flat = sum(S, []) 
            
        self.shape_inds = sum([[i for j in range(lenf(P))] for i, P in enumerate(S)], [])

        self.start_inds = [0]
        for P in S:
            self.start_inds += [self.start_inds[-1] + lenf(P)]
        self.shape = S
    
    def flat_index(self, i, j):
        return self.start_inds[i] + j
    
    def shape_index(self, i):
        return vma.get_shape_index(i, self.start_inds, self.shape_inds)
    
    def __getitem__(self, i):
    	return vma.get_shape_index(i, self.start_inds, self.shape_inds)
    
    def sort_wrapped(self, flat_a, flat_b, direction=0):
        ia, a = self.shape_index(flat_a)
        ib, b = self.shape_index(flat_b)
        if ia != ib:
            return flat_a, flat_b
        i = ia
        n = self.shape[i].shape[1]
        if (a-b)%n < (b-a)%n:
            a, b = b, a
        return self.flat_index(i, a), self.flat_index(i, b)
    #endf
    
    def wrapped_midpoint_index(self, flat_a, flat_b, safe=False):
        ia, a = self.shape_index(flat_a)
        ib, b = self.shape_index(flat_b)
        # If the two points are on two different contours 
        # simply return the midpoint as a workaround
        if ia != ib:
            if safe:
                return -1
            p = (self.shape[ia][:,a] + self.shape[ib][:,b])/2
            #plut.fill_circle(p, 5, 'k')
            return p
        i = ia
        n = self.shape[i].shape[1]
        d = ((b - a)%n) // 2
        mid = (a + d)%n
        return self.flat_index(i, mid)
    #endf
    
    def wrapped_midpoint(self, flat_a, flat_b):
        ia, a = self.shape_index(flat_a)
        ib, b = self.shape_index(flat_b)
        # If the two points are on two different contours 
        # simply return the midpoint as a workaround
        if ia != ib: 
            p = (self.shape[ia][:,a] + self.shape[ib][:,b])/2
            #plut.fill_circle(p, 5, 'k')
            return p
        i = ia
        n = self.shape[i].shape[1]
        d = ((b - a)%n) // 2
        mid = (a + d)%n
        p = self.shape[i][:,mid]
        #plut.fill_circle(p, 5, 'g')
        return p
    
    def range_internal(self, flat_a, flat_b, extend=0, extend_a=0, extend_b=0, safe=False):
        if extend:
            extend_a = extend_b = extend
        ia, a = self.shape_index(flat_a)
        ib, b = self.shape_index(flat_b)
        
        if ia != ib:
            if safe:
                raise ValueError
            return [flat_a, flat_b]
        
        i = ia
        n = self.shape[i].shape[1]
        a = (a-extend_a)%n
        d = ((b - a)%n) + 1 + extend_b
        
        return list([self.flat_index(i, (a + j)%n) for j in range(d)])
    
    # def wrapped_region(self, flat_i, extend, safe=False):
    #     flat_a, flat_b = self.sort_wrapped(flat_a, flat_b)
    #     return self.range_internal(flat_a, flat_b, extend, safe)

    def wrapped_range(self, flat_a, flat_b, extend=0, extend_a=0, extend_b=0, safe=False):
        #flat_a, flat_b = self.sort_wrapped(flat_a, flat_b)
        return self.range_internal(flat_a, flat_b, extend, extend_a, extend_b, safe)
    
    
    def wrapped_range_bidirectional(self, flat_a, flat_b, safe=False):
        return self.range_internal(flat_a, flat_b, 0, safe=safe), self.range_internal(flat_b, flat_a, 0, safe=safe)
    
    def wrapped_contour(self, flat_a, flat_b, extend=0, extend_a=0, extend_b=0, safe=False):
        I = self.wrapped_range(flat_a, flat_b, extend, extend_a, extend_b, safe)
        return self.flat[:,I]

    def wrapped_contour_shortest(self, flat_a, flat_b, extend=0, extend_a=0, extend_b=0, safe=False):
        flat_a, flat_b = self.sort_wrapped(flat_a, flat_b)
        I = self.wrapped_range(flat_a, flat_b, extend, extend_a, extend_b, safe)
        return self.flat[:,I]

    @property
    def size(self):
        return self.flat.shape[1]
#endcls

def saliency_values(features, signs=[-1,1]):
    S = [f.saliency for f in features if (f.sign in signs)]
    return np.array(S)

def saliency_values_and_extrema(features, signs=[-1,1]):
    S = [f.saliency for f in features if (f.sign in signs)]
    I = [f.i for f in features if (f.sign in signs)]
    return S, I

def compute_CSFs(MA, shape, ds, size, debug_draw=False):
    ''' Converts CSFs to local representation'''
    # Compute CSFs for each contour segment
    #sym.cfg.vma_thresh = cfg.vma_thresh #<-make sure we have the same threshold
    sym.cfg.r_thresh = size * cfg.max_radius_height_ratio
    sym.cfg.minima_r_thresh = size * cfg.max_radius_height_ratio

    #sym.cfg.saliency_thresh = cfg.feature_saliency_thresh
    #sym.cfg.anchor_thresh = cfg.anchor_expansion_tol*ds
    
    stats = {}
    #brk()
    flags = 0
    if cfg.casa_has_minima:
        #pdb.set_trace()
        flags = sym.COMPUTE_MINIMA
    feature_list = sym.compute_shape_features(shape, closed=True, n_steps=cfg.num_sym_passes, flags=flags, draw_steps=cfg.draw_steps, stats=stats)

    if cfg.anchor_expansion_tol > 0:
        feature_list  = [sym.expand_all_anchors(P, features, True, cfg.anchor_expansion_tol) 
                            for P, features in zip(shape, feature_list)]

    # This is not strictly necessary, unless we want to visualize the CSF symmetry axes
    # Slower because it recomputes MA for each CSF
    for P, features in zip(shape, feature_list):
        #pdb.set_trace()
        sym.compute_CSFs(features, P, closed=True, compute_axis=cfg.compute_CSF_axes)
    
    if 'nested_CSFs' in stats:
        MA.graph['nested_CSFs'] = True

  
    if type(debug_draw)==int:
        # Hack to debug draw saliency for specific extremum in figures
        i = debug_draw
        debug_draw = False
        P, features = shape[0], feature_list[0]
        m = len(features)
        f0, f1, f2 = features[(i-1)%m], features[i], features[(i+1)%m]
        sym.compute_depth_saliency(P, f0, f1, f2, True, True)

    flat = FlatShape(shape)
    
    def flatten_anchors(i, anchors):
        return (flat.flat_index(i, anchors[0]), flat.flat_index(i, anchors[1]))

    local_features = []
    feature_count = [0 for i in range(len(feature_list))]
    
    # comopute local sym axes
    for i, features in enumerate(feature_list):
        m = len(features)
        P = shape[i]
        n = P.shape[1]

        for j in range(m):
            f = features[j]

            r = f.r
            #if cfg.corner_radius_thresh > 0. and r < cfg.corner_radius_thresh*ds:
            #    r = cfg.corner_radius_thresh*ds

            is_sat = True if ('is_sat' in f.data and f.data['is_sat']) else False
            
            local_features.append(CSFLocal(i=flat.flat_index(i, f.i),
                                           pos=P[:,f.i], #f.extrema_pos,
                                           anchors=flatten_anchors(i, f.anchors),
                                           center=f.center,
                                           r=r,
                                           sign=f.sign,
                                           saliency=f.data['saliency'],
                                           is_sat=is_sat,
                                           support=f.data['support'],
                                           support_inds=[[flat.flat_index(i, j) for j in seg] for seg in f.data['support_inds']],
                                           contact=f.data['contact'],
                                           contact_inds=[flat.flat_index(i, j) for j in f.data['contact_inds']],
                                           MA=f.data['MA'],
                                           area=f.data['area']))
            feature_count[i] += 1
            
    return local_features, feature_count

def weighted_distance(MA, p, n):
    disks = MA.graph['disks']
    return geom.distance(p, disks[n].center) - disks[n].r

def collapse_nodes(G, nodes):
    nodes = list(set(nodes))
    if len(nodes)==1:
        return nodes[0]

    glue = set()
    combs = combinations(nodes, 2)
    for a, b in combs:
        try:
            path = nx.shortest_path(G, a, b)
        except nx.NetworkXNoPath:
            print('Cannot collapse disconnected nodes')
            raise ValueError
        glue |= set(path)
    neighbors = set()
    for n in nodes:
        neighbors |= set(G.neighbors(n))
    neighbors -= glue

    for n in nodes:
        G.remove_node(n)
    
    a = nodes[0]
    G.add_node(a)
    for b in neighbors:
        G.add_edge(a, b)
    return a

def merge_forks(MA, thresh=0.20):
    disks = MA.graph['disks']
    vpos = MA.graph['vpos']
    forks = [v for v in MA.nodes() if MA.degree(v) > 2]
    #for j in junctions:
    #    plut.stroke_circle(np.array(vpos[j]), disks[j].r, 'b')
    #return G
    
    combs = combinations(forks, 2)
    visited = set()
    uf = UnionFind()
    # compute groups of junctions the overlap area ratio of which is greater than thresh
    for j1, j2 in combs:
        uf[j1]
        uf[j2]
        if MA.has_edge(j1, j2) and geom.distance(disks[j1].center, disks[j2].center) < thresh: # must_merge(G, j1, j2, thresh): 
            uf.union(j1, j2)
            
    groups = uf.get_sets()
    # merge groups
    for group in groups.values():
        if len(group) < 2:
            continue
        #pdb.set_trace()
        midp = np.mean([vpos[n] for n in group], axis=0)
        radii = [disks[n].r for n in group]
        r = np.max(radii)
        n = collapse_nodes(MA, group)
        disks[n] = disks[n]._replace(center = midp)
        MA.graph['vpos'][n] = disks[n].center
        MA.graph['vpos_chord'][n] = disks[n].center
    #endf
#endf        
        
def cleanup_MA(MA, ds):
    discard = set()
    disks = MA.graph['disks']
    for n in MA.nodes():
        if MA.degree(n)==0 and disks[n].r < ds*2:
            discard.add(n)
            
    for n in discard:
        MA.remove_node(n)
    return MA


def compute_tangents(features, ds):
    hermite_pairs = []
    sleeve_tol = cfg.tangent_sleeve_tolerance*ds

    for f in features:
        L = f.support[0]
        R = f.support[1]

        tl = tangent_cover(L, sleeve_tol)[0]
        tr = tangent_cover(R, sleeve_tol)[0]
        
        if f.sign < 0. and cfg.debug_draw:
            draw_tangent(L[:,0], tl)
            draw_tangent(R[:,0], tr)
        
        hermite_pairs.append([(L[:,0], tl), (R[:,0], tr)]) #Tl[0], Tr[0]])
    return hermite_pairs

def compute_skeleton_and_features(shape, fork_merge_thresh=0.0, size=None, exterior=False, debug_draw=False, prune_steps=0, get_full_MA=False, compute_sym_features=True):
    if size is None:
        box = geom.bounding_box(shape)
        size = max(geom.rect_h(box), geom.rect_w(box)) #geom.bounding_box(shape))

    # approx arc lenth increment
    ds = np.mean(np.concatenate([geom.chord_lengths(P) for P in shape]))
    
    if cfg.smoothing > 0:
        print('Smoothing')
        shape_smoothed = [geom.gaussian_smooth_contour(P, cfg.smoothing) for P in shape]
        if False: #debug_draw:
            plut.stroke_shape(shape_smoothed, 'm')
    else:
        shape_smoothed = shape
    
    with perf_timer('Computing voronoi skeleton') as tt:
        E, MA = vma.voronoi_skeleton(shape_smoothed, thresh=cfg.vma_thresh, closed=True, internal_flag=1)
    
    cleanup_MA(MA, ds)
    
    if compute_sym_features:
        with perf_timer('Computing concavities and convexities')  as tt:
            features, feature_count = compute_CSFs(MA, shape_smoothed, ds, size=size, debug_draw=debug_draw)
        concavities = [f for f in features if f.sign < 0]
        convexities = [f for f in features if f.sign > 0]
        MA.graph['features'] = features
        MA.graph['feature_count'] = feature_count

    if fork_merge_thresh > 0:
        merge_forks(MA, fork_merge_thresh*ds)
    
    # Exterior MA
    _, exterior_MA = vma.voronoi_skeleton(shape, thresh=cfg.vma_thresh, closed=True, internal_flag=2)
    cleanup_MA(exterior_MA, ds) #<- some cases producing spurious isolated nodes near corners
    
    MA.graph['shape'] = shape
    MA.graph['pruned'] = {}
    MA.graph['flat'] = FlatShape(shape)
    MA.graph['ds'] = ds
    MA.graph['flexures'] = {}
    MA.graph['shape_smoothed'] = shape_smoothed
    MA.graph['feature_hermite_pairs'] = compute_tangents(features, ds)

    # quite some redundancy here
    exterior_MA.graph['flat'] = MA.graph['flat']
    exterior_MA.graph['shape'] = MA.graph['shape']
    exterior_MA.graph['ds'] = MA.graph['ds']

    return MA, exterior_MA, features

def contact_region(MA, f, extend=0):
    flat = MA.graph['flat']
    a, b = f.anchors
    return flat.wrapped_range(a, b, extend=extend)


def compute_casa(MA, features, sign=1):
    MA_ext = MA.copy()
    forks = set([n for n in MA.nodes() if MA.degree(n) > 2])

    # Identify all features that are not generated by a MA branch
    disks = MA.graph['disks']
    points = MA.graph['points']
    vpos = MA.graph['vpos']
    ds = MA.graph['ds']
    flat = MA.graph['flat']

    flexures = {}
    
    # all features with given sign
    M = [i for i, f in enumerate(features) if f.sign==sign]
    
    # map from outline points contained in a contact region, to the corresponding feature
    outline_to_feature = {}

    for fi in M:
        f = features[fi]
        contact = contact_region(MA, f, extend=2) #<- Workaround here. we extend
                                                  #   the contact region,
                                                  #   because some rather large
                                                  #   features might produce a
                                                  #   pruned branch in MA, which
                                                  #   results in anchors missing
                                                  #   along the contact region.
                                                  #   Need to check (as an
                                                  #   example see CSFs for Ultra
                                                  #   Regular 'B')
        for j in contact:
            outline_to_feature[j] = fi
        # See "n.svg" in svg/junctions for where the above breaks
        # Pc = np.array([points[i] for i in contact]).T
        # if fi == 7:
        #     pdb.set_trace()
        #     print('test')
    feature_to_node = defaultdict(set)

    # anchors for each MA node. For nodes that have an angle between ribs < 90 degrees,
    # also consider the midpoint (TODO test me)
    nodes = list(MA.nodes())
    node_anchors = []
    for n in nodes:
        anchors = list(disks[n].anchors)
        center = disks[n].center
        pa, pb = [points[p] for p in anchors]
        if False: #geom.angle_between(pa - center, pb - center) < geom.radians(90): # np.dot(pa - center, pb - center) > 0:
            mid = flat.wrapped_midpoint_index(*anchors, safe=True)
            if mid >= 0:
                anchors.append(mid)

        node_anchors.append(anchors)

    # Use this to assign features to nodes
    for n, p in zip(nodes, node_anchors):
        if MA.degree(n) > 2:
            continue
        p = disks[n].anchors
        for j in p:
            if j in outline_to_feature:
                feature_to_node[outline_to_feature[j]].add(n)

    if not len(MA.nodes()):
        return MA

    #brk()
    newnode = np.max([n for n in MA.nodes()])+1
    
    candidates = []
    
    for fi, nodes in feature_to_node.items():
        if not nodes:
            continue
        f = features[fi]
        
        terminal_nodes = [n for n in nodes if MA.degree(n)==1]
        if terminal_nodes:
            nodes = terminal_nodes
            is_flexure = False
            nodes = sorted(nodes, key=lambda n: 
                                geom.distance(disks[n].center, points[f.i]))
            n = nodes[0]
            # Only extend a terminal if disk overlap is less than theshold
            # overlap = geom.circle_overlap(f.center, f.r, disks[n].center, disks[n].r)
            # if overlap > 0.95:
            #    n = None
        else:
            is_flexure = True
            nodes = sorted(nodes, key=lambda n: 
                                -np.dot(geom.normalize(points[f.i]-disks[n].center), 
                                       geom.normalize(points[f.i]-f.center))) #disks[n].center))
            n = nodes[0]
                
        if n is not None:
            candidates.append((n, fi, is_flexure))
    #endfor

    # Ambiguous cases exist 
    exts = [] #candidates
    for ni, fi, i_flexure in candidates:
        if not i_flexure:
            exts.append((ni, fi, i_flexure))
            continue
        valid = True
        for nj, fj, j_flexure in candidates:
            if fi == fj or j_flexure:
                continue
            
            # if geom.distance(disks[ni].center, disks[nj].center) < disks[nj].r:
            #    valid = False
            #    break
            
        # Check for all existing forks, we don't want a flexure to be within a
        # small epsilon of one. An example Hanzi 34233, TODO check for
        # counterexamples? the fatberg effect.
        # for f in forks:
        #     if geom.distance(disks[f].center, disks[ni].center) < MA.graph['ds']: # disks[f].r:
        #         valid = False
        #         break
        #endfor
        if valid:
            exts.append((ni, fi, i_flexure))
    vpos_new = dict()
    for n in MA.nodes():
        vpos_new[n] = vpos[n]

    tip_features = {}
    flexure_convexities = set()

    for n, fi, is_flexure in exts:
        if is_flexure:
            flexures[n] = fi
            flexure_convexities.add(fi)
            #plut.fill_circle(disks[n].center, 11, plut.colors.cyan, zorder=1000)
            
        #csfdisk = vma.Disk(f.center, f.r, f.anchors, 0, False)
        f = features[fi]
        nt = newnode
        newnode += 1
        MA_ext.add_node(nt)
        MA_ext.add_edge(n, nt)
        tip_features[nt] = fi

        prev_pos = disks[n].center
        new_pos = points[f.i]
        d = new_pos - prev_pos
        l = geom.distance(new_pos, prev_pos)
        new_pos = prev_pos + (d/l)*(l-ds*0.5)

        vpos_new[nt] = new_pos #points[f.i]
        disks[nt] = vma.Disk(new_pos, 0, (f.i, f.i), 0, False)

    vpos = np.zeros((newnode, 2))

    for n, pos in vpos_new.items():
        vpos[n] = pos
        MA_ext.graph['centroids'][n] = pos

    MA_ext.graph['vpos'] = vpos
    MA_ext.graph['flexures'] = flexures
    MA_ext.graph['MA_forks'] = forks
    MA_ext.graph['tip_features'] = tip_features
    MA_ext.graph['flexure_convexities'] = flexure_convexities
    
    return MA_ext
    
def is_casa_only_branch(MA, branch):
    """Returns True if the branch is a MA-ext branch only

    
    Args:
        MA (nx.Graph): Medial Axis (Extended)
        branch (list): branch
    """

    if MA.degree(branch[-1]) != 1:
        branch = branch[::-1]
    disks = MA.graph['disks']
    MA_forks = MA.graph['MA_forks']
    return len(branch)==2 and disks[branch[-1]].r == 0. and branch[0] not in MA_forks


#%%
def flatten_features(shape, feature_list):
    flat = FlatShape(shape)
    features_flat = []
    for i, features in enumerate(feature_list):
        for f in features:
            features_flat.append(f._replace(i=flat.flat_index(i, f.i), anchors=(flat.flat_index(i, f.anchors[0]), flat.flat_index(i, f.anchors[1]))))
    return features_flat
#endf

def concavity_axis_segment(c, simpl_epsilon=3, max_dist=20.):
    axis = c.local_axis
    R = np.array([disk.r for disk in axis])/2
    P = np.array([disk.center for disk in axis]).T
    # limit length of axis
    L = geom.cum_chord_lengths(P, closed=False)
    for n, l in enumerate(L):
        if l > max_dist and n >= 2:
            break
    R = R[:n]
    P = P[:,:n]

    # estimate a component count based on polyline simplification
    Psimp = geom.dp_simplify(P, simpl_epsilon, False)
    #plut.stroke_poly(Psimp, 'r', closed=False, linewidth=2.)
    return Psimp[:,:2].T
#endf

def concavity_axis_segments(MA, simpl_epsilon=3, max_dist=20.):
    ''' Returns a straight segment approximation of the local axis to each concavity''' 
    segments = []
    #branches = [[n for n in branch if MA.degree(n) <=2] for branch in branches]
    concavities = MA.graph['concavities']
    for f in concavities:
        axis = f.local_axis
        
        R = np.array([disk.r for disk in axis])/2
        P = np.array([disk.center for disk in axis]).T
        # limit length of axis
        L = geom.cum_chord_lengths(P, closed=False)
        for n, l in enumerate(L):
            if l > max_dist and n >= 2:
                break
        R = R[:n]
        P = P[:,:n]

        # estimate a component count based on polyline simplification
        Psimp = geom.dp_simplify(P, simpl_epsilon, False)
        #plut.stroke_poly(Psimp, 'r', closed=False, linewidth=2.)
        segments.append(Psimp[:,:2].T)
        
    return segments
#endf

def is_point_in_concavity(G, p, eps=0.01):
    concavities = G.graph['concavities']
    points = G.graph['points']
    disks = G.graph['disks']
    
    for i, cdisk in enumerate(concavities):
        pc = points[cdisk.i]
        if np.dot(pc - cdisk.center, p - cdisk.center) > 0: # Make sure the normal corresponds to current vertex
            d = geom.distance(p, cdisk.center)
            if d < (cdisk.r + eps):
                return True
    
    return False
#endf

def get_incident_concavity_indices(G, n, thresh=1.0, concavities=[]): # TODO change to eps!
    if not concavities:
        concavities = G.graph['concavities']
    points = G.graph['points']
    disks = G.graph['disks']
    disk = disks[n]
    inds = []
    for i, cdisk in enumerate(concavities):
        p = points[cdisk.i]
        if np.dot(p - disk.center, p - cdisk.center) < 0: # Make sure the normal corresponds to current vertex
            d = geom.distance(disk.center, cdisk.center)
            if d < (disk.r + cdisk.r) + thresh:
                inds.append(i)
    return inds
#endf

def draw_features(MA, convexities=False, markersize=7, draw_areas=False):
    features = MA.graph['features']
    points = MA.graph['points']
    for f in features:
        r = f.r #max(f.r,  10)
        if f.sign < 0:
            plut.stroke_circle(f.center, r, [1.,0.5,0.], alpha=1, linewidth=0.25)
            #plut.draw_marker(points[f.i], 'ro', markersize=2)
            plut.fill_circle(points[f.i], markersize, 'r')
            if draw_areas:
                plut.fill_poly(f.area, 'r', alpha=0.5)
        if convexities:
            if f.sign > 0:
                plut.stroke_circle(f.center, r, [0,0.32,1.], alpha=1, linewidth=0.25)
                plut.fill_circle(points[f.i], markersize, 'b')
                #plut.draw_marker(points[f.i], 'bo')
#endf

def draw_branch(MA, branch, clr=np.ones(3)*0.5, alpha=1, linewidth=0.25):
    vpos = MA.graph['vpos']
    disks = MA.graph['disks']
    P = [p for p in branch_contour(branch, vpos).T]
    P = np.array(P).T
    plut.stroke_poly(P, clr, closed=False, alpha=alpha, linewidth=linewidth)
#endf

def draw_spokes(MA, clr='r', alpha=1, linewidth=0.25):
    disks = MA.graph['disks']
    features = MA.graph['features']
    P = MA.graph['flat'].flat
    
    feature_anchors = set()
    for f in features:
        if f.sign >= 0:
            continue
        feature_anchors |= set(list(f.anchors))
    spokes = set()
    for n in MA.nodes:
        anchors = disks[n].anchors
        for i in anchors:
            if i in feature_anchors:
                spokes.add((n, i))
    for n, i in spokes:
        plut.draw_line(disks[n].center, P[:,i], clr, linewidth=linewidth)
#endf

def draw_skeleton(MA, clr=np.ones(3)*0.5, alpha=1, linewidth=0.25, draw_spokes=False):
    vpos = MA.graph['vpos']
    disks = MA.graph['disks']
    branches = graph_branches(MA)
    for branch in branches:
        draw_branch(MA, branch, clr, alpha, linewidth)
        
    if draw_spokes:
        points = MA.graph['points']
        for n in MA.nodes():
            a, b = disks[n].anchors
            plut.draw_line(vpos[n], points[a], np.ones(3)*0.5, alpha=0.2)
            plut.draw_line(vpos[n], points[b], 'k', alpha=0.2)
        #enfor
    #endif
#endf

def draw_shape(shape):
    plut.fill_stroke_shape(shape, np.ones(3)*0.95, np.ones(3)*0.25, linewidth=0.5) # alpha=0.5) #, linestyle=':')
    
def draw_shape_and_skeleton(MA, draw_spokes=False, draw_forks=False, convexities=False, draw_areas=False, markersize=7, skel_alpha=1, features=True):
    plut.fill_stroke_shape(MA.graph['shape'], np.ones(3)*0.95, np.ones(3)*0.25, linewidth=0.5, zorder=-2000) # alpha=0.5) #, linestyle=':')
    vpos = MA.graph['vpos']
    disks = MA.graph['disks']
    draw_skeleton(MA, alpha=skel_alpha)
    #branches = graph_branches(MA)
    if features:
        draw_features(MA, convexities=convexities, markersize=markersize, draw_areas=draw_areas)
    
    #SMA = [branch_contour(branch, vpos) for branch in branches]
    #plut.stroke_shape(SMA, 'k', closed=False, linestyle=':', alpha=0.4)
    if draw_spokes:
        points = MA.graph['points']
        for n in MA.nodes():
            a, b = disks[n].anchors
            plut.draw_line(vpos[n], points[a], 'k', alpha=0.2)
            plut.draw_line(vpos[n], points[b], 'k', alpha=0.2)
        #endfor
    #endif

    if draw_forks:
        forks = [n for n in MA.nodes() if MA.degree(n) > 2]
        for f in forks:
            plut.stroke_circle(disks[f].center, disks[f].r, np.ones(3)*0.8, linewidth=0.25)
    #endif
#endf

def debug_features(MA, branch, F=[], min_radius=2, draw_spokes=False):
    features = MA.graph['features']
    disks = MA.graph['disks']
    points = MA.graph['points']
    plt.clf()
    plt.figure(figsize=(5,5))
    plut.stroke_shape(MA.graph['shape'], 'k')
    draw_skeleton(MA)
    
    vpos = MA.graph['vpos']
    
    if len(branch):
        if type(branch[0]) != list:
            branches = [branch]
        else:
            branches = branch
    else:
        branches = []
        
    for branch in branches:
        P = branch_contour(branch, vpos)
        plut.stroke_poly(P, 'r', closed=False, linewidth=2.)
        if draw_spokes:
            for n in branch:
                anch = disks[n].anchors
                for p in anch:
                    plut.draw_line(disks[n].center, points[p], 'k', alpha=0.5)
    
    for i, fi in enumerate(F):
        if fi is None:
            continue
        
        f = features[fi]
        s = f.sign
        r = f.r
        if s < 0:
            clr = 'r'
        elif s==0:
            clr = 'k'
            r = 20
        else:
            clr = 'b'
        plut.fill_circle(f.center, max(f.r, min_radius), clr, alpha=0.5)
        plt.text(*f.center+[20,20], '%d'%(fi), color='k')
    plut.show(axis_limits=geom.bounding_box(MA.graph['shape'], 100))
#endf
    
def debug_forks(MA, forks=[]):
    disks = MA.graph['disks']
    if not forks:
        forks = [n for n in MA.nodes() if MA.degree(n) > 2]
    
    plt.clf()
    plt.figure(figsize=(3,3))
    plut.stroke_shape(MA.graph['shape'], 'k')
    draw_skeleton(MA)
    for fork in forks:
        plut.fill_circle(disks[fork].center, disks[fork].r, 'c', alpha=0.3)
        plt.text(*disks[fork].center+[20,20], '%d'%(fork), color='k')   
    plut.show(axis_limits=geom.bounding_box(MA.graph['shape'], 100))
#endf

    
def debug_skeleton(MA, forks=[]):
    plt.clf()
    plt.figure(figsize=(3,3))
    plut.stroke_shape(MA.graph['shape'], 'k')
    draw_skeleton(MA)
    plut.show(axis_limits=geom.bounding_box(MA.graph['shape'], 100))
#endf

def draw_CSF(f, clr, offset=0, linewidth=1., draw_axis=False):

    from autograff.geom.shapely_wrap import parallel_offset

    contact = f.contact
    offset = 0 #signs[i%2]*14.
    
    if draw_axis:
        vma.draw_skeleton(f.MA, clr=clr)
        plut.draw_line(f.center, f.pos, clr, linewidth=linewidth*0.5, linestyle=':')
    
    plut.fill_circle(f.center, f.r, clr, alpha=0.15)
    plut.stroke_circle(f.center, f.r, clr, linewidth=linewidth*0.25)
    if draw_axis:
        plut.draw_line(f.center, contact[:,0], np.ones(3)*0.5, linewidth=linewidth*0.5, linestyle=':')
        plut.draw_line(f.center, contact[:,-1], np.ones(3)*0.5, linewidth=linewidth*0.5, linestyle=':')
    plut.fill_circle(f.center, linewidth*5, 'k')
    
    left = parallel_offset(f.support[0], offset)
    #pdb.set_trace()
    right = parallel_offset(f.support[1], offset)
    contact = parallel_offset(f.contact, offset)
    plut.fill_circle(f.pos, linewidth*10, 'r', zorder=1000)
    if type(left)==list:
        left = np.hstack(left)
    if type(right)==list:
        right = np.hstack(right)
    if type(contact)==list:
        contact = np.hstack(contact)
        
    plut.stroke_poly(left, clr, closed=False, linewidth=linewidth*1.5, alpha=0.5)
    plut.stroke_poly(right, clr, closed=False, linewidth=linewidth*1.5, alpha=0.5)
    plut.stroke_poly(contact, clr, closed=False, linewidth=linewidth*1., linestyle=':')
    plut.stroke_poly(contact, 'k', closed=False, linewidth=linewidth*2.)
    


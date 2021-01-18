"""Basic implementation of Voronoi Medial Axis approximation
   Ogniewicz and Ilg (92) Voronoi skeletons: Theory and applications
"""
from __future__ import division
import numpy as np
from collections import namedtuple, defaultdict
import matplotlib.pyplot as plt
import networkx as nx

import autograff.plut as plut
import autograff.geom as geom
import autograff.utils as utils
from autograff.graph import graph_branches, branch_contour, peripheral_branches, peripheral_branches_bidirectional

from scipy.spatial import Voronoi, Delaunay 
import copy
import pdb
import time

class perf_timer:
    def __init__(self, name=''):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.t)*1000
        if False: #self.name:
            print('%s: elapsed time %.3f'%(self.name, self.elapsed))

            
# Configuration
cfg = lambda: None
cfg.verbose = True

Disk = namedtuple('Disk', 'center r anchors residual inf') 
Edge = namedtuple('Edge', 'center r verts anchors residual inf')

def debug_print(s):
    if cfg.verbose:
        print(s)

def branch_length(vma, branch):
    """Branch length, not considering infinity Voronoi vertices
    
    Args:
        vma (tuple): Voronoi MA temp dat    a
        branch (list): MA nodes
    
    Returns:
        [float]: branch length
    """    
    verts, edges, anchors, centers, radii, res_, inf, einf = vma
    branch = [v for v in branch if not v in inf] # inf[v]]
    l = 1e-5 # min lenght to avoid divs by zero later
    for a, b in zip(branch, branch[1:]):
        l += np.linalg.norm(verts[a] - verts[b])
    return l

###############################
## Residual functions

def boundary_potential(P):
    D = np.diff(P, axis=1)
    s = np.sqrt(D[0,:]**2 + D[1,:]**2)
    W = np.cumsum(np.concatenate([[0.], s]))
    return W

def boundary_potential(P, part_ids=None, closed=False):
    ''' Boundary potential function (contour length for each point in (flattened) shape'''
    if part_ids is not None:
        W = []
        S = unflatten_shape(P, part_ids)
        for P in S:
            W.append(geom.cum_chord_lengths(P, closed=False))
        return np.hstack(W), [geom.chord_length(P, closed=closed) for P in S] # [w[-1] for w in W]
    else:
        return geom.cum_chord_lenghts(P, closed=False), geom.chord_length(P, closed=closed)
        
    D = np.diff(P, axis=1)
    s = np.sqrt(D[0,:]**2 + D[1,:]**2)
    W = np.cumsum(np.concatenate([[0.], s]))
    return W
    

def potential_residual(P, a, b, r, Win, center, part_ids=None, closed=False):
    ''' Potential residual (arc length based)'''
    # check for disjoint boundary segments
    if part_ids is not None:
        if part_ids[a] != part_ids[b]:
            return 1e25
        W, Lbs = Win
        Lb = Lbs[part_ids[a]]
    else:
        W, Lb = Win

    #return abs(W[a] - W[b]) # open contour
    if closed:
        return min(abs(W[a] - W[b]), 
                   Lb - abs(W[a] - W[b]))
    return abs(W[a] - W[b]) # open contour                   


def boundary_length(P, a, b, Win, shape_inds=None, closed=False):
    ''' Returns boundary length between points a and b'''
    return potential_residual(P, a, b, 0., Win, None, shape_inds, closed)
        
def circularity_residual(P, a, b, r, W, center, part_ids=None, closed=False):
    w = potential_residual(P, a, b, r, W, center, part_ids, closed)
    c = np.linalg.norm(P[:,a] - P[:,b])
    theta = -2. * np.arcsin(c / (2*r))  
    return w - abs(theta)*r

def bicircularity_residual(P, a, b, r, W, center, part_ids=None, closed=False):
    w = potential_residual(P, a, b, r, W, center, part_ids, closed)
    c = np.linalg.norm(P[:,a] - P[:,b])
    theta = -2. * np.arcsin(c / (2*r))
    return (2/np.pi)*((np.pi/2)*w - abs(theta)*r)

def chord_residual(P, a, b, r, W, center, part_ids=None, closed=False):
    # corresponds to chordal axis
    w = potential_residual(P, a, b, r, W, center, part_ids, closed)
    return w - np.linalg.norm(P[:,a] - P[:,b])

def lambda_residual(P, a, b, r, W, center, part_ids=None, closed=False):
    n1 = center - P[:,a]
    n2 = center - P[:,b]
    t1 = geom.perp(n1)
    t2 = geom.perp(n2)
    theta = geom.angle_between(n1, n2)
    return 10. / (np.cos(theta/2)) - 10. ## 1 - np.cos(theta/2) #1. / (np.cos(theta/2)) #+1e-10)

def scale_shape_and_skeleton(S, MA, scale):
    S = [P*scale for P in S]
    disks = MA.graph['disks']
    for n, disk in disks.items():
        disks[n] = disk._replace(center=disk.center*scale,
                                 r=disk.r*scale)
    edges = MA.graph['data']
    for e, edge in edges.items():
        if type(e) == tuple:
            edges[e] = edge._replace(center=edge.center*scale,
                                    r=edge.r*scale)
    
    MA.graph['vpos'] = np.array(MA.graph['vpos'])*scale
    return S, MA


def voronoi_skeleton(S, thresh, residual=chord_residual, closed=False, get_branches=False, get_voronoi=False, farthest=False, internal_flag=0, debug_draw_residuals=False, terminal_branches=None ):
    ''' Extract Vornoi Medial Axis for a set of open or closed contours
        internal_flag = 0 (intrnal/external MA), 1 (internal), 2 (external)
        Outputs:
            E: list of extremity vertices (indices in Disk array)
            MA: Networkx graph for skeleton this contains:
                MA.graph['disks']: dict of Disk namedtuples, containing info for each disk (see above) and indexed by vertex index
                MA.graph['data']: dict of Edge namedtuples, contains info for each edge (see above), indexed by pairs (tuple) of vertices
                MA.graph['vma']: raw Voronoi MA data as computed by preprocess_VMA (for debug purposes, should not be necessary to use)
                MA.graph['vpos']: list of positions for each vertex
            -----
            Optional:
            if get_voronoi is True:
                (vor, delu): Voronoi diagram and Delaunay triangulation
            if get_branches is True:
                branches: list of branches -> each branch is a list of vertices (unordered)  
    '''

    if type(S) != list:
        S = [S]
    
    ds = np.mean(np.concatenate([geom.chord_lengths(P) for P in S])) 
    #thresh = ds*thresh
    with perf_timer('Computing VMA'):
        vma, vor_delu = VMA(S, thresh, residual=residual, closed=closed, farthest=farthest, debug_draw_residuals=debug_draw_residuals)
    vor, delu = vor_delu
    # Find extremities
    with perf_timer('Make graph'):
        MA = make_graph(vma, vor_delu, farthest, closed)

    # Skeleton pyramid is not implemented
    #if False: #pyramid_thresh > 0: 
    #    skeleton_pyramid(MA, farthest)
    #    MA = skeleton_subgraph(MA, pyramid_thresh)
    
    verts, edges, anchors, centers, radii, res_, inf, einf = vma
    
    def query(v):
        flag = geom.point_in_shape(verts[v], S)
        #plut.fill_circle(verts[v], 1, 'b')
        if internal_flag > 1:
            return not flag
        elif internal_flag:
            return flag
        return True

    to_remove = []    

    # if internal_flag:
    #     to_remove = set()
    #     comps = list(nx.connected_component_subgraphs(MA))
    #     print(len(comps))
    #     for comp in comps:
            
    #         branches = graph_branches(comp)
    #         for branch in branches:
    #             v = branch[len(branch)//2]
    #             # Hacky avoid degree 1 because terminal nodes might be "overshooting" and cause
    #             # branches to be pruned 
    #             if not query(v): 
    #                 to_remove = to_remove.union(branch) #list(comp.nodes()))
    #                 break
    if internal_flag:
        with perf_timer('checking interior branches'):
            to_remove = set()

            branches = graph_branches(MA)
            for branch in branches:
                v = branch[len(branch)//2]
                # Hacky avoid degree 1 because terminal nodes might be "overshooting" and cause
                # branches to be pruned
                if not query(v):
                    for i in [0, -1]:
                        if not branch:
                            continue
                        if query(branch[i]):
                            branch.pop(i)
                    to_remove = to_remove.union(branch) #list(comp.nodes()))
                    #break
            for v in to_remove:
                MA.graph['disks'].pop(v)

            MA.remove_nodes_from(list(to_remove))

    with perf_timer('peripheral branch pruning'):
        branches = peripheral_branches(MA, outwards=True)
        disks = MA.graph['disks']
        for branch in branches:
            ca, ra = disks[branch[0]].center, disks[branch[0]].r
            cb, rb = disks[branch[-1]].center, disks[branch[-1]].r
            IoU = geom.circle_intersection_area(ca, ra, cb, rb) / geom.circle_union_area(ca, ra, cb, rb)

            if IoU > 0.8: #branch_length(vma, branch) < ds*2:
                MA.remove_nodes_from(branch[1:])#list(to_remove))

    # Peripheral branches, not considering an infinity vertex as a terminal
    with perf_timer('storing peripheral branches'):
        branches = peripheral_branches_bidirectional(MA, not_terminal=inf)
        E = []
        branch_lengths = []
        for branch in branches:
            if MA.degree[branch[0]]==1 and not branch[0] in inf: #inf[branch[0]]:
                E.append(branch[0])
                branch_lengths.append(branch_length(vma, branch))
                if terminal_branches is not None:
                    terminal_branches.append(branch)
            if MA.degree[branch[-1]]==1 and not branch[-1] in inf: #inf[branch[-1]]:
                E.append(branch[-1])
                branch_lengths.append(branch_length(vma, branch))
                if terminal_branches is not None:
                    terminal_branches.append(branch[::-1])


    MA.graph['data']['branch_lengths'] = {e:l for e, l in zip(E, branch_lengths)}

    delu = MA.graph['delu']
    points = MA.graph['points']
    centroids = {}
    for n in MA.nodes():
        if n < len(delu.simplices):
            tri = delu.simplices[n]
            centroids[n] = np.mean([points[i] for i in tri], axis=0)
    MA.graph['centroids'] = centroids

    res = [E, MA]

    if get_voronoi:
        res = res + [vor, delu]
    if get_branches:
        res = res + [branches]
    
    return res
    
def branch_sls_contour(MA, nodes, smooth=0.):
    # Dirty attempt at SLS approximation
    disks = MA.graph['disks']
    P = MA.graph['points']
    Psls = []
    for n in nodes:
        a, b = disks[n].anchors
        Psls.append((P[a] + P[b])/2)
    Psls = np.array(Psls).T
    if False: #smooth > 0. and Psls.shape[1] > 1:
        Psls = geom.uniform_sample_n(Psls, 7)
        Psls = geom.bspline(100, P)
        #P = geom.gaussian_smooth_contour(P, smooth, closed=False)
    return Psls

def draw_branches(MA, branches):
    vpos = MA.graph['vpos']
    for i, branch in enumerate(branches):
        Pb = np.array([vpos[v] for v in branch]).T
        plut.stroke_poly(Pb, plut.default_color(i), closed=False)
        
def draw_skeleton(MA, clr='r', ratio=1., linewidth=0.75, alpha=0.5, draw_disks=False, disk_alpha=0.4, label='', aux_MA=None, chord=False):
    ''' Draw Voronoi Medial Axis, and optionally the disks for each node'''
    if MA==None:
        return
    # Check if we provided another medial axis graph
    # in case we will chose that for the vertices and edges
    # this might be useful in case we want to draw a subgraph of the medial axis
    vertex_MA = MA
    if aux_MA is not None:
        vertex_MA = aux_MA

    if chord:
        pos = MA.graph['vpos_chord']
    else:
        pos = MA.graph['vpos']
        
    if 'data' in MA.graph:
        data = MA.graph['data']
    else:
        data = []

    for a, b in vertex_MA.edges():
        linestyle='-'
        if data and (a,b) in data and data[(a,b)].inf:
            linestyle=':'
        plut.draw_line(pos[a]*ratio, pos[b]*ratio, clr, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label=label)
        label=''
    # for v in MA.nodes():
    #     if MA.degree(v) == 1:
    #         plut.fill_circle(pos[v], 1, 'r')
    if draw_disks:
        disks = MA.graph['disks']
        for v in vertex_MA.nodes():
          plut.stroke_circle(disks[v].center*ratio, disks[v].r*ratio, clr, alpha=disk_alpha)

def draw_disks(MA, clr='r', alpha=0.5, ratio=1., degree_filter=lambda d: True, label='', fill=False):
    ''' Draw Voronoi Medial Axis disks'''
    if MA==None:
        return
    disks = MA.graph['disks']
    
    for v in MA.nodes():
        # hack for label

        #plut.draw_line(disks[v].center*ratio, disks[v].center*ratio+[ratio,0], 'w', alpha=0., label=label)
        #label=''
        if degree_filter(MA.degree(v)):
            if fill:
                plut.fill_circle(disks[v].center*ratio, disks[v].r*ratio, clr, alpha=alpha*0.5)
            plut.stroke_circle(disks[v].center*ratio, disks[v].r*ratio, clr, alpha=alpha)
        

###############################
## Shape helpers

def get_pisa_point(P, a, b, center, r, neg=False):
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
    

def flatten_shape(S):
    ''' flatten a shape to a point sequence,
        returns points (2xn) and shape indices'''
    shape_inds = sum([[i for j in range(P.shape[1])] for i,P in enumerate(S)], [])
    return np.hstack(S), shape_inds
    
def unflatten_shape(P, shape_inds):
    ''' Reconstruct shape from point sequence and shape indices
    '''
    # probably more elegant way to do this?
    I = [0] + list(np.where(np.diff(shape_inds) > 0)[0]+1) + [len(shape_inds)]
    #print I
    return [P[:,a:b] for a, b in zip(I, I[1:])]
 
def get_shape_indices(S, axis=1):
    shape_inds = sum([[i for j in range(P.shape[axis])] for i,P in enumerate(S)], [])
    
    start_inds = [0]
    for P in S:
        start_inds += [start_inds[-1] + P.shape[axis]]
   
    return shape_inds, start_inds

def get_shape_index(i, start_inds, shape_inds):
    si = shape_inds[i]
    return si, i - start_inds[si]

def get_flat_index(si, start_inds):
    return start_inds[si[0]] + si[1]
 
def get_flat_anchors(si, anchors, start_inds):
    return (get_flat_index(si, anchors[0], start_inds),
            get_flat_index(si, anchors[1], start_inds))

def get_shape_anchors(anchors, start_inds, shape_inds):
    return (get_shape_index(anchors[0], start_inds, shape_inds),
            get_shape_index(anchors[1], start_inds, shape_inds))

def at_infinity(v):
    return v[0] < 0 or v[1] < 0


def preprocess_VMA(vor_delu, thresh, residual=chord_residual, shape_inds=None, closed=False, farthest=False, debug_draw_residuals=False):
    ''' Compute Voronoi Medial Axis, internal implementation, assumes Voronoi and Delaunay triangulation are given '''
    vor, delu = vor_delu
    
    # If not shape IDs, assume a single contour
    if shape_inds is None:
        shape_inds = np.zeros(len(vor.points)).astype(int)

    rp, rv = [], []

    V = list(vor.vertices)
    center = vor.points.mean(axis=0)
    
    farthest_mul = 1.
    if farthest:
        farthest_mul = -1
        
    infinite = set()
    is_edge_inf = []

    if farthest:
        print('Thresh farthest: ' + str(thresh))
        
    # Mark infinite edges and vertices and set positions (for graphing)
    for p, v in zip(vor.ridge_points, vor.ridge_vertices):
        v = np.asarray(v)

        rp.append(p)

        if not at_infinity(v):
            rv.append(v)
            is_edge_inf.append(0)
            #infinite[v[0]] = 0
            #infinite[v[1]] = 0
        else: #not farthest: 
            i_finite = np.where(v >= 0)[0][0]
            # Mark infinity verts
            #infinite[v[i_finite]] = 0
            infinite.add(len(V)) #infinite[len(V)] = 1
            # and edge
            is_edge_inf.append(1+i_finite)

            i = v[i_finite] 

            t = vor.points[p[1]] - vor.points[p[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[p].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            
            far_point = vor.vertices[i] + direction * 3500 * farthest_mul #ptp_bound.max()
            
            rv.append([i, len(V)])
            V.append(far_point)

    # precompute boundary data
    P = np.array(vor.points).T
    W = boundary_potential(P, shape_inds, closed)
    
    residuals = []
    edges = []
    anchors = []
    radii = []
    centers = []
    edge_inf = []

    # if farthest:
    #     def func(v):            
    #         #print v
    #         return v < thresh
    # else:
    
    func = lambda v: v > thresh
    
    # Do some statistics
    residuals = []
    
    # Extract MA edges and residuals
    for p, v, inf in zip(rp, rv, is_edge_inf):
        #if p[0] >= P.shape[1] or p[1] >= P.shape[1]
        a, b = P[:,p[0]], P[:,p[1]]

        # If infinity set center to non infinite vertex
        if inf:
            center = V[v[inf-1]]
        else:
            # otherwise to edge center    
            #pdb.set_trace()
            center = (V[v[0]] + V[v[1]]) / 2 #(a + b)/2 # 
            #center = (a+b)/2

        # center and radius are temporary here, we compute the correct 
        # radius and center (per vertex) based on the corresponding delaunay triangle later
        r = np.linalg.norm(a - center)
        #r = geom.circumcircle_radius(*tri)
        
        res = residual(P, p[0], p[1], r, W, center, shape_inds, closed)

        residuals.append(res)
        if farthest:
            pass #res = 1.
        
        ok = func(res)
        if farthest and inf: # Skip infinity edges in Farthest
            ok = False
        if ok: 
            edges.append(v)
            anchors.append(p)
            centers.append(center)
            radii.append(r)
            residuals.append(res)
            edge_inf.append(inf)

            #plut.draw_line(V[v[0]], V[v[1]], 'r')
        else:
            pass

    # if farthest:
    #     print("FVD Stats:")
    #     print('Mean: ' + str(np.mean(residuals)))
    #     print('Min: ' + str(np.min(residuals)))
    #     print('Max: ' + str(np.max(residuals)))
    return np.array(V), edges, anchors, centers, radii, residuals, infinite, edge_inf

def voronoi_diagram(P, farthest=False):
    return Voronoi(P.T, furthest_site=farthest, incremental=False) #, qhull_options='Qbb Qc QJ')
    
def delaunay(P, farthest=False):
    return Delaunay(P.T, furthest_site=farthest, incremental=False) #, qhull_options='Qbb Qc QJ')
    
def VMA(S, thresh, residual=chord_residual, closed=True, farthest=False, debug_draw_residuals=False):
    ''' Computes VMA for a shape (list of contours)
        outputs the VMA edges and info as well as the whole Voronoi diagram '''
    shape_inds = sum([[i for j in range(P.shape[1])] for i,P in enumerate(S)], [])
    P = np.hstack(S)
    with perf_timer('Computing Delaunay') as tt:
        delu = Delaunay(P.T, furthest_site=farthest, incremental=False, qhull_options='Qbb Qc QJ')
    with perf_timer('Computing Voronoi Diagram') as tt:
        vor = Voronoi(P.T, furthest_site=farthest, incremental=False, qhull_options='Qbb Qc QJ')
    with perf_timer('Preprocessing VMA') as tt:
        vma = preprocess_VMA((vor, delu), thresh, residual=residual, shape_inds=shape_inds, closed=closed, farthest=farthest, debug_draw_residuals=debug_draw_residuals)
    return vma, (vor, delu)
 
# STILL A MESS.
def make_graph(vma, vor_delu, farthest, closed):
    ''' Build MA graph from Voronoi'''
    vor, delu = vor_delu
    G = nx.Graph()
    G.graph['vma'] = vma  
    G.graph['data'] = {}
    G.graph['disks'] = {}
    G.graph['vpos'] = vma[0]
    G.graph['points'] = np.array(vor.points)
    G.graph['vor'] = vor
    G.graph['delu'] = delu

    def set_edge_data(a,b,data):
        G.graph['data'][(a,b)] = data
        G.graph['data'][(b,a)] = data
    
    verts, edges, anchors, centers, radii, residuals, infinite, edge_inf = vma
    vpos = np.array(verts)

    # OLD
    for i in range(verts.shape[0]):
        G.add_node(i)
    
    i = 0
    for i, e in enumerate(edges):
        a, b = e
        G.add_edge(a,b)
        set_edge_data(a,b, Edge(center=centers[i],
                           r=radii[i],
                           verts=(verts[a], verts[b]), 
                          anchors=anchors[i],
                          residual=residuals[i],
                          inf=edge_inf[i]))
    
    # remove nodes with degree zero
    to_remove = []
    for n in G.nodes():
       if G.degree[n] < 1:
           to_remove.append(n)

    G.remove_nodes_from(to_remove)
    
    # add vertex data from edges, not sure what is the correct way to do this?
    data = G.graph['data']
    tris = delu.points[delu.simplices]
    points = G.graph['points']

    # traverse each branch and set vertices
    branches = graph_branches(G)
    for branch in branches:
        # possibly never start from tip
        if G.degree(branch[0])==1:
            branch = branch[::-1]
        for a, b in zip(branch, branch[1:]):
            e = data[(a, b)]
            v = a
            if v in infinite: #infinite[v]:
                if farthest:
                    r = 15 
                else:
                    r = 100000
            else:
                r = geom.circumcircle_radius(*tris[v])
            
            disk = Disk(center=e.center, #verts[v],
                                   r=r,
                                   anchors=e.anchors,
                                   residual=e.residual,
                                   inf=v in infinite) #infinite[v])

            G.graph['disks'][v] = disk
            vpos[v] = disk.center
        #endfor

        # take care of last vertex
        if branch: 
            v = branch[-1]
            if v in infinite: #infinite[v]:
                if farthest:
                    r = 15 
                else:
                    r = 100000
            else:
                r = geom.circumcircle_radius(*tris[v])

            disk = Disk(center=verts[v],
                                   r=r,
                                   anchors=e.anchors,
                                   residual=e.residual,
                                   inf=v in infinite) #infinite[v])
            G.graph['disks'][v] = disk
            vpos[v] = disk.center
            
    # Now sort out missing vertices and forks
    disks = G.graph['disks']

    for v in G.nodes():  
        if v in disks and not G.degree(v) > 2:
            continue

        if v in infinite: #infinite[v]:
            if farthest:
                r = 15 
            else:
                r = 100000
        else:
            r = geom.circumcircle_radius(*tris[v])

        # select anchor pair that has distance closest to twice the radius
        candidate_edges = []
        anchor_dists = [] 
        
        bors = list(G.neighbors(v))

        for bor in bors:
            e = data[(v, bor)]
            candidate_edges.append(e)
            anchor_dists.append(abs(geom.distance(*points[e.anchors]) - r*2))
        
        e = candidate_edges[np.argmin(anchor_dists)]
        disk = Disk(center=verts[v],
                        r=r,
                        anchors=e.anchors,
                        residual=e.residual,
                        inf=v in infinite) #infinite[v])
                        
        G.graph['disks'][v] = disk
        vpos[v] = disk.center
    #endfor

    G.graph['vpos'] = vpos

    delu = G.graph['delu']  
    tris = delu.simplices

    vpos_chord = copy.deepcopy(G.graph['vpos'])
    for v in G.nodes():
        if G.degree(v) <= 2:
            anchors = disks[v].anchors
            p1, p2 = vor.points[anchors[0]], vor.points[anchors[1]]
            vpos_chord[v] = (p1 + p2)/2
        # else:
        #     a, b, c = G.graph['points'][tris[v]]
        #     vpos_chord[v] = geom.circumcenter(a, b, c)
    G.graph['vpos_chord'] = vpos_chord
    return G

def terminal_nodes(G):
    return nodes_with_degree(G, lambda degree: degree==1 )

def nodes_with_degree(G, comp_func):
    nodes = []
    for n in G.nodes():
        if comp_func(G.degree(n)):
            nodes.append(n)
    return nodes

def voronoi_edges(vor, farthest=False):
    ''' Voronoi diagram edges, adapted from SciPy Voronoi drawing code'''

    def at_infinity(v):
        return v[0] < 0 or v[1] < 0

    V = vor.vertices

    finite_segments = []
    infinite_segments = []
    segs = []

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    farthest_mul = 1.
    if farthest:
        farthest_mul = -1
    
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            segs.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * 3000 * farthest_mul #ptp_bound.max()

            segs.append([vor.vertices[i], far_point])
    return segs            

def draw_voronoi(vor, clr, alpha=1., linewidth=0.76, linestyle='-', draw_inf=True, farthest=False, label=''):
    ''' Draw voronoi diagram, adapted from SciPy code'''
    if vor is None:
        return
        
    def at_infinity(v):
        return v[0] < 0 or v[1] < 0

    V = vor.vertices

    finite_segments = []
    infinite_segments = []
    segs = []
    inf_segs = []

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    farthest_mul = 1.
    if farthest:
        farthest_mul = -1
    
    
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            segs.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * 3000 * farthest_mul #ptp_bound.max()

            inf_segs.append([vor.vertices[i], far_point])

    #color(0,1,1,0.4)
    for seg in segs:
        v = np.array(seg)
        x = v[:,0]
        y = v[:,1]
        plut.draw_line([x[0], y[0]], [x[1], y[1]], clr, alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
        label=''
    #color(1,0,1,0.4)
    if draw_inf:
        for seg in inf_segs:
            v = np.array(seg)
            x = v[:,0]
            y = v[:,1]
            #x, y = geom.line_clip(x, y, box)
            #if x is None:
            #    continue
            plut.draw_line([x[0], y[0]], [x[1], y[1]], clr, alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
            label=''

def get_MA_delaunay_triangles(MA):
    delu = MA.graph['delu']  
    nodes = [n for n in MA.nodes() if MA.degree(n) > 0]
    tris = delu.simplices
    tri_MA = []
    for i, tri in enumerate(tris):
        if i in nodes:
            tri_MA.append(tri)
    return tri_MA

def draw_delaunay_in_shape(delu, S, clr='k', alpha=0.5, linestyle='-'):
    ''' Draw Delaunay triangulation'''
    if delu==None:
        return
    tris = delu.points[delu.simplices]
    midpts = []
    for tri in tris:
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            pts = [(a + (b-a)*t) for t in np.linspace(0.1, 0.9, 3)]
            out = False
            for p in pts:
                if not geom.point_in_shape(p, S):
                    out = True

            if not out:
                plut.draw_line(a, b, clr, alpha=alpha, linestyle=linestyle)
                plut.fill_circle((a+b)/2, 0.4, 'm')
                midpts.append((a+b)/2)
    return np.array(midpts).T
        
def draw_delaunay(delu, clr='k', alpha=0.5):
    ''' Draw Delaunay triangulation'''
    if delu==None:
        return
    tris = delu.points[delu.simplices]
    for tri in tris:
        plut.stroke_poly(tri.T, clr, closed=True, alpha=alpha)
        
def draw_pruned_delaunay(MA, delu, clr='k', alpha=0.5, fill=False):
    ''' Draw Delaunay triangulation pruned by VMA threshold'''
    nodes = MA.nodes()
    tris = delu.points[delu.simplices]
    for i, tri in enumerate(tris):
        if i in nodes:
            if fill:
                plut.fill_poly(tri.T, clr, alpha=alpha*0.5)
            plut.stroke_poly(tri.T, clr, alpha=alpha)


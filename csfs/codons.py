''' Utilities to generate codons [Richards and Hoffman (1985) Codon Constraints on Closed 2D Shapes]
    and related features'''
from __future__ import division

import time
import math
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import autograff.geom as geom
import autograff.plut as plut
import autograff.geom.euler_spiral as es

def draw_tangent(p, th, l=50, clr='r'):
    d = np.array([np.cos(th), np.sin(th)])*l
    plut.draw_arrow(p, p+d, clr, style='-|>')

def euler_spiral(p1, p2, th1, th2):
    ts = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
    s1, s2 = es.fit_euler_spiral(th1-ts, th2-ts)
    P = es.euler_spiral(p1, p2, s1, s2)

    return P

def codon(p1, p2, p3, th1, th2, th3, debug_draw=False):
    ''' Creates a codon by merging two Euler spirals
        with a common tangent at the extrema
    '''
    P1 = euler_spiral(p1, p2, th1, th2)
    P2 = euler_spiral(p2, p3, th2, th3)
    if debug_draw:
        draw_tangent(p1, th1)
        draw_tangent(p2, th2)
        draw_tangent(p3, th3)
    return np.hstack([P1, P2[:,1:]])

def superellipse(t, a, b, r):
    return np.vstack([np.sign(np.cos(t))*a*np.power(np.abs(np.cos(t)), 2./r),
                      np.sign(np.sin(t))*b*np.power(np.abs(np.sin(t)), 2./r)])

def ffl_codon(f1, f2, w=0.5, h=1., n=100 ):
    ''' FFL Codon generator
        Draws a corner feature based on the figures in
        Leymarie and Levine 1988, Curvature Morphology.
        f1 and f2 are the powers of the left and right superellipse segments.
        f = 1. results in a straight line
        f > 1 bulges outward
        f < 1 bulges inward
    '''
    t = np.linspace(0,np.pi/2,100)
    P1 = superellipse(t, -w, -h, f1)
    P2 = superellipse(list(reversed(t)), w, -h, f2)
    return np.hstack([P1, P2[:,1:]])

codon_types = ['inf', '0+', '0-', '1+', '1-', '2']

def make_codon(codon_type, debug_draw=False):
    ''' Create a default codon of a given type (See Leyton '85)'''
    if codon_type == 'inf':
        return codon([-100,100], [0,0], [100,-100], -np.pi/4, -np.pi/4, -np.pi/4, debug_draw=debug_draw)
    elif codon_type == '0+':
        return codon([-100,0], [0,100], [100,0], np.pi/2, np.pi, -np.pi/2, debug_draw=debug_draw)
    elif codon_type == '0-':
        amt = 0.9
        P = codon([-100,100], [0,0], [120,130], -np.pi/2 - amt, np.pi, np.pi/2 + amt, debug_draw=debug_draw)
        return P
    elif codon_type == '1+':
        amt = 1.5
        P = codon([100,100], [30,0], [-100,100], -np.pi/2 + amt, amt, -np.pi/2-0.6, debug_draw=debug_draw)
        return P
    elif codon_type == '1-':
        amt = 0.1
        P = codon([100,100], [0,0], [-70,150], np.pi/2 + amt, 0, np.pi/2-0.4, debug_draw=debug_draw)
        return P
    elif codon_type == '2':
        amt = 0.6
        P = codon([100,150], [0,0], [-100,150], np.pi/2 + amt, 0, -np.pi/2-amt, debug_draw=debug_draw)
        return P
    print('Unknown codon type -> ' + codon_type)
    raise ValueError

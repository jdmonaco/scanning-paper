#encoding: utf-8
"""
tools.colormaps -- Custom colormap definitions

Exported namespace: diffmap

Written by Joe Monaco
Center for Theoretical Neuroscience
Copyright (c) 2007-2008 Columbia Unversity. All Rights Reserved.  
"""

from matplotlib.colors import LinearSegmentedColormap as LSC


def get_diffmap_for(M, mid_value=0.0, **kwargs):
    """For a given intensity matrix and mid-point value *mid_value*, return
    the difference map (see diffmap) with the proper mid-point color.
    """
    Mmin, Mmax = M.min(), M.max()
    return diffmap(mid=(mid_value-Mmin) / (Mmax - Mmin), **kwargs)
    

def diffmap(mid=0.5, use_black=False):
    """Conventional differencing map with graded red and blue for values less 
    than and greater than, respectively, the mean of the data. Values approaching 
    the mean are increasingly whitened, and the mean value is white.
    
    Keyword arguments:
    mid -- specify the midpoint value, colored white or black
    use_black -- if True, the midpoint value is black instead of white by default
    """
    m = int(not use_black)
    segmentdata = { 'red':   [(0, 1, 1), (mid, m, m), (1, 0, 0)],
                    'green': [(0, 0, 0), (mid, m, m), (1, 0, 0)],
                    'blue':  [(0, 0, 0), (mid, m, m), (1, 1, 1)] }
    return LSC('RdWhBu', segmentdata)   

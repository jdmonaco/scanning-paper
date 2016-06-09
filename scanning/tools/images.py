#encoding: utf-8
"""
tools.images -- Toolbox functions for creating image output

Exported namespace: image_blast, array_to_rgba, array_to_image, diffmap

Written by Joe Monaco
Center for Theoretical Neuroscience
Copyright (c) 2007-2008 Columbia Unversity. All Rights Reserved.
"""

import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

if sys.platform == "win32":
    import Image
else:
    from PIL import Image


def image_blast(M, savedir, stem='image', fmt='%s_%03d', rev=False, **kwargs):
    """Save a rank-3 stacked intensity matrix *M* to a set of individual PNG
    image files in the directory *savedir*.

    If *savedir* does not exist it will be created. Set **stem** to specify
    the filename suffix.

    Keyword arguments:
    stem -- file name stem to be used for output images
    fmt -- a unique_path fmt specification (need an %s followed by a %d)
    rev -- indicate use of a reversed fmt specification (%d followed by a %s)

    Extra keyword arguments will get passed through to array_to_rgba. See its
    doc string for details.
    """
    assert M.ndim == 3, 'requires rank-3 array of intensity values'
    d = path.realpath(str(savedir))
    if not path.exists(d):
        _os.makedirs(d)
    stem = path.join(d, stem)
    N = M.shape[0]
    first, middle, last = "", "", ""
    for i,m in enumerate(M):
        image_fn = unique_path(stem, fmt=fmt, ext="png", reverse_fmt=rev)
        if i == 0:
            first = image_fn
        elif i == N-1:
            last = image_fn
        array_to_image(m, image_fn, **kwargs)
    if N == 2:
        middle += '\n'
    elif N > 2:
        middle += '\n\t...\n'
    print first, middle, last
    return

def masked_array_to_rgba(mat, mask=None, mask_color='w', **kwds):
    """Convert intensity value matrix to RGBA image array (MxNx4) while setting
    masked values to the color indicated by mask_color (any MPL color spec).

    Remaining keywords are passed to array_to_rgba.
    """
    if mask is None and np.ma.isMA(mat):
        mask = mat.mask
    assert mat.shape == mask.shape, "shape mismatch for mask array"
    from matplotlib.colors import colorConverter
    mask_rgba = tuple(int(255*v) for v in colorConverter.to_rgba(mask_color))
    rgba = array_to_rgba(mat, **kwds)
    rgba[mask] = mask_rgba
    return rgba

def array_to_rgba(mat, cmap=None, norm=True, cmin=0, cmax=1):
    """Intensity matrix (float64) -> RGBA colormapped matrix (uint8)

    Keyword arguments:
    cmap -- a matplotlib.cm colormap object
    norm -- whether the color range is normalized to values in M

    If *norm* is set to False:
    cmin -- minimum clipping bound of the color range (default 0)
    cmax -- maximum clipping bound of the color range (default 1)
    """
    if cmap is None:
        cmap = cm.jet
    elif type(cmap) is str:
        cmap = getattr(cm, cmap)
    M = mat.copy()
    data_min, data_max = M.min(), M.max()
    if norm:
        cmin, cmax = data_min, data_max
    else:
        if cmin > data_min:
            M[M < cmin] = cmin # clip lower bound
        if cmax < data_max:
            M[M > cmax] = cmax # clip uppder bound
    return cmap((M-cmin)/float(cmax-cmin), bytes=True)

def rgba_to_image(M, filename):
    """Save RGBA image matrix (MxNx4) to image file (use PIL fmts)
    """
    if M.ndim != 3 or M.shape[2] != 4:
        raise ValueError, 'requires RGBA image matrix'
    img = Image.fromarray(M, 'RGBA')
    img.save(filename)

def array_to_image(M, filename, mask=None, **kwargs):
    """Save matrix to image file (using PIL formats)

    Arguments:
    M -- 2D intensity matrix of values for producing the image
    filename -- image will be saved to this filename, with the format
        determined by the filename extension
    mask -- optionally specify a masking array

    Remaining keywords are passed to array_to_rgba for scaling, etc. Returns
    True|False to indicate successful|failed image save.
    """
    if mask is None and not np.ma.isMA(M):
        if M.ndim != 2:
            raise ValueError, 'requires 2D instensity value matrix'
        to_rgba = array_to_rgba
    else:
        to_rgba = masked_array_to_rgba
        kwargs.update(mask=mask)
    try:
        rgba_to_image(to_rgba(M, **kwargs), filename)
    except:
        sys.stderr.write('array_to_image: failed saving %s'%filename)
        return False
    return True

def tile2D(M, mask=None, gridvalue=0.5, shape=None):
    """
    Construct a tiled 2D matrix from a 3D matrix

    Keyword arguments:
    mask -- an (H,W)-shaped binary masking array for each cell
    gridvalue -- the intensity value for the grid
    shape -- a (rows, columns) tuple specifying the shape of the tiling to use

    If shape is specified, rows+columns should equal M.shape[0].
    """
    if len(M.shape) != 3:
        return
    N, H, W = M.shape
    if mask is not None and (H,W) != mask.shape:
        mask = None
    if shape and (type(shape) is type(()) and len(shape) == 2):
        rows, cols = shape
    else:
        rows, cols = tiling_dims(N)
    Mtiled = np.zeros((rows*H, cols*W), 'd')
    for i in xrange(N):
        r, c = int(i/cols), np.fmod(i, cols)
        if mask is None:
            Mtiled[r*H:(r+1)*H, c*W:(c+1)*W] = M[i]
        else:
            Mtiled[r*H:(r+1)*H, c*W:(c+1)*W] = mask * M[i]
    Mtiled[H::H,:] = gridvalue
    Mtiled[:,W::W] = gridvalue
    return Mtiled

def tiling_dims(N):
    """Square-ish (rows, columns) for tiling N things
    """
    d = np.ceil(np.sqrt(N))
    return int(np.ceil(N / d)), int(d)

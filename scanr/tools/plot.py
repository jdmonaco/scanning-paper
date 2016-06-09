#encoding: utf-8
"""
tools.plot -- Plotting functions

Exported namespace: cplot

Copyright (c) 2007-2008 Columbia Unversity. All Rights Reserved.  
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from .images import tiling_dims, masked_array_to_rgba


def shaded_error(x, mu, err, **kwds):
    """Plot a shaded error interval (mu-err, mu+err) around mu
    
    Additional arguments as in shaded_region.
    """
    return shaded_region(x, mu - err, mu + err, **kwds)

def shaded_region(x, lower, upper, ax=None, **fmt):
    """Plot a shaded region [lower, upper] over the range x
    
    Remaining keywords are passed to the Polygon constructor. The Polygon 
    object is returned.
    """
    if ax is None:
        ax = plt.gca()
    x, lo, hi = map(np.array, [x, lower, upper])
    style = dict(lw=0, fill=True, zorder=-1, clip_box=ax.bbox)
    style.update(fmt)
    P = Polygon(np.c_[np.r_[x,x[::-1]], np.r_[lo, hi[::-1]]], 
        **style)
    ax.add_artist(P)
    plt.draw()
    return P

def heatmap(x, y, ax=None, interp='nearest', cmask='w', cmap='jet', 
    **hist_kwds):
    """Plot a masked intensity map of (x,y) scatter data
    
    Arguments:
    x, y -- scatter data for the heat map
    ax -- axis to plot into, defaults to current axis
    interp -- an imshow interpolation value, defaults to nearest
    
    Remaining keywords are passed to the histogram2d function. The histogram
    data and x, y bin edges are returned.
    """
    H, xedges, yedges = np.histogram2d(x, y, **hist_kwds)
    densitymap(H, (xedges[0], xedges[-1]), (yedges[0], yedges[-1]), ax=ax,
        interp=interp, cmask=cmask, cmap=cmap)
    return H, xedges, yedges

def densitymap(p, x_range, y_range, ax=None, interp='nearest', cmask='w', 
    **pcolor_kwds):
    """Plot a density map of 2D data where zeros are masked
    
    Arguments:
    p -- density data to plot (first dimension corresponds to x-component)
    x/y_range -- (min, max) bounds of x/y data ranges
    ax -- axis to plot into, defaults to current axis
    interp -- an imshow interpolation value, defaults to nearest
    cmask -- color for masked density elements
    
    Remaining keywords are passed to array_to_rgba (cmap, norm, cmin/max)
    """
    img = masked_array_to_rgba(p.T, mask=(p.T==0), mask_color=cmask, 
        **pcolor_kwds)
    if ax is None:
        ax = plt.gca()
    ax.imshow(img, origin='lower', aspect='auto', interpolation=interp, 
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]], zorder=-100)
    ax.set(xlim=[x_range[0], x_range[-1]], ylim=[y_range[0], y_range[-1]])
    plt.draw()
    return ax
    
def grouped_bar_plot(data, groups, values, errors=None, baselines=None, ax=None, 
    width=0.8, label_str='%s', legend=True, legend_loc=1, **kwds):
    """Grouped bar plot of M groups with N values each
    
    Handles single group bar plots as well (M=1), just pass in 1D arrays and
    a single group name.
    
    Arguments:
    data -- MxN matrix of data values to plot
    groups -- list of group names, to be used as x-tick labels
    values -- list of (name, color) tuple pairs specifying the values within 
        each group with a corresponding mpl color specification
    errors -- optional MxN matrix of errors corresponding to the data
    baselines -- optional MxN matrix of baseline levels to be plotted as
        dashed lines across the corresponding bars
    ax -- axis object to contain the bar plot (default to new figure)
    width -- bar width as a proportion of maximum available horizontal space
    label_str -- formatting string for value names used for the legend
    legend -- whether to display the values legend
    legend_loc -- legend location passed to MPL legend command
    
    Remaining keywords are passed to the MPL bar command. Rectangle handles 
    for the bars are returned.
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()
    data = np.asarray(data)
    if errors is not None:
        errors = np.asarray(errors)
        assert errors.shape==data.shape, 'errors mismatch with data shape'
    if data.ndim == 1:
        groups = [groups]
        data = np.asarray([data])
        if errors is not None:
            errors = np.asarray([errors])
        if baselines is not None:
            baselines = np.asarray([baselines])
    if type(values[0]) not in (list, tuple):
        colors = 'bgrcmykw'
        values = [(name, colors[np.fmod(i, len(colors))]) for i,name in 
            enumerate(values)]
    
    value_list = [ name for name, color in values ]
    color_dict = { name: color for name, color in values }
    group_size = len(value_list)
    bar_width = width / group_size
    centers = np.arange(0, len(groups))
    lefts = []
    heights = []
    yerr = []
    colors = []
    for i, group in enumerate(groups):
        for j, value in enumerate(value_list):
            lefts.append(i+(j-group_size/2.0)*bar_width)
            heights.append(data[i, j])
            colors.append(color_dict[value])
            if errors is not None:
                yerr.append(errors[i, j])
            if baselines is not None:
                ax.plot([lefts[-1], lefts[-1]+bar_width], 
                    [baselines[i, j]]*2, 'k--', lw=2, zorder=10)

    bar_kwds = dict(width=bar_width, color=colors, linewidth=1)
    bar_kwds.update(**kwds)
    if errors is not None:
        bar_kwds.update(yerr=yerr, capsize=0, ecolor='k')
        
    h = ax.bar(lefts, heights, **bar_kwds)
    if legend:
        ax.legend(h[:group_size], [label_str%value for value in value_list], 
            loc=legend_loc)
        
    ax.set_xticks(centers)
    ax.set_xticklabels(groups, size='small')
    ax.set_xlim(-0.5, len(groups)-0.5)
    return h

def quicktitle(ax, text, **kwds):
    """Put short title on top an axis plot, optimized for low-margin plots
    
    Keywords are passed to ax.text(...)
    """
    text_fmt = dict(ha='center', va='bottom', size='small', zorder=100)
    text_fmt.update(**kwds)
    text_fn = hasattr(ax, 'text2D') and ax.text2D or ax.text
    h = text_fn(0.5, 1.0, unicode(text), color='k', transform=ax.transAxes, **text_fmt)
    plt.draw()
    return h

def textlabel(ax, text, side='right', **kwds):
    """Put short text label in a box on the top right corner of an axis plot
    
    Keywords are passed to ax.text(...)
    """
    text_fmt = dict(ha=side, va='top', size='medium', zorder=100)
    text_fmt.update(**kwds)
    text_fn = hasattr(ax, 'text2D') and ax.text2D or ax.text
    h = text_fn(dict(left=0, right=1)[side], 1, unicode(text), color='k', 
        transform=ax.transAxes, bbox=dict(fc='w'), **text_fmt)
    plt.draw()
    return h

def cplot(x, y, c, lw=6, **kw):
    """Phase plane line plot where each point is colored (uses scatter)
    
    Keywords are passed to the pylab *scatter* call.
    """
    kw.update(s=lw, marker='o', linewidths=(0,), edgecolors='none')
    return plt.scatter(x, y, c=c, **kw)


class AxesList(list):
    
    def add_figure(self, f=None):
        if f is None:
            f = plt.gcf()
        elif type(f) is int:
            f = plt.figure(2)
        self.extend(filter(lambda a: hasattr(a, "draw_artist"), 
            f.get_children()))
    
    def make_panels(self, position, projection={}):
        """Create axes with customized positions mapped to keys
        
        The position description should be a mapping (dict) from keys to position
        (l, b, w, h) rects, subplot (r,c,N) tuples, or subplot integer codes (e.g.,
        221). The specified axes are created in the current figure and a mapping
        from keys to subplot axes is returned. 
        
        Optionally, non-standard axes projections ('3d', 'polar') can be specified
        in the projection dict.
        """
        f = plt.gcf()
        f.clf()
        axdict = {}
        if '3d' in projection.values():
            from mpl_toolkits.mplot3d import Axes3D
        for k in position:
            pos = position[k]
            proj = projection.get(k, 'rectilinear')
            if np.iterable(pos) and len(pos) == 3:
                axdict[k] = f.add_subplot(*pos, projection=proj)
            elif np.iterable(pos) and len(pos) == 4:
                axdict[k] = f.add_axes(pos, projection=proj)
            elif type(pos) is int:
                axdict[k] = f.add_subplot(pos, projection=proj)
            else:
                raise ValueError, 'bad subplot/axes: %s'%pos
        self.extend(axdict.values())
        return axdict
    
    def make_grid(self, shape):
        """Create a grid of subplots in the current figure, based on either the
        number of required plots or a (nrows, ncols) tuple
        """
        f = plt.gcf()
        f.clf()
        if type(shape) is int:
            N = shape
            rows, cols = tiling_dims(N)
        elif type(shape) is tuple and len(shape) == 2:
            rows, cols = shape
            N = rows*cols
        else:
            raise ValueError, 'invalid grid shape parameter: %s'%str(shape)
        for r in xrange(rows):
            for c in xrange(cols):
                panel = cols*r + c + 1
                if panel > N:
                    break
                plt.subplot(rows, cols, panel)
        self.add_figure(f)
    
    def bounding_box(self):
        lims = np.array([ax.axis() for ax in self]).T
        left, right = lims[0].min(), lims[1].max()
        bottom, top = lims[2].min(), lims[3].max()
        return left, right, bottom, top
    
    def add_padding(self, bbox, factor=0.1):
        dx, dy = bbox[1]-bbox[0], bbox[3]-bbox[2]
        left, right = bbox[0]-dx*factor, bbox[1]+dx*factor
        bottom, top = bbox[2]-dy*factor, bbox[3]+dy*factor
        return left, right, bottom, top
    
    def xnorm(self, padding=None):
        return self.normalize(padding=padding, yaxis=False)
    def ynorm(self, padding=None):
        return self.normalize(padding=padding, xaxis=False)
        
    def normalize(self, padding=None, xaxis=True, yaxis=True):
        bbox = self.bounding_box()
        if padding:
            bbox = self.add_padding(bbox, factor=float(padding))
        for ax in self:
            if xaxis: 
                ax.set(xlim=bbox[:2])
            if yaxis: 
                ax.set(ylim=bbox[2:])
        return self
    
    def map(self, func, args):
        assert len(args) == len(self), 'argument size mismatch'
        plt.ioff()
        [getattr(self[i], func)(args[i]) for i in xrange(len(self))]
        self.draw()
        return self
    
    def apply(self, func="set", *args, **kwds):
        plt.ioff()
        [getattr(ax, func)(*args, **kwds) for ax in self]
        self.draw()
        return self
    
    def set(self, **kwds):
        return self.apply(**kwds)
    
    def axis(self, mode):
        [ax.axis(mode) for ax in self]
        return self
    def equal(self):
        return self.axis('equal')
    def scaled(self):
        return self.axis('scaled')
    def tight(self):
        return self.axis('tight')
    def image(self):
        return self.axis('image')
    def off(self):
        return self.axis('off')
    def on(self):
        [ax.set_axis_on() for ax in self]
        return self
    
    def gallery(self):
        self.equal().normalize().off()

    def draw(self):
        plt.ion()
        plt.draw()

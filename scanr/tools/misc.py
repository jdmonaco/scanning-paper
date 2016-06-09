#encoding: utf-8
"""
tools.misc -- Miscellaneous toolbox functions

Exported namespace: contiguous_groups, fit_exp_linear, halfwave,
    set_max_recursion, Null

Copyright (c) 2007-2008 Columbia Unversity. All Rights Reserved.
"""

from __future__ import division, print_function

from collections import deque
from functools import wraps
import cPickle
import numpy as np
import sys


def align_center(M, ix, align='rows'):
    """Create a center-aligned copy of a 2D matrix, where the aligned rows or
    columns are wrapped around

    For matrix with even number of columns, the lower-index column around the
    center is used as the alignment index.

    Arguments:
    M -- 2D matrix to be aligned
    ix -- index array for bins to align in each row or column
    align -- whether to align 'rows' or 'columns' (default 'rows')
    """
    if align == 'cols':
        M = M.T

    R, C = M.shape
    W = np.empty_like(M)
    delta = ix - int((C - 0.5) / 2)
    for i in xrange(R):
        d = delta[i]
        for j in xrange(C):
            W[i, j] = M[i, (j+d)%C]

    if align == 'cols':
        return W.T
    return W


def view_wait_with_status(view, out, timeout=60):
    """Wait on an IPython parallel engine view, outputting a full status
    message periodically (set by timeout).
    """
    while not view.wait(timeout=timeout):
        status = view.queue_status()
        msg = ['Engine View Status:']
        for eid in sorted(status.keys()):
            if type(eid) is int:
                msg.append('Engine %d: '%eid + ', '.join(
                    ['%d %s'%(status[eid][k], k)
                        for k in sorted(status[eid].keys())]))
            else:
                msg.append('%s: %d'%(eid.title(), status[eid]))
        out('\n'.join(msg))
    return


def chain(source, *transforms):
    """Chain a series of generators in intuitive order
    http://tartley.com/?p=1471
    """
    args = source
    for transform in transforms:
        args = transform(args)
    return args

def memoize(f):
    cache = {}
    @wraps(f)
    def wrapper(*args, **kwds):
        key = (tuple(args), tuple(sorted(kwds.iteritems())))
        if key not in cache:
            cache[key] = f(*args, **kwds)
        return cache[key]
    return wrapper

def memoize_limited(max_cache_size):
    def real_memoize_limited(f):
        cache = {}
        key_queue = deque()
        @wraps(f)
        def wrapper(*args, **kwds):
            key = (tuple(args), tuple(sorted(kwds.iteritems())))
            if key not in cache:
                cache[key] = f(*args, **kwds)
                key_queue.append(key)
                if len(key_queue) > max_cache_size:
                    was_cached =cache.pop(key_queue.popleft(), None)
            return cache[key]
        return wrapper
    return real_memoize_limited


class Reify(object):

    """
    Convert nested dictionary into hierarchical tree object
    """

    def __init__(self, *args, **onto):
        for arg in args:
            if type(arg) is AutoVivification:
                arg = arg.dict_copy()
            if type(arg) is dict:
                onto.update(arg)
        self.__name__ = onto.pop('__name__', 'root')
        for key in onto.keys():
            if ' ' in key or '-' in key:
                print('warning: cannot reify key: %s' % key)
                continue
            if type(onto[key]) is dict:
                onto[key]['__name__'] = '%s.%s' % (self.__name__, key)
                setattr(self, key, Reify(onto[key]))
            else:
                setattr(self, key, onto[key])

    def __str__(self):
        return self.__name__

    def __repr__(self):
        n = len(self.__dict__) - 1
        return '%s: %d %s' % (str(self), n, (n>1) and 'children' or 'child')


class DataSpreadsheet(object):

    """
    Pass in a filename and list(tuple(colname, 's|d|f')) to define columns,
    call get_record for a record dictionary, fill it up, and then call
    write_record and close when you're done.
    """

    def __init__(self, fn, cols, sep=','):
        self.col_init = dict(s='', d=0, f=0.0)
        self.cols = cols
        self.record_string = sep.join(['%%(%s)%s'%col for col in self.cols])+'\n'
        self.filename = fn
        self.spreadsheet = file(self.filename, 'w')
        self.spreadsheet.write(','.join(self.get_col_names())+'\n')
        sys.stdout.write('Opened spreadsheet %s.\n'%self.filename)

    def get_col_names(self):
        return [col for col, dtype in self.cols]

    def get_record_string(self):
        return self.record_string

    def get_record(self):
        return { col: self.col_init[dtype] for col, dtype in self.cols }

    def write_record(self, record):
        self.spreadsheet.write(self.record_string%record)

    def close(self):
        self.spreadsheet.close()
        sys.stdout.write('Closed spreadsheet %s.\n'%self.filename)


def outer_pairs(A, B):
    """Generator for iterating through all possible pairs of items in the
    given pair of sequences.
    """
    for first in A:
        for second in B:
            yield (first, second)

def unique_pairs(seq, cross=None):
    """Generator for iterating through all the unique pairs of items in the
    given sequence.
    """
    N = len(seq)
    for i, first in enumerate(seq[:-1]):
        for j in xrange(i+1, N):
            second = seq[j]
            yield (first, second)

def contiguous_groups(ingroup, min_size=1):
	"""Find contiguous groups of samples in a binary group membership array

    Arguments:
    ingroup -- array indicating in/out group membership as True/False
    min_size -- minimum group size to detect

    Returns list of slice index tuples. Note, because each tuple represents
    a slice, the second element is non-inclusive (i.e., (first_in_group,
    first_out_of_group)).
	"""
	ingroup = np.asarray(ingroup).astype('?')
	last_ix = len(ingroup) - 1
	groups = []
	cursor = 0
	new_start = -1
	while cursor <= last_ix:
		if ingroup[cursor]:
			if new_start == -1:
				new_start = cursor
			while cursor <= last_ix and ingroup[cursor]:
				cursor += 1
		elif new_start >= 0:
			if cursor - new_start >= min_size:
				groups.append((new_start, cursor))
			new_start = -1
			while cursor <= last_ix and not ingroup[cursor]:
				cursor += 1
		else:
		    cursor += 1
	if new_start >= 0 and cursor - new_start >= min_size:
		groups.append((new_start, cursor))
	return groups

def merge_adjacent_groups(groups, tol=0):
    """Combine adjacent index groups as returned by contiguous_groups with
    some allowable tolerance for gaps between groups.
    """
    merged = []
    if len(groups):
        groups = map(list, sorted(groups)) # sort and listify indices
        merged.append(groups[0])
        for g in groups[1:]:
            if g[0] <= merged[-1][1] + tol:
                merged[-1][1] = g[1]
            else:
                merged.append(g)
    return merged

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

def halfwave(x, copy=False):
    """Half-wave rectifier for arrays or scalars

    NOTE: Specify copy=True if array data should be copied before performing
    halwave rectification.
    """
    if type(x) is np.ndarray and x.ndim:
        if copy:
            x = x.copy()
        x[x<0.0] = 0.0
    else:
        x = float(x)
        if x < 0.0:
            x = 0.0
    return x

def ravelnz(a):
	"""Generator yields non-zero elements of an array
	"""
	for b in np.asarray(a).ravel():
		if b and np.isfinite(b):
		    yield b

def set_max_recursion():
    """Set platform-dependent maximum recursion depth

    NOTE: These values were obtained using the find_recursionlimit.py
    script that comes with python.
    """
    if sys.platform == 'darwin':
        sys.setrecursionlimit(4400)
    elif sys.platform == 'linux2':
        sys.setrecursionlimit(6700)
    return


class AutoVivification(dict):

    """
    Implementation of perl's autovivification feature

    From:
    http://stackoverflow.com/questions/635483/what-is-the-best-way-to-
        implement-nested-dictionaries-in-python
    """

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def dict_copy(self):
        """Return a pure dict deep copy of this tree
        """
        mydict = {}

        def populate(base, d):
            """Recursively copy base into d
            """
            for k in base.keys():
                v = base[k]
                if type(v) is type(base):
                    d[k] = dict()
                    populate(v, d[k])
                else:
                    d[k] = v

        populate(self, mydict)
        return mydict


class Null(object):

    """
    Null object design pattern

    From:
    Python Cookbook, Second Edition: Recipe 6.17
    """

    def __new__(cls, *p, **kw):
        "force only one instance"
        if '_inst' not in vars(cls):
            cls._inst = object.__new__(cls, *p, **kw)
        return cls._inst

    def __init__(self, *p, **kw): pass
    def __call__(self, *p, **kw): return self
    def __str__(self): return "Null()"
    def __repr__(self): return "Null()"
    def __nonzero__(self): return False
    def __getattr__(self, name): return self
    def __delattr__(self, name): return self
    def __setattr__(self, name): return self
    def __getitem__(self, i): return self
    def __setitem__(self, *p): pass

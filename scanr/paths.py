# encoding: utf-8
"""
scanr.paths -- Functions for getting directory, file, and H5 group paths into
    the filesystem data tree or the H5 data file

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os, sys
import tables as tb
import numpy as np
from glob import glob
from os.path import join as opj

# Package imports
from .config import Config

# Configuration shortcuts
CfgPath = Config['path']
CfgData = Config['h5']
if 'file_types' not in locals():
    file_types = tuple(k[:-5] for k in CfgPath if k.endswith('_file'))
if 'tree_types' not in locals():
    tree_types = tuple(k[:-5] for k in CfgData if k.endswith('_tree'))


# Get paths to files

def get_data_file_path(ftype, rat=None, day=None, tetrode=None, session=None,
    cluster=None, do_glob=False, search=True):
    """Construct a full path to a data file under the data root directory

    The required string argument must match of the detected file types listed
    under scanr.paths.file_types.

    Keyword arguments provide the necessary information to specify the file.
    All file types require *rat* and *day*; files within tetrode directories
    typically require *tetrode*, *session*, and *cluster*.

    Keyword arguments:
    rat -- rat id number
    day -- recording day number
    tetrode -- tetrode number
    session -- recording session id number (i.e., 1-5 for double rotation)
    cluster -- cluster id number
    do_glob -- return a list of glob-matching files; ftype must be a config
        [path] key with a _glob suffix as well as a _file suffix
    search -- if file does not exist, search up the directory tree until
        found or return None (default True)

    Returns fully qualified path string to the data file if it exists,
    otherwise returns None.
    """
    if ftype not in file_types:
        raise ValueError, 'file type must be one of %s'%str(file_types)

    suffix = do_glob and 'glob' or 'file'
    pathspec = dict(    rat=rat,
                        day=day,
                        tetrode=tetrode )
    filespec = dict(    tetrode=pathspec['tetrode'],
                        session=session,
                        cluster=cluster )

    def generate_path():
        file_fmt = CfgPath['_'.join([ftype, suffix])]
        try:
            _newpath = opj(get_path(**pathspec), file_fmt%filespec)
        except TypeError:
            raise TypeError, 'missing formatting info for %s file: \'%s\''% \
                (ftype, file_fmt)
        else:
            return _newpath

    def there(fn):
        # if Config['debug_mode']:
        #     sys.stdout.write('Trying %s\n'%fn)
        return os.path.exists(fn)

    if do_glob:
        return glob(generate_path())
    else:
        fpath = generate_path()
        if not search or there(fpath):
            return fpath
        else:
            # Handle filetype-specific variations on filenames
            if ftype == 'center':
                fpath = fpath[:-3] # truncate .mX extension
                if there(fpath):
                    return fpath
            elif ftype in ('cluster', 'spiketrain'):
                tail, head = os.path.split(fpath)
                for pre in 'hex', 'ring':
                    check = opj(tail, pre+head)
                    if there(check):
                        return check

            # If not initially found, search up the directory tree
            _pathspec = pathspec.copy()
            for step in 'tetrode', 'day', 'rat':
                pathspec[step] = None
                fpath = generate_path()
                if there(fpath):
                    return fpath
            if Config['debug_mode']:
                sys.stderr.write(
                    'data_file_path: %s not found for %s and %s\n'%
                        (ftype, _pathspec, filespec))
            return None

# Get paths to directories

def get_path(rat=None, day=None, tetrode=None):
    """Construct a full path to a directory in the data root

    The number of consecutive arguments passed in determine the depth of the
    path that is constructed and returned.

    By default, paths are relative to Config['data_root'].

    Keyword arguments:
    rat -- rat id number
    day -- recording day number
    tetrode -- tetrode number

    Returns fully qualified path string to the data directory.
    """
    d = dict(rat=rat, day=day, tetrode=tetrode)
    thepath = Config['data_root']
    if rat is not None:
        thepath = opj(thepath, CfgPath['subject_dir']%d)
        if day is not None:
            thepath = opj(thepath, CfgPath['session_dir']%d)
            if tetrode is not None:
                thepath = opj(thepath, CfgPath['tetrode_dir']%d)
    return thepath

# Get paths to groups

def get_group_path(tree='kdata', name=None, rat=None, day=None, session=None):
    """Construct a full path to a group node in the H5 data file

    The H5 data file has different trees, so the tree is specified by the
    *tree* keyword, which defaults to \'kdata\'. Remaining arguments same
    as for get_path().

    Returns fully qualified path string to the data group node.
    """
    d = dict(rat=rat, day=day, session=session)
    if tree not in tree_types:
        raise ValueError, 'tree must be one of %s'%str(tree_types)
    thepath = CfgData[tree + '_tree']
    if tree == 'kdata':
        if rat is not None:
            thepath += '/' + CfgData['rat']%d
            if day is not None:
                thepath += '/' + CfgData['day']%d
                if session is not None:
                    thepath += '/' + CfgData['session']%d
    elif name is not None:
        thepath += '/' + str(name)
    return thepath

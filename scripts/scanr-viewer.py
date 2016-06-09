#!/usr/bin/env python

"""
scanr_viewer.py -- Script for running the ScanrGUI visualization

Created by Joe Monaco on May 22, 2012.
Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License.
See http://www.opensource.org/licenses/mit-license.php.
"""

if __name__ == '__main__':
    from scanr.gui import ScanrGUI
    ScanrGUI(debug=False).configure_traits()
else:
    print 'Please run as a script.'


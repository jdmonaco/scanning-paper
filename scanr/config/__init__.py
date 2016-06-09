# encoding: utf-8
"""
scanr.config -- Load analysis parameters from customizable configuration file

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

import os, sys
from configobj import ConfigObj, ConfigObjError, flatten_errors
from validate import Validator

# Get configuration and spec file paths
## Testing for %APPDATA% and %HOMEDRIVE% take care of Win32, while the rest
## fall back to ~/.scanr.
if 'APPDATA' in os.environ:
    config_home = os.environ['APPDATA']
    config_fn = 'scanr.ini'
elif 'HOMEDRIVE' in os.environ:
    config_home = os.environ['HOMEDRIVE']
    config_fn = 'scanr.ini'
else:
    config_home = os.environ['HOME']
    config_fn = '.scanr'
config_home = os.path.normpath(config_home)
config_path = os.path.join(config_home, config_fn)
spec_path = os.path.join(os.path.dirname(__file__), 'defaults.rc')

# Create config object based on file if it exists
try:
    cfg = ConfigObj(config_path, configspec=spec_path, interpolation=False)
except ConfigObjError:
    raise ValueError, 'Could not read config file.'

# Validate the configuration based on spec
val = Validator()
if os.path.exists(cfg.filename):
    check = cfg.validate(val)
else:
    sys.stdout.write('No config found: setting up new config file:\n%s\n'%('-'*44))
    cfg['data_root'] = \
        os.path.abspath(raw_input('Enter path to the root data directory:\n\t'))
    check = cfg.validate(val, copy=True)

# Handle configuration errors
if check != True:
    sys.stderr.write('Found config validation errors:\n')
    for (seclist, key, junk) in flatten_errors(cfg, check):
        if key is None:
            sys.stderr.write('\t- Missing section %s'%('/'.join(seclist)))
        else:
            sys.stderr.write('\t- In %s section, \'%s\' option failed\n'%
                ('/'.join(seclist), key))
            
elif not os.path.exists(cfg.filename):
    sys.stdout.write('Saving new config file: %s\n'%cfg.filename)
    cfg.write()   
else:
    sys.stdout.write('Found valid config file at: %s\n'%cfg.filename)

# Make configuration package available as global Config
Config = cfg

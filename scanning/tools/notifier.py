"""
notifier.py -- Logging and system notifications for AutoTeXify
    
Author: Joe Monaco <self@joemona.co>, github.com/jdmonaco
Last updated: June 5, 2014
"""

from __future__ import division, print_function

import os
import sys
import time
import subprocess

from .shell import Shell


class Notifier(object):
    
    def __init__(self, prog=None, log=True):
        self.prog = prog is None and sys.argv[0] or prog
        self._do_log = log
        self.on()
        
    def on(self):
        self._notifier = Shell.which('terminal-notifier')
        self._osascript = Shell.which('osascript')
    
    def off(self):
        self._notifier = self._osascript = None
    
    def notify(self, msg, title=None, subtitle=None):
        if title is None:
            title = self.prog
        s = msg[0].upper() + msg[1:]
        if self._notifier:
            self._terminal_notify(s, title, subtitle)
        elif self._osascript:
            self._applescript_notify(s, title, subtitle)
        if subtitle:
            self.log(subtitle, ':', s)
        else:
            self.log(s)
    
    def log(self, *msg):
        if self._do_log:
            print('[', self.prog, ':', time.strftime('%I:%M:%S %p'), ']', *msg)

    def remove(self):
        if self._notifier:
            self._terminal_notify_remove()
    
    def _terminal_notify(self, msg, title, subtitle):
        cmd = [self._notifier]
        cmd.extend(['-message', msg])
        cmd.extend(['-title', title])
        if subtitle:
            cmd.extend(['-subtitle', subtitle])
        cmd.extend(['-group', self.prog])
        if subprocess.call(cmd) != 0:
            self.log('warning: failed to send terminal-notifier msg:', msg)
            
    def _terminal_notify_remove(self, tries=5, delay=1.0):
        while subprocess.check_output([self._notifier, '-remove', self.prog]) and tries:
            time.sleep(delay)
            tries -= 1
    
    def _applescript_notify(self, msg, title, subtitle):
        cmd = 'display notification "%s" with title "%s"' % (msg, title)
        if subtitle:
            cmd += 'subtitle "%s"' % subtitle
        if subprocess.call([self._osascript, '-e', cmd]) != 0:
            self.log('warning: failed to send osascript msg:', msg)
                    

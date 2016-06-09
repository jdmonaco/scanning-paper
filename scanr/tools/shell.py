"""
Replication of some shell functions.
"""

import os


class Shell(object):
    
    @classmethod
    def which(self, cmd, mode=os.F_OK | os.X_OK):
        def _access_check(fn, mode):
            return (os.path.exists(fn) and os.access(fn, mode)
                    and not os.path.isdir(fn))
        
        if os.path.dirname(cmd):
            if _access_check(cmd, mode):
                return cmd
            return None
        
        path = os.environ.get("PATH", os.defpath)
        if not path:
            return None
        path = path.split(os.pathsep)
        
        seen = set()
        for adir in path:
            normdir = os.path.normcase(adir)
            if not normdir in seen:
                seen.add(normdir)
                name = os.path.join(adir, cmd)
                if _access_check(name, mode):
                    return name
        return None
#!/usr/bin/env python
from distutils.core import setup
import os


OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'


def get_tcl_root():
    """Get the root tcl folder - prefer environment variable if is set, but
    otherwise default to the parent folder of the python project. In either
    case check for the existence of the library file.
    """
    from os.path import dirname, realpath, isfile, join

    default = dirname(dirname(realpath(__file__)))
    TCL_ROOT = os.environ.get('TCL_ROOT', default)

    if not isfile(join(TCL_ROOT, 'lib', 'libtcl.so')):
        raise OSError("Could not find $TCL_ROOT/lib/libtcl.so - please make "
                      "sure TCL_ROOT points to a directory with the built "
                      "library.")

    return TCL_ROOT


def write_config():
    from configparser import ConfigParser

    config = ConfigParser()
    config['lib'] = {'TCL_ROOT': get_tcl_root()}

    with open(os.path.join("tcl", "tcl.cfg"), "w") as configfile:
        config.write(configfile)


write_config()


setup(name="tcl",
      version="0.1.0",
      description="Tensor Contraction Library",
      author="Paul Springer",
      author_email="springer@aices.rwth-aachen.de",
      packages=['tcl'],
      package_data={'tcl': ['tcl.cfg']},
      )



print("")
output = "# "+ FAIL + "IMPORTANT"+ENDC+": execute 'export TCL_ROOT=%s/../' #"%(os.path.dirname(os.path.realpath(__file__)))
print('#'*(len(output)-2*len(FAIL)+1))
print(output)
print('#'*(len(output)-2*len(FAIL)+1))
print("")

import platform
from collections import namedtuple
from os import path

System = namedtuple('System', 'os arch kernel os_version')

def get_system():
    """
    Returns system info as a namedtuple

    import build_utils as bu
    bu.get_system()
    System(os='macos', arch='x86_64', kernel='17.5.0', os_version='17.5.0')
    """
    # sys = platform.platform().split('-')
    kernel = platform.release()
    os = platform.system()  # tell me linux or darwin
    if os == 'Linux':
        dist = platform.linux_distribution()
        os = dist[0]
        os_version = dist[1]
        arch = platform.machine()
        if not ((arch.find('armv6') > -1) or (arch.find('armv7') > -1)):
            raise Exception("This os is not supported")
    elif os == 'Darwin':
        os = 'macos'
        os_version = platform.mac_ver()[0]
        if platform.machine() == 'x86_64':
            arch = sys[2]
        else:
            raise Exception("This os is not supported")
    else:
        raise Exception("This os is not supported")

    return System(os, arch, kernel, os_version)

# from psutils
def get_pkg_version(relfile):
    """
    Given a relative path to a file, this searches it and returns the version
    string. The nice thing about this, it doesn't execute any python code when
    it searches. This is an issues during setup when I have C extensions that
    have not been compiled yet, but python code looks for the library (which
    is not built yet).

    import build_utils as bu
    bu.get_pkg_version('build_utils/__init__.py')
    "0.2.0"
    """
    # HERE = path.abspath(path.dirname(__file__))
    # INIT = path.join(HERE, pkg + '/' + file)
    # INIT = pkg
    pkg = path.abspath(relfile)
    with open(pkg, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                ret = eval(line.strip().split(' = ')[1])
                assert ret.count('.') == 2, ret
                for num in ret.split('.'):
                    assert num.isdigit(), ret
                return ret
        raise ValueError("couldn't find version string")

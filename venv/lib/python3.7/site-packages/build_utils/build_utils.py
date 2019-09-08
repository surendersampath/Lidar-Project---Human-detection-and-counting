
from setuptools.command.test import test as TestCommand
from setuptools.dist import Distribution
import os
from setuptools import Command
import sys
from colorama import Fore, Back, Style
from shutil import rmtree


class BinaryDistribution(Distribution):
    binary = False

    def is_pure(self):
        # if self.binary:
        #     os.system('rm -fr src/build')
        #     os.system('rm -fr src/lib')
        #     os.system('cd src && mkdir build && cd build && cmake .. && make')
        return self.binary


class BuildCommand(Command):
    """Build binaries/packages"""
    pkg = None
    test = True     # run tests
    py2 = True      # build python 2 package
    py3 = True      # build python 3 package
    rm_egg = False  # rm egg-info directory
    rm_so = False   # rm shared library, this is for c extensions

    description = 'Build and publish the package.'

    # # The format is (long option, short option, description)
    user_options = []

    def initialize_options(self):
        # Each user option must be listed here with their default value.
        pass

    def finalize_options(self):
        pass

    def cprint(self, color, msg):
        print(color + msg + Style.RESET_ALL)

    def rmdir(self, folder):
        try:
            rmtree(folder)
            self.cprint(Fore.RED, ">> Deleted Folder {}".format(folder))
        except OSError:
            pass

    def rm(self, file):
        try:
            os.system('rm {}'.format(file))
            self.cprint(Fore.RED, ">> Deleted File {}".format(file))
        except OSError:
            pass

    def run(self):
        if not self.pkg:
            raise Exception('BuildCommand::pkg is not set')

        print(Fore.BLUE + '+----------------------------------')
        print('| Package: {}'.format(self.pkg))
        print('+----------------------------------')
        print('| Python 2: tests & build: {}'.format(self.py2))
        print('| Python 3: tests & build: {}'.format(self.py3))
        print('+----------------------------------\n\n' + Style.RESET_ALL)

        pkg = self.pkg
        self.cprint(Fore.RED, 'Delete dist directory and clean up binary files')
        self.cprint(Fore.RED, '-----------------------------------------------')

        self.rmdir('dist')
        self.rmdir('build')
        self.rmdir('.eggs')
        if self.rm_egg:
            self.rmdir('{}.egg-info'.format(pkg))
        if self.rm_so:
            self.rm('*.so')
            self.rm('{}/*.so'.format(pkg))
        self.rm('{}/*.pyc'.format(pkg))
        self.rmdir('{}/__pycache__'.format(pkg))
        self.cprint(Fore.RED, '-----------------------------------------------\n\n')

        if self.test:
            print('Run Nose tests')
            if self.py2:
                ret = os.system("unset PYTHONPATH; python2 -m nose -w tests -v test.py")
                if ret > 0:
                    self.cprint(Fore.WHITE + Back.RED, '<<< Python2 nose tests failed >>>')
                    return
            if self.py3:
                ret = os.system("unset PYTHONPATH; python3 -m nose -w tests -v test.py")
                if ret > 0:
                    self.cprint(Fore.WHITE + Back.RED, '<<< Python3 nose tests failed >>>')
                    return

        print('Building packages ...')
        self.cprint(Fore.WHITE + Back.MAGENTA,'>> Python source ----------------------------------------------')
        os.system("unset PYTHONPATH; python setup.py sdist")
        if self.py2:
            self.cprint(Fore.WHITE + Back.CYAN,'>> Python 2 Wheel ---------------------------------------------------')
            os.system("unset PYTHONPATH; python2 setup.py bdist_wheel")
        if self.py3:
            self.cprint(Fore.WHITE + Back.BLUE,'>> Python 3 Wheel ---------------------------------------------------')
            os.system("unset PYTHONPATH; python3 setup.py bdist_wheel")

# class BuildCommand(TestCommand):
#     """Build binaries/packages"""
#     pkg = None
#     test = True
#     py2 = True
#     py3 = True
#
#     def run_tests(self):
#         if not self.pkg:
#             raise Exception('BuildCommand::pkg is not set')
#
#         print('+----------------------------------')
#         print('| Package: {}'.format(self.pkg))
#         print('+----------------------------------')
#         print('| Python 2: tests & build: {}'.format(self.py2))
#         print('| Python 3: tests & build: {}'.format(self.py3))
#         print('+----------------------------------\n\n')
#
#         pkg = self.pkg
#         print('Delete dist directory and clean up binary files')
#         os.system('rm -fr dist')
#         os.system('rm -fr build')
#         os.system('rm -fr .eggs')
#         # os.system('rm -fr {}.egg-info'.format(pkg))
#         os.system('rm {}/*.pyc'.format(pkg))
#         os.system('rm -fr {}/__pycache__'.format(pkg))
#
#         if self.test:
#             print('Run Nose tests')
#             if self.py2:
#                 ret = os.system("unset PYTHONPATH; python2 -m nose -w tests -v test.py")
#                 if ret > 0:
#                     print('<<< Python2 nose tests failed >>>')
#                     return
#             if self.py3:
#                 ret = os.system("unset PYTHONPATH; python3 -m nose -w tests -v test.py")
#                 if ret > 0:
#                     print('<<< Python3 nose tests failed >>>')
#                     return
#
#         print('Building packages ...')
#         print('>> Python source ----------------------------------------------')
#         os.system("unset PYTHONPATH; python setup.py sdist")
#         if self.py2:
#             print('>> Python 2 ---------------------------------------------------')
#             os.system("unset PYTHONPATH; python2 setup.py bdist_wheel")
#         if self.py3:
#             print('>> Python 3 ---------------------------------------------------')
#             os.system("unset PYTHONPATH; python3 setup.py bdist_wheel")


class SetGitTag(Command):
    """Set version tag on github"""

    description = 'Tags the package in the git repo'
    user_options = []
    version = None

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if self.version is None:
            raise Exception("SetGitTag needs version set to something")

        print('Pushing git tags')
        os.system('git tag v{0}'.format(self.version))
        os.system('git push --tags')


class PublishCommand(TestCommand):
    """Publish to Pypi"""
    pkg = None
    version = None

    def run_tests(self):
        if not self.pkg or not self.version:
            raise Exception('PublishCommand::pkg or version is not set')

        pkg = self.pkg
        version = self.version
        print('Publishing to PyPi ...')
        os.system("unset PYTHONPATH; twine upload dist/{}-{}*".format(pkg, version))

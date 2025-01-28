from setuptools import find_packages, setup

VERSION = '0.0.1'
DESCRIPTION = "Geodesiq package for compiling hardware optimised quantum circuits."

setup(
   name='Geodesiq',
   version=VERSION,
   package_dir={'': 'geodesiq'},
   packages=find_packages(where='geodesiq'),
   license='GPLv3',
   author='dyylan',
   author_email='dylan.lewis.19@ucl.ac.uk',
   description=DESCRIPTION,
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown'
)

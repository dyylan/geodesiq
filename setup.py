from setuptools import find_packages, setup


setup(
   name='geodesiq',
   version='0.0.1',
   package_dir={'': 'geodesiq'},
   packages=find_packages(where='geodesiq'),
   license='GPLv3',
   author='dyylan',
   author_email='dylan.lewis.19@ucl.ac.uk',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown'
)

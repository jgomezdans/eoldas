from setuptools import setup, find_packages
import sys, os

DISTNAME = "eoldas"
DESCRIPTION = "An Earth Observation Land Data Assimilation System (EO-LDAS)"
LONG_DESCRIPTION = open('README.txt').read()
MAINTAINER = 'Jose Gomez-Dans/NCEO & University College London'
MAINTAINER_EMAIL = "j.gomez-dans@ucl.ac.uk"
URL = 'http://github.com/jgomezdans/eoldas'
LICENSE = 'Undecided'
VERSION = "1.0.7"
DOWNLOAD_URL="https://github.com/jgomezdans/eoldas/zipball/master"

setup(name='eoldas',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      keywords='',
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url='http://www.assimila.eu/eoldas',
      download_url=DOWNLOAD_URL,
      license='',
      packages=['eoldas'],
      package_dir={'eoldas': 'eoldas'},
      package_data={'eoldas': ['config_files/*.conf']},
      #packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=["SemiDiscrete>1.0.0"],
          # -*- Extra requirements: -*-
          # It require the RT codes, but the package names are still in flux
          #],
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Fortran',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
              ],
      
        scripts=['scripts/eoldas.py'],
      )

from setuptools import setup, find_packages
import sys, os

version = '1.0'

setup(name='eoldas',
      version=version,
      description="An Earth Observation Land Data Assimilation System (EO-LDAS)",
      long_description="""\
""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='',
      author='J Gomez-Dans',
      author_email='j.gomez-dans@ucl.ac.uk',
      url='http://www.assimila.eu/eoldas',
      license='',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )

#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys

setup(name='csfs',
        version='0.1',
        description='Curvilinear Shape Features and CASA axis',
        url='',
        author='Daniel Berio',
        author_email='drand48@gmail.com',
        license='MIT',
        packages=find_packages(),
        install_requires = ['numpy','scipy','matplotlib', 'autograff'],
        zip_safe=False)

"""Setup installation file for Attentive Head Scanning"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='scanr',
    version='0.1.0',
    description='Attentive Head Scanning',
    long_description=long_description,
    url='https://github.com/jdmonaco/scanr',
    author='Joseph Monaco',
    author_email='jmonaco@jhu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4'
    ],
    keywords='model simulation neuroscience',
    packages=['scanr'])

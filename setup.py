#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='arabidopsis-rosette-analysis',
    version='0.3.5',
    description='Extract traits from top-view images of Arabidopsis plants. ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Suxing Liu',
    author_email='suxing.liu@uga.edu',
    license='BSD-3-Clause',
    url='https://github.com/Computational-Plant-Science/arabidopsis-rosette-analysis',
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'arabidopsis = core.cli:cli'
        ]
    },
    python_requires='>=3.6.8',
    install_requires=[
        'click',
        'psutil',
        'numba',
        'pandas',
        'networkx',
        'skan',
        'tabulate',
        'imutils',
        'python-magic',
        'seaborn',
        'openpyxl',
        'opencv-python',
        'matplotlib',
        'scikit-learn',
        'scikit-image',
        'scikit-build',
        'scipy',
        'Pillow'
    ],
    setup_requires=['wheel'],
    tests_require=['pytest', 'coveralls'])

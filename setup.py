#!/usr/bin/env python3
"""Metadata for package to allow installation with pip."""

import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open(os.path.join("echonet", "__version__.py")).read())

setuptools.setup(
    name="echonet",
    description="Interpretable AI for beat-to-beat cardiac function assessment.",
    version=__version__,
    url="https://echonet.github.io/dynamic",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "scikit-image",
        "tqdm",
        "sklearn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)

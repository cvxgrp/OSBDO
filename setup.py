from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="osbdo",
    version="0.0.5",
    packages=["osbdo"],
    license="GPLv3",
    description="Oracle-Structured Bundle Distributed Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy >= 1.22.2",
        "scipy >= 1.8.0",
        "cvxpy >= 1.2.0",
        "matplotlib >= 1.16.0"],
    url="https://github.com/cvxgrp/osbdo",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
)

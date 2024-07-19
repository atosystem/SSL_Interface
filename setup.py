from setuptools import setup, find_packages
import os

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

def read_file(file):
    with open(file) as f:
        return f.read()
    
long_description = read_file("README.md")
version = read_file("VERSION")
requirements = read_requirements("requirements.txt")

setup(
    name = 'SSL_Interface',
    version = version,
    author = 'Yi-Jen Shih',
    author_email = 'yjshih@utexas.edu',
    url = 'https://github.com/atosystem/SSL_Interface',
    description = 'Interface module for Self-supervised Speech Models',
    long_description_content_type = "text/markdown",  # If this causes a warning, upgrade your setuptools package
    long_description = long_description,
    license = "BSD license",
    packages = find_packages(exclude=["test"]),  # Don't include test directory in binary distribution
    install_requires = requirements,
    include_package_data=True,
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ]  # Update these accordingly
)

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyquadfilter",
    version="0.2.0",
    description="Implementation of digital bi-quad filter in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kurene/pyquadfilter",
    author="Kurene@wizard-notes.com",
    packages=["pyquadfilter"],
    install_requires=["scipy", "numpy"],
    classifiers=(
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
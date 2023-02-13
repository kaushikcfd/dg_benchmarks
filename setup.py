#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

ver_dic = {}
version_file_name = "dg_benchmarks/version.py"
with open(version_file_name) as version_file:
    version_file_contents = version_file.read()

exec(compile(version_file_contents, version_file_name, "exec"), ver_dic)

try:
    import mirgecom  # noqa: F401
except ModuleNotFoundError:
    raise RuntimeError("Install MIRGE-Com as:\n"
                       " - git clone https://github.com/illinois-ceesd/emirge\n"
                       " - cd emirge\n"
                       " - ./install.sh")

setup(
    name="dg_benchmarks",
    version=ver_dic["VERSION_TEXT"],
    description="Run DG-FEM Benchmarks.",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires="~=3.8",
    install_requires=["loopy",
                      "pytato",
                      "jaxlib",
                      "tabulate",
                      "pytz",
                      ],
    package_data={"dg_benchmarks": ["py.typed"]},
    author="Kaushik Kulkarni",
    url="https://github.com/kaushikcfd/dg_benchmarks",
    author_email="kgk2@illinois.edu",
    license="MIT",
    packages=find_packages(),
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Betalens - 量化分析与回测框架
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="betalens",
    version="1.0.0",
    author="Janis",
    author_email="",
    description="量化分析与回测框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Janiszzz/betalens",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "psycopg2-binary>=2.9.0",
        "matplotlib>=3.4.0",
        "openpyxl>=3.0.0",
        "prettytable>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # 可以添加命令行工具入口
        ],
    },
    include_package_data=True,
    zip_safe=False,
)


# SPDX-License-Identifier: Apache-2.0
from setuptools import setup, find_packages

setup(
    name="vllm-cautious-decoding",
    version="0.1.0",
    description="Cautious Tree Search Decoding plugin for vLLM",
    author="Carlo Cetrone",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "vllm>=0.6.0",
    ],
    entry_points={
        "vllm.general_plugins": [
            "cautious_decoding = cautious_decoding.plugin:register",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

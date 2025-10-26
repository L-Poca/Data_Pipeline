"""Setup script for Data Pipeline package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-16") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="data-pipeline",
    version="0.1.0",
    author="Rafael Cepa, Cirine, Steven Moire",
    author_email="rafael.cepa@example.fr",
    description="COVID-19 Radiography Analysis Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/L-Poca/Data_Pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "flake8",
            "autopep8",
            "nbqa",
        ],
    },
)

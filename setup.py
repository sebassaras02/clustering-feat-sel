from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="featclus",
    version="0.1.1",
    description="A Python library for feature selection in clustering models",
    long_description=long_description,
    author="Sebastian Sarasti",
    author_email="sebitas.alejo@hotmail.com",
    url="https://github.com/sebassaras02/featclus",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.7",
)

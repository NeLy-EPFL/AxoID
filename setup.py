from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="axoid",
    version="0.1",
    author="Nicolas Talabot",
    author_email="nicolas.talabot@gmail.com",
    description="Region of Interests detection and tracking for 2-photon data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/ROI_detection_and_tracking",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 16.04",
    ],
)
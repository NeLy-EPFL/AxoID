from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="axoid",
    version="0.1",
    author="Nicolas Talabot",
    author_email="nicolas.talabot@gmail.com",
    description="Regions of interest detection and tracking over 2-photon neuroimaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/AxoID",
    packages=["axoid"],
    entry_points={
        "console_scripts": [
            "axoid=axoid.main:main",
            "axoid-gui=axoid.GUI.main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
#        "Operating System :: OS Independent",
    ],
    install_requires=[
        "opencv-python==3.1.0.5",
        "imgaug",
        "numpy",
        "PyQt5",
        "scikit-image",
        "scikit-learn",
        "torch",
        "torchvision",
    ],
)

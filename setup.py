#!/usr/bin/env python

import setuptools
import platform

if (
    platform.processor() == "arm" or platform.processor() == "i386"
) and platform.system() == "Darwin":
    tensorflow_os = "tensorflow>=2.14.0,<=2.15.1"
else:
    tensorflow_os = "tensorflow>=2.12.0,<=2.15.1"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="decimer_segmentation",
    version="1.6.0",
    author="Kohulan Rajan",
    author_email="kohulan.rajan@uni-jena.de",
    maintainer="Kohulan Rajan",
    maintainer_email="kohulan.rajan@uni-jena.de",
    description="DECIMER Segmentation - Extraction of chemical structure depictions from scientific literature",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kohulan/DECIMER-Image-Segmentation",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=[
        tensorflow_os,
        "numpy>=1.2.0",
        "scikit-image>=0.2.0",
        "pillow",
        "opencv-python",
        "matplotlib",
        "IPython",
        "numba",
        "scipy",
        "requests",
        "pystow",
        # Reads PDF files (Apache-2.0 / BSD-3-Clause); 4.30 to 5.x are tested.
        "pypdfium2>=4.30.0,<6",
    ],
    extras_require={
        # PyMuPDF, an alternative PDF reader (pdf_backend="pymupdf"). It is licensed
        # under the AGPL-3.0, so it is not installed unless asked for:
        # pip install "decimer-segmentation[pdf]"
        "pdf": ["PyMuPDF"],
    },
    package_data={"decimer_segmentation": ["mrcnn/*.*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

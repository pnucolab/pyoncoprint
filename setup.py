import io
import setuptools

with io.open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyoncoprint",
    version="0.2.5",
    author="Jeongbin Park",
    author_email="jeongbin.park@pusan.ac.kr",
    description="PyOncoPrint",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/pjb7687/pyoncoprint",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
    ]
)

from setuptools import find_packages, setup

from numba_celltree.aot_compile import cc

with open("README.md") as f:
    long_description = f.read()

setup(
    name="numba_celltree",
    description="Cell Tree Spatial Index",
    long_description=long_description,
    url="https://github.com/huite/numba_celltree",
    author="Huite Bootsma",
    author_email="huite.bootsma@deltares.nl",
    license="MIT",
    packages=find_packages(),
    package_dir={"numba_celltree": "numba_celltree"},
    ext_modules=[cc.distutils_extension()],  # AOT compilation
    test_suite="tests",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    python_requires=">=3.6",
    install_requires=["numpy", "numba>=0.50"],
    extras_require={
        "dev": [
            "black",
            "pytest",
            "pytest-cov",
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords="spatial index unstructured grid mesh",
)

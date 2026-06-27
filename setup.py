from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='compact',
    packages=find_packages(),

    ext_modules=cythonize(
    "compact/velocities/pairvel.pyx",
    compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()]
)
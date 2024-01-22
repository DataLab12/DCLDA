from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include



setup(
    ext_modules=cythonize("*.pyx", quiet=False, language_level='3', annotate=True),
    include_dirs=[get_include()],
    zip_safe=False,
)
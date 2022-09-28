from distutils.core import setup
from Cython.Build import cythonize

setup(
  ext_modules = cythonize(["keras_model.py","record_module.py","serial_module.py","test_module.py","train_module.py","validation_module.py"]),
)
from distutils.core import setup, Extension
import sysconfig
import os
import numpy

def main():
    extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
    extra_compile_args += ["-std=c++17", "-Wall", "-Wextra", "-Wno-missing-field-initializers"]
    setup(name="noise",
          version="1.0.0",
          description="Python binding for the FastNoiseLite library",
          author="Luca Fonstad",
          author_email="luca_fonstad@brown.edu",
          ext_modules=[Extension("noise", ["main.cpp"], include_dirs=[numpy.get_include(), os.path.join(os.getcwd(),"FastNoiseLite","Cpp")], extra_compile_args=extra_compile_args, language="c++17")])

if __name__ == "__main__":
    main()

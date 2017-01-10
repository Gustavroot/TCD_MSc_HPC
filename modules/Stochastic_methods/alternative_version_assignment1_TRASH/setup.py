from distutils.core import setup, Extension

# define the extension module
module1 = Extension('pyranlux', libraries = ['ranlxd'], sources=['pyranlux.c'])

# run the setup
setup(ext_modules=[module1])

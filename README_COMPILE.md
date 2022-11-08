# Compiling PYJAX

in `plumed2` directory:


```
pip install find_libpython pybind11


./configure  LDFLAGS="-Wl,-rpath,`python3-config --prefix`/lib `find_libpython`"

make -j 10
# add plumed in the path
source sourceme.sh
# make sure the libraries for python packages are visible
export LD_LIBRARY_PATH="`python3-config --prefix`/lib/:$LD_LIBRARY_PATH"
```

to test the installation

```
regtest/pyjax
make
```
USE=core tools bias colvar function
# include pybind11 from pip/conda
ADDCPPFLAGS += -I$(shell python3-config --prefix)/include
# include python headers
ADDCPPFLAGS += $(shell python3-config --includes)

include ../maketools/make.module



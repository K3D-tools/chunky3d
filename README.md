# chunky3d
A 3D array-like NumPy-based data structure for large sparsely-populated volumes

## Build
[![Build Status](https://travis-ci.org/K3D-tools/chunky3d.svg?branch=master)](https://travis-ci.org/K3D-tools/chunky3d)

# Introduction

This library provides a data structure, `Sparse`, which represents 3D volumetric data
and supports a subset of `np.ndarray` features.

## Example

```
>>> import numpy as np
>>> from chunky3d import Sparse

>>> s = Sparse(shape=(64, 64, 64))
>>> s[0, 0, 0]
0

>>> s.dtype
numpy.float64

>>> s.nchunks
8

>>> s.nchunks_initialized
0

>>> s[1, 2, 3] = 3
>>> s.nchunks_initialized
1

>>> s[:2, 2, 3:5]
array([[0., 0.],
       [3., 0.]])
```

# Features

* `chunky3d.sparse_func` - a collection of functions for analyzing chunked arrays, including 
  morphological operations (opening, closing), thinning, connected components
* Fast load and save using `msgpack`
* Operations on arrays using `.run()`, with possible acceleration using `multiprocessing`
* Accelerated lookup using `numba`
* Interpolation (point probe)
* Origin and spacing: representing 3D space with non-uniform spacing for different axes

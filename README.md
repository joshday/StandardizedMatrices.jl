# StandardizedMatrices

[![Build Status](https://travis-ci.org/joshday/StandardizedMatrices.jl.svg?branch=master)](https://travis-ci.org/joshday/StandardizedMatrices.jl)


Statisticians often work with standardized matrices.  If `x` is an `n×p` data matrix with observations in rows, we want to work with `z = StatsBase.zscore(x, 1)`.  In cases where `x` is large, it may be undesirable to copy the data into a standardized matrix or change the underlying data (`StatsBase.zscore!(x, 1)`).


This package defines a `StandardizedMatrix` type that treats a matrix as standardized without copying or changing data in place.

# A Motivating Example

Suppose we wish to do matrix-vector multiplication with a standardized matrix where the original matrix is sparse.  Typically, standardizing a sparse matrix destroys the sparsity.

```julia
using StatsBase, StandardizedMatrices

# generate some data
n, p = 100_000, 1000
x = sprandn(n, p, .01)
β = randn(p)

xdense = zscore(x, 1)		# this destroys the sparsity
z = StandardizedMatrix(x)	# this acts as standardized, but keeps sparse benefits

@time xdense * β;
@time z * β;  # Almost 100 times speedup on my machine
```

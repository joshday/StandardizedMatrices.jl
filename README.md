[![CI](https://github.com/joshday/StandardizedMatrices.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/joshday/StandardizedMatrices.jl/actions/workflows/CI.yml)
[![Docs Build](https://github.com/joshday/StandardizedMatrices.jl/actions/workflows/Docs.yml/badge.svg)](https://github.com/joshday/StandardizedMatrices.jl/actions/workflows/Docs.yml)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue)](https://joshday.github.io/StandardizedMatrices.jl/stable/)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue)](https://joshday.github.io/StandardizedMatrices.jl/dev/)

# StandardizedMatrices




Statisticians often work with standardized matrices.  If `x` is a data matrix with observations in rows, we want to work with `z = StatsBase.zscore(x, 1)`.  This package defines a `StandardizedMatrix` type that treats a matrix as standardized without copying or changing data in place.

# A Motivating Example

Suppose our original matrix is sparse and we want to perform matrix-vector multiplication with a standardized version.  Typically, standardizing a sparse matrix destroys the sparsity.

```julia
using StatsBase, BenchmarkTools, StandardizedMatrices, SparseArrays, Statistics

# generate some data
n, p = 100_000, 1000
x = sprandn(n, p, .01)
β = randn(p)

xdense = zscore(x, 1)		# this destroys the sparsity
z = StandardizedMatrix(x)	# this acts as standardized, but keeps sparse benefits

b1 = @benchmark xdense * β
b2 = @benchmark z * β
ratio(median(b1), median(b2))  # StandardizedMatrix is roughly 13 times faster
```


# Methods implemented:

- `*()`
- `mul!(Y, A::StandardizedMatrix, B)`
- `mul!(Y, A::Adjoint{<:StandardizedMatrix}, B)`

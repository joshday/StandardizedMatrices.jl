module StandardizedMatrices
export StandardizedMatrix

#-----------------------------------------------------------------------------# types
typealias AVec{T} AbstractVector{T}
typealias AMat{T} AbstractMatrix{T}
typealias VecF Vector{Float64}

#----------------------------------------------------------------# StandardizedMatrix
"""
Treat a matrix as standardized, similar to `z = StatsBase.zscore(x, 1)`, without
altering the original data.

`z = StandardizedMatrix(x)`
"""
immutable StandardizedMatrix{T <: AbstractFloat, S <: AMat} <: AMat{T}
	data::S
	μ::VecF			# column means
	σinv::VecF		# inverse of column stdevs
	function StandardizedMatrix(x::AMat, μ::AVec, σ::AVec)
		n, p = size(x)
		μp, σp = length(μ), length(σ)
		@assert p == μp "size(x, 2) == $p doesn't match length(μ) == $μp"
		@assert p == σp "size(x, 2) == $p doesn't match length(σ) == $σp"
		new(x, μ, 1 ./ σ)
	end
end
# This acts as inner constructor.  All constructors call this.  See:
# http://docs.julialang.org/en/latest/manual/constructors/#parametric-constructors
function StandardizedMatrix(x::AMat, μ::AVec, σ::AVec)
	T = eltype((x[1] - μ[1]) / σ[1])
	StandardizedMatrix{T, typeof(x)}(x, μ, σ)
end
StandardizedMatrix(x::AMat) = StandardizedMatrix(x, vec(mean(x, 1)), vec(std(x, 1)))



#----------------------------------------------------------------------# Base methods
# http://docs.julialang.org/en/release-0.4/manual/interfaces/#man-interfaces-abstractarray
Base.size(o::StandardizedMatrix) 				= size(o.data)
Base.linearindexing(o::StandardizedMatrix)		= Base.linearindexing(o.data)
function Base.getindex(o::StandardizedMatrix, i::Int, j::Int)
	v = getindex(o.data, i, j)
	return (v - o.μ[j]) * o.σinv[j]
end
function Base.getindex(o::StandardizedMatrix, i::Int)
	v = getindex(o.data, i)
	j = floor(Int, i / size(o, 1))
	return (v - o.μ[j]) * o.σinv[j]
end

# Recalculate μ and σ (for when underlying data is altered)
# function recalc!(o::StandardizedMatrix)
# 	o.μ[:] = vec(mean(o.data, 1))
# 	o.σinv[:] = 1.0 ./ vec(std(o.data, 1))
# 	o
# end
# function Base.setindex!(o::StandardizedMatrix, ind...)
# 	setindex!(o.data, ind...)
# 	recalc!(o)
# end



# Matrix-Vector multiplication
function Base.A_mul_B!(y::AVec, A::StandardizedMatrix, b::AVec)
	A_mul_B!(y, A.data, Diagonal(A.σinv) * b)
	m = mean(y)
	for i in eachindex(y)
		@inbounds y[i] -= m
	end
end
function Base.At_mul_B!{T <: Real}(y::AVec{T}, A::StandardizedMatrix, b::AVec{T})
	At_mul_B!(y, A.data, b - mean(b))
	for i in eachindex(y)
		@inbounds y[i] *= A.σinv[i]
	end
end
function Base.(:*){T <: Real}(A::StandardizedMatrix, b::AVec{T})
	y = zeros(T, size(A, 1))
	A_mul_B!(y, A, b)
	y
end


end

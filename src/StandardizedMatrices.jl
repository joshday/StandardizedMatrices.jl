module StandardizedMatrices

import LearnBase.ObsDim: ObsDimension, First, Last
export ZMatrix

#-----------------------------------------------------------------------# types
const AVec{T} 	= AbstractVector{T}
const AMat{T} 	= AbstractMatrix{T}
const VecF 		= Vector{Float64}

#-----------------------------------------------------------------------# ZMatrix
immutable ZMatrix{
		T <: Number,
		S <: AMat,
		M <: AVec,
		V <: AVec,
		D <: ObsDimension} <: AMat{T}
	data::S
	μ::M		# column means
	σinv::V		# inverse of column stdevs
	dim::D
end
function ZMatrix(x::AMat, μ::AVec, σ::AVec, dim::ObsDimension = First())
	T = eltype((x[1] - μ[1]) / σ[1])
	if !dimsmatch(x, μ, σ, dim)
		throw(DimensionMismatch("x: $(size(x)), μ: $(length(μ)), σ: $(length(σ)), dim: $dim"))
	end
	σinv = inv.(σ)
	ZMatrix{T, typeof(x), typeof(μ), typeof(σinv), typeof(dim)}(x, μ, σinv, dim)
end
dimsmatch(x, μ, σ, dim::First) = (length(μ) == length(σ) == size(x, 2))
dimsmatch(x, μ, σ, dim::Last)  = (length(μ) == length(σ) == size(x, 1))

ZMatrix(x::AMat) = ZMatrix(x, vec(mean(x, 1)), vec(std(x, 1)))

#-----------------------------------------------------------------------# AbstractArray Interface
# https://docs.julialang.org/en/latest/manual/interfaces/#man-interface-array-1
Base.size(o::ZMatrix) = size(o.data)
Base.IndexStyle(o::ZMatrix)	= IndexCartesian() #IndexStyle(o.data)

variable_ind(::First, i, j) = j
variable_ind(::Last, i, j) = i

function Base.getindex(o::ZMatrix, i::Int, j::Int)
	v = getindex(o.data, i, j)
	k = variable_ind(o.dim, i, j)
	return (v - o.μ[k]) * o.σinv[k]
end

function Base.setindex(o::ZMatrix, value, i::Int, j::Int)
	k = variable_ind(o.dim, i, j)
	o.data[i, j] = inv(o.σinv[k]) * value + o.μ[k]
end





#-----------------------------------------------------------------------# TODO:
function Base.:*{T <: Real}(A::ZMatrix, B::AVec{T})
	y = zeros(typeof(A[1] * B[1]), size(A, 1))
	A_mul_B!(y, A, B)
	y
end
function Base.:*{T <: Real}(A::ZMatrix, B::AMat{T})
	y = zeros(typeof(A[1] * B[1]), size(A, 1), size(B, 2))
	A_mul_B!(y, A, B)
	y
end

#------------------------------------------------------# Matrix-Vector multiplication
function Base.A_mul_B!(y::AVec, A::ZMatrix, b::AVec)
	A_mul_B!(y, A.data, Diagonal(A.σinv) * b)
	center!(y)
end
function Base.At_mul_B!(y::AVec, A::ZMatrix, b::AVec)
	At_mul_B!(y, A.data, b - mean(b))
	_scale!(y, A.σinv)
end

#------------------------------------------------------# Matrix-Matrix multiplication
function Base.A_mul_B!(y::AMat, A::ZMatrix, b::AMat)
	A_mul_B!(y, A.data, Diagonal(A.σinv) * b)
	center!(y)
end
function Base.At_mul_B!(y::AMat, A::ZMatrix, b::AMat)
	At_mul_B!(y, A.data, b .- mean(b, 1))
	_scale!(y, A.σinv)
end

#---------------------------------------------------------------------------# helpers
function center!(v::Vector)
	μ = mean(v)
	for i in eachindex(v)
		@inbounds v[i] -= μ
	end
	v
end
function center!(m::Matrix)
	μ = mean(m, 1)
	@inbounds for j in 1:size(m, 2)
		μj = μ[j]
		for i in 1:size(m, 1)
			m[i, j] -= μj
		end
	end
	m
end
function _scale!(a::Vector, b::Vector)
	for i in eachindex(a)
		@inbounds a[i] *= b[i]
	end
	a
end
function _scale!(a::Matrix, b::Vector)
	for j in 1:size(a, 2)
		for i in 1:size(a, 1)
			a[i, j] *= b[i]
		end
	end
	a
end

end #module

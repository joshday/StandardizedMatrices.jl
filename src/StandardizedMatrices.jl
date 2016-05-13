module StandardizedMatrices
export StandardizedMatrix

#-----------------------------------------------------------------------------# types
typealias AVec{T} AbstractVector{T}

#----------------------------------------------------------------# StandardizedMatrix
"""
`StandardizedMatrix(x, μ = mean(x, 1), σ = std(x, 1))`
"""
immutable StandardizedMatrix{T, S <: AbstractMatrix} <: AbstractMatrix{T}
	data::S
	μ::Vector{Float64}
	σinv::Vector{Float64}
end
function StandardizedMatrix(x::AbstractMatrix, μ = mean(x, 1), σ = std(x, 1))
	@assert size(x, 2) == length(μ) == length(σ) "Incompatible dimensions"
	StandardizedMatrix{eltype(x), typeof(x)}(x, vec(μ), 1 ./ vec(σ))
end


#----------------------------------------------------------------------# Base methods
function Base.show(io::IO, o::StandardizedMatrix)
	println(io, replace("$(typeof(o)) with data:", "StandardizedMatrices.", ""))
	show(io, o.data)
end
Base.size(o::StandardizedMatrix, args...) 		= size(o.data, args...)
Base.getindex(o::StandardizedMatrix, args...) 	= getindex(o.data, args...)
Base.setindex!(o::StandardizedMatrix, args...) 	= setindex!(o.data, args...)
Base.length(o::StandardizedMatrix) 				= length(o.data)


# Matrix-Vector multiplication
function Base.A_mul_B!(y::AVec, A::StandardizedMatrix, b::AVec)
	A_mul_B!(y, A.data, b .* A.σinv)
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

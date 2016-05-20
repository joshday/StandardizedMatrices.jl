module Tests
using StandardizedMatrices, StatsBase
if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

n, p = 100, 5
x = randn(n, p)
x2 = zscore(x, 1)
z = StandardizedMatrix(x)
b = randn(p)
b2 = randn(n)

@testset "Multiplication with Vector" begin
	@test isapprox(z * b, x2 * b)
	storage = zeros(n)
	A_mul_B!(storage, z, b)
	@test isapprox(storage, x2 * b)
	storage = zeros(p)
	At_mul_B!(storage, z, b2)
	@test isapprox(storage, x2' * b2)
end

@testset "Indexing" begin
	for i in eachindex(z)
		@test z[i] == x2[i]
	end
end


end #module

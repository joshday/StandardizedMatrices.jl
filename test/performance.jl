module Performance

using StandardizedMatrices, BenchmarkTools, StatsBase

n, p = 100_000, 1000
x = randn(n, p)
x2 = zscore(x, 1)
z = StandardizedMatrix(x)
β = randn(p)
@show isapprox(x2 * β, z * β)


info("A_mul_B! (dense)")
b1 = @benchmark(
	A_mul_B!(storage, x, β),
	setup = (storage = zeros(n); x = StandardizedMatrix(randn(n, p)); β = randn(p))
)
b2 = @benchmark(
	A_mul_B!(storage, x, β),
	setup = (storage = zeros(n); x = randn(n, p); β = randn(p))
)
@show ratio(minimum(b1), minimum(b2))


info("A_mul_B! (sparse)")
b1 = @benchmark(
	A_mul_B!(storage, x, β),
	setup = (storage = zeros(n); x = StandardizedMatrix(sprandn(n, p, .05)); β = randn(p))
)
b2 = @benchmark(
	A_mul_B!(storage, x, β),
	setup = (storage = zeros(n); x = zscore(sprandn(n, p, .05), 1); β = randn(p))
)
@show ratio(minimum(b1), minimum(b2))


end

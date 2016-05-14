using StandardizedMatrices
using FactCheck, StatsBase


facts("Matrix-Vector multiplication") do
	x = randn(100,100)
	x2 = zscore(x, 1)
	z = StandardizedMatrix(x)
	b = randn(100)

	@fact x2 * b --> roughly(z * b)
end

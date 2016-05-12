module Performance

using StandardizedMatrices, BenchmarkTools, StatsBase

n, p = 100_000, 1000


info("A_mul_B! (dense)")
x = randn(n, p)
y = randn(p)
z = StandardizedMatrix(x)
storage = zeros(n)
b1 = @benchmark A_mul_B!(storage, z, y)
storage = zeros(n)
b2 = @benchmark A_mul_B!(storage, x, y)
display(ratio(minimum(b1), minimum(b2)))

info("A_mul_B! (sparse)")
x = sprandn(n, p, .01)
x2 = zscore(x, 1)
y = randn(p)
z = StandardizedMatrix(x)
storage = zeros(n)
b1 = @benchmark A_mul_B!(storage, z, y)
storage = zeros(n)
b2 = @benchmark A_mul_B!(storage, x2, y)
display(ratio(minimum(b1), minimum(b2)))

# info("At_mul_B! (dense)")
# x = randn(p, n)
# y = randn(p)
# z = StandardizedMatrix(x)
# storage = zeros(n)
# b1 = @benchmark At_mul_B!(storage, z, y)
# storage = zeros(n)
# b2 = @benchmark At_mul_B!(storage, x, y)
# display(ratio(minimum(b1), minimum(b2)))
#
# info("At_mul_B! (sparse)")
# x = sprandn(p, n, .01)
# y = randn(p)
# z = StandardizedMatrix(x)
# storage = zeros(n)
# b1 = @benchmark At_mul_B!(storage, z, y)
# storage = zeros(n)
# b2 = @benchmark At_mul_B!(storage, x, y)
# display(ratio(minimum(b1), minimum(b2)))


end

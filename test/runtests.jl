using Test
using Polynomials
using PolynomialsFFT

@testset "qwen" begin

    n = 4
    x = [1.0, 1, 0, 2]
    y = [4.0, - 1.0, -2.0, 1.0]

    ref = Polynomial(x) * Polynomial(y)

    res = multiply_polynomials(x, y)

    @test ref == Polynomial(res)


end

@testset "deepseek" begin

    n = 4
    x = [1.0, 1, 0, 2]
    y = [4.0, - 1.0, -2.0, 1.0]

    ref = Polynomial(x) * Polynomial(y)

    res = multiply_polynomials_deepseek(x, y)

    @test ref == Polynomial(res)


end


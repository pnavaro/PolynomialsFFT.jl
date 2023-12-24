using Test
using PolynomialsFFT

@testset "fft" begin

    n = 4
    x = [1, 1, 0, 2]
    y = PolynomialsFFT.fft(n, x)

    @test y ≈ [4.0, 1. - 1.0im, -2.0, 1.0 + 1.0im]
    v = PolynomialsFFT.ifft(4, y)
    @test v ≈ x

end


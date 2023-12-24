# -*- coding: utf-8 -*-
"""
Calculate the fast Fourier transform of n numbers in x̄.
(Note that n must be a power of 2 (n = 2ᵏ))
Returns:
ȳ - A complex array of size n 
"""
function fft(n::Integer, x̄::Array{<:Number})
    # Julia is 1-indexed
    if n == 1
        return [x̄[1]]
    end

    evens = [x̄[Int(2i)] for i = 1:n/2]
    odds = [x̄[Int(2i-1)] for i = 1:n/2]
    # Since Julia is 1-indexed, we flip the odds and evens at the recursive step
    ū = FFT(Int(n/2), odds)
    v̄ = FFT(Int(n/2), evens)
    ȳ = zeros(Complex, n)
    for j = 1:n
        τ = exp(2π*im*(j-1)/n) 
        ȳ[j] = ū[(j-1)%Int(n/2)+1] + τ * v̄[(j-1)%Int(n/2)+1]
    end
    ȳ
end

n = 4
x̄ = [1 1 0im 2]
ȳ = FFT(n, x̄)
display(ȳ)

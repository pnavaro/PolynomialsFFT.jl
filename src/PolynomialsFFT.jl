module PolynomialsFFT

"""
    fft(n, x̄)

Calculate the fast Fourier transform of n numbers in x̄.

(Note that n must be a power of 2 (n = 2ᵏ))

Returns:

- ȳ : A complex array of size n 
"""
function fft(n::Integer, x̄)

    @assert ispow2(n)

    n == 1 && return [x̄[1]]

    evens = [x̄[2i] for i = 1:n÷2]
    odds = [x̄[2i-1] for i = 1:n÷2]
    
    # Since Julia is 1-indexed, we flip the odds and evens at the recursive step
    #
    ū = fft(n÷2, odds)
    v̄ = fft(n÷2, evens)
    ȳ = zeros(Complex, n)

    for j = 1:n
        τ = exp(2π * 1im * (j-1) / n) 
        ȳ[j] = ū[(j-1)%(n÷2)+1] + τ * v̄[(j-1)%(n÷2)+1]
    end

    ȳ

end


"""
    ifft(n, y)

Compute the inverse Fourier transform
"""
function ifft(n::Int, y)
    
    @assert ispow2(n)

    n == 1 && return [y[1]]

    evens = [y[2i] for i = 1:n÷2]
    odds = [y[2i-1] for i = 1:n÷2]
    
    ū = fft(n÷2, odds)
    v̄ = fft(n÷2, evens)
    x = similar(y)
    for j = 1:n
        τ = exp(2π * im * (j-1)/n) 
        x[j] = ū[(j-1) % (n÷2)+1] + v̄[(j-1)% (n÷2)+1] / τ
    end
    x ./ n
end

end 

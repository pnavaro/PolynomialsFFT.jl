using Test
using FFTPolynomialProduct

@testset "fft" begin
n = 4
x = [1, 1, 0, 2]
y = fft(n, x)
@test  y ≈ 
end

# + [markdown] slideshow={"slide_type": "slide"}
# # Time Complexity
# Let $N = n$, and $T(N) = F(\mathbf{a}, w_n[j])$,
# \begin{align}
# T(N) = 2 \cdot T(\frac{N}{2}) + N,
# \end{align}
# where the $+N$ comes from the loop after the recursive step.
#
# This algorithm runs in $\mathcal{O}(N \lg N)$ when $N$ is a power of 2.

# + [markdown] slideshow={"slide_type": "subslide"}
# *Proof:*
# \begin{align}
# T(N) & = 2 \cdot T(\frac{N}{2}) + N \\
# \end{align}
# Let $N = 2^k$, and $t_k = T(2^k)$
# \begin{align}
# t_k & = 2t_{k-1} + 2^k \\
# t_k - 2t_{k-1} & = 2^k \\
# \end{align}

# + [markdown] slideshow={"slide_type": "subslide"}
# We can use the characteristic equation to solve this inhomogeneous system.
# \begin{align}
# (x - 2)^2 & = 0 \\
# \end{align}
# So we end up with
# \begin{align}
# t_k & = c_1 2^k + c_2 k 2^k; k = \lg N\\
# T(N) & = c_1 N + c_2 N \lg N \\
# T(N) & = \mathcal{O}(N \lg N) \\
# \end{align}

# + [markdown] slideshow={"slide_type": "slide"}
# # Example
# Say you want to perform the following multiplication:
# \begin{align}
# (1 + x)(1 + x + x^2).
# \end{align}
# We can use the FFT algorithm to find the point-value representation of each polynomial, then multiply those two together:

# + [markdown] slideshow={"slide_type": "fragment"}
# Calculating the FFT of each,
# \begin{align}
# FFT(1 + x) & = [2, 1+i, 0, 1-i] \\
# FFT(1 + x + x^2) & = [3, i, 1, -i], \\
# \end{align}
# which leads to the point-value representation of their product: $[6, -1 + i, 0, -1-i]$.

# + slideshow={"slide_type": "subslide"}
n = 4
x̄₁ = fft(4, [1,1,0,0])

# + slideshow={"slide_type": "subslide"}
ȳ₁ = FFT(4, [1,1,1,0])
# -

x̄₁ .* ȳ₁

function ifft(x, n::Int, y)
    
    n == 1 && return [y[1]]

    evens = [y[2i] for i = 1:n÷2]
    odds = [y[2i-1] for i = 1:n÷2]
    
    ū = fft(n÷2, odds)
    v̄ = fft(n÷2, evens)
    x = zeros(Complex, n)
    for j = 1:n
        τ = exp(2π * im * (j-1)/n) 
        x[j] = ū[(j-1) % (n÷2)+1] + v̄[(j-1)% (n÷2)+1] / τ
    end
    x ./ n
end

v = ifft(4, [6,−1+1im,0,−1−1im])

Polynomial([1,1,0,0]) * Polynomial([1,1,1,0])

Polynomial(round.(v))



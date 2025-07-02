using CUDA
using CUDA.FFT

function multiply_polynomials_fft_gpu(a::Vector{Float32}, b::Vector{Float32})
    n = length(a) + length(b) - 1
    m = nextpow(2, n)

    # Padding et transfert sur le GPU
    a_gpu = CUDA.zeros(ComplexF32, m)
    b_gpu = CUDA.zeros(ComplexF32, m)
    a_gpu[1:length(a)] .= ComplexF32.(a, 0)
    b_gpu[1:length(b)] .= ComplexF32.(b, 0)

    # FFT sur GPU
    fa_gpu = cufft(a_gpu)
    fb_gpu = cufft(b_gpu)

    # Multiplication point à point sur GPU
    fc_gpu = fa_gpu .* fb_gpu

    # IFFT
    c_gpu = cufft_inverse(fc_gpu)

    # Récupération sur CPU
    c = collect(real.(c_gpu[1:n]))
    return c
end

# Exemple :
a = Float32[1, 2, 3]  # 1 + 2x + 3x^2
b = Float32[4, 5]     # 4 + 5x
c = multiply_polynomials_fft_gpu(a, b)
println(c)


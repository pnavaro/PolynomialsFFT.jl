using FFTW
using LinearAlgebra

export multiply_polynomials_deepseek

function multiply_polynomials_deepseek(p::Vector{T}, q::Vector{T}) where {T<:Number}
    n = nextpow(2, length(p) + length(q) - 1)
    p_pad = zeros(Complex{T}, n)
    q_pad = zeros(Complex{T}, n)
    
    @inbounds @simd for i in eachindex(p)
        p_pad[i] = p[i]
    end
    
    @inbounds @simd for i in eachindex(q)
        q_pad[i] = q[i]
    end
    
    fft_p = fft(p_pad)
    fft_q = fft(q_pad)
    
    result = ifft(fft_p .* fft_q) 
    
    return round.(real(result))

end

#=
using CUDA, CUDA.CUFFT, KernelAbstractions

function multiply_polynomials_gpu(p::CuArray{T}, q::CuArray{T}) where {T <: Real}
    len_p = length(p)
    len_q = length(q)
    n = nextpow(2, len_p + len_q - 1)

    # Allouer et remplir les tableaux avec zero-padding
    p_pad = CUDA.zeros(Complex{T}, n)
    q_pad = CUDA.zeros(Complex{T}, n)

    # Copier les données (asynchrone)
    copyto!(p_pad, 1, p, 1, len_p)
    copyto!(q_pad, 1, q, 1, len_q)

    # FFT sur GPU
    fft_p = CUFFT.fft!(p_pad)
    fft_q = CUFFT.fft!(q_pad)

    # Multiplication pointwise (kernel optimisé)
    CUDA.@sync begin
        @kernel function pointwise_mul!(out, a, b)
            i = @index(Global)
            @inbounds out[i] = a[i] * b[i]
        end

        kernel = pointwise_mul!(CUDA.CUDADevice(), 256)  # 256 threads par bloc
        kernel(p_pad, fft_p, fft_q, ndrange=n)
    end

    # IFFT et extraction de la partie réelle
    result = CUFFT.ifft!(p_pad)
    real_part = real.(result)

    return real_part[1:(len_p + len_q - 1)]  # Tronquer aux coefficients non nuls
end

# Exemple d'utilisation
using CUDA
p = CuArray(Float32[1, 2, 3])  # Préférer Float32 sur GPU
q = CuArray(Float32[4, 5])
@time multiply_polynomials_gpu(p, q)  # ≈ [4.0f0, 13.0f0, 22.0f0, 15.0f0]

=#

using FFTW

# Version optimisée avec pré-allocation et plans FFTW
function poly_mult_fft_optimized!(result::Vector{T}, p1::Vector{T}, p2::Vector{T},
                                  work1::Vector{Complex{T}}, work2::Vector{Complex{T}},
                                  fft_plan, ifft_plan) where T <: Real
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = length(work1)

    # Copie avec padding
    work1[1:n1] .= complex.(p1)
    work1[n1+1:end] .= 0
    work2[1:n2] .= complex.(p2)
    work2[n2+1:end] .= 0

    # FFT en place
    mul!(work1, fft_plan, work1)
    mul!(work2, fft_plan, work2)

    # Multiplication et IFFT
    work1 .*= work2
    mul!(work1, ifft_plan, work1)

    # Extraction résultat
    for i in 1:n
        result[i] = real(work1[i])
    end

    return result
end

# Factory pour créer les plans et buffers
function create_poly_mult_context(max_size::Int)
    fft_size = nextpow(2, 2 * max_size - 1)
    work1 = Vector{ComplexF64}(undef, fft_size)
    work2 = Vector{ComplexF64}(undef, fft_size)
    fft_plan = plan_fft!(work1)
    ifft_plan = plan_ifft!(work1)

    return (work1, work2, fft_plan, ifft_plan)
end

export poly_mult_fft_fast

# Version simple optimisée sans pré-allocation
function poly_mult_fft_fast(p1::Vector{T}, p2::Vector{T}) where T <: Real
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = nextpow(2, n)

    # Allocation directe à la bonne taille
    buf1 = zeros(Complex{T}, fft_size)
    buf2 = zeros(Complex{T}, fft_size)

    buf1[1:n1] .= p1
    buf2[1:n2] .= p2

    fft_buf1 = fft(buf1)
    fft_buf2 = fft(buf2)
    fft_buf1 .*= fft_buf2

    buf1 .=  ifft(fft_buf1)

    return round.(real(buf1[1:n]))
end
#=

# Version GPU avec CuArrays
function poly_mult_fft_gpu(p1::Vector{T}, p2::Vector{T}) where T <: Real
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = nextpow(2, n)

    # Transfer vers GPU avec padding
    gpu_p1 = CUDA.zeros(Complex{T}, fft_size)
    gpu_p2 = CUDA.zeros(Complex{T}, fft_size)

    gpu_p1[1:n1] .= CuArray(complex.(p1))
    gpu_p2[1:n2] .= CuArray(complex.(p2))

    # FFT sur GPU
    fft!(gpu_p1)
    fft!(gpu_p2)

    # Multiplication
    gpu_p1 .*= gpu_p2

    # IFFT
    ifft!(gpu_p1)

    # Retour CPU
    result = Array(real.(gpu_p1[1:n]))
    return result
end

#
# Version GPU optimisée avec plans pré-compilés
function poly_mult_fft_gpu_planned!(result::Vector{T}, p1::Vector{T}, p2::Vector{T},
                                   gpu_buf1::CuArray{Complex{T}}, gpu_buf2::CuArray{Complex{T}},
                                   fft_plan, ifft_plan) where T <: Real
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = length(gpu_buf1)

    # Upload avec padding
    gpu_buf1[1:n1] .= CuArray(complex.(p1))
    gpu_buf1[n1+1:end] .= 0
    gpu_buf2[1:n2] .= CuArray(complex.(p2))
    gpu_buf2[n2+1:end] .= 0

    # FFT planifiées
    mul!(gpu_buf1, fft_plan, gpu_buf1)
    mul!(gpu_buf2, fft_plan, gpu_buf2)

    # Multiplication et IFFT
    gpu_buf1 .*= gpu_buf2
    mul!(gpu_buf1, ifft_plan, gpu_buf1)

    # Download résultat
    copyto!(result, real.(Array(gpu_buf1[1:n])))
    return result
end

# Factory pour contexte GPU
function create_gpu_poly_mult_context(max_size::Int, ::Type{T}=Float64) where T <: Real
    fft_size = nextpow(2, 2 * max_size - 1)
    gpu_buf1 = CUDA.zeros(Complex{T}, fft_size)
    gpu_buf2 = CUDA.zeros(Complex{T}, fft_size)

    fft_plan = plan_fft!(gpu_buf1)
    ifft_plan = plan_ifft!(gpu_buf1)

    return (gpu_buf1, gpu_buf2, fft_plan, ifft_plan)
end

# Version GPU avec rfft pour réels
function poly_mult_rfft_gpu(p1::Vector{T}, p2::Vector{T}) where T <: Real
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = nextpow(2, n)

    # Upload avec padding
    gpu_p1 = CuArray([p1; zeros(T, fft_size - n1)])
    gpu_p2 = CuArray([p2; zeros(T, fft_size - n2)])

    # rFFT
    f1 = rfft(gpu_p1)
    f2 = rfft(gpu_p2)

    # Multiplication et iFFT
    f1 .*= f2
    result_gpu = irfft(f1, fft_size)
end

# Version optimisée Apple Silicon (M1/M2/M3/M4)
function poly_mult_fft_metal(p1::Vector{T}, p2::Vector{T}) where T <: Real
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = nextpow(2, n)

    # Transfer vers Metal GPU
    metal_p1 = MtlArray(zeros(Complex{T}, fft_size))
    metal_p2 = MtlArray(zeros(Complex{T}, fft_size))

    metal_p1[1:n1] .= MtlArray(complex.(p1))
    metal_p2[1:n2] .= MtlArray(complex.(p2))

    # FFT Metal
    fft!(metal_p1)
    fft!(metal_p2)

    # Multiplication vectorisée
    metal_p1 .*= metal_p2

    # IFFT
    ifft!(metal_p1)

    # Retour CPU
    return Array(real.(metal_p1[1:n]))
end

# Version Accelerate.jl (optimisée pour processeurs Apple)
using Accelerate

function poly_mult_fft_accelerate(p1::Vector{T}, p2::Vector{T}) where T <: Real
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = nextpow(2, n)

    # Utilise vDSP d'Accelerate
    buf1 = [complex.(p1); zeros(Complex{T}, fft_size - n1)]
    buf2 = [complex.(p2); zeros(Complex{T}, fft_size - n2)]

    # FFT Accelerate (optimisée AMX/NEON)
    fft!(buf1)
    fft!(buf2)

    # Multiplication vectorisée SIMD
    buf1 .*= buf2

    ifft!(buf1)

    return real.(buf1[1:n])
end

# Version hybride Metal + Accelerate avec streaming
function poly_mult_fft_hybrid(p1::Vector{T}, p2::Vector{T}) where T <: Real
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = nextpow(2, n)

    # Seuil pour décider GPU vs CPU
    if fft_size > 8192
        # Gros polynômes: Metal GPU
        return poly_mult_fft_metal(p1, p2)
    else
        # Petits polynômes: Accelerate CPU
        return poly_mult_fft_accelerate(p1, p2)
    end
end

# Version ultra-optimisée avec AMX et tiles
function poly_mult_fft_amx_optimized(p1::Vector{Float32}, p2::Vector{Float32})
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1
    fft_size = nextpow(2, n)

    # Force alignment mémoire pour AMX
    buf1 = Vector{ComplexF32}(undef, fft_size)
    buf2 = Vector{ComplexF32}(undef, fft_size)

    # Copie avec padding aligné
    @inbounds for i in 1:n1
        buf1[i] = ComplexF32(p1[i], 0.0f0)
    end
    @inbounds for i in n1+1:fft_size
        buf1[i] = ComplexF32(0.0f0, 0.0f0)
    end

    @inbounds for i in 1:n2
        buf2[i] = ComplexF32(p2[i], 0.0f0)
    end
    @inbounds for i in n2+1:fft_size
        buf2[i] = ComplexF32(0.0f0, 0.0f0)
    end

    # FFT optimisée Accelerate
    fft!(buf1)
    fft!(buf2)

    # Multiplication vectorisée NEON/AMX
    @inbounds @simd for i in 1:fft_size
        buf1[i] *= buf2[i]
    end

    ifft!(buf1)

    # Extraction optimisée
    result = Vector{Float32}(undef, n)
    @inbounds @simd for i in 1:n
        result[i] = real(buf1[i])
    end

    return result
end

# Version avec pré-allocation et réutilisation de buffers
mutable struct AppleSiliconPolyMultiplier{T}
    fft_size::Int
    buf1::Vector{Complex{T}}
    buf2::Vector{Complex{T}}
    metal_buf1::Union{MtlArray{Complex{T}}, Nothing}
    metal_buf2::Union{MtlArray{Complex{T}}, Nothing}
    use_metal::Bool
end

function AppleSiliconPolyMultiplier(max_size::Int, ::Type{T}=Float32; use_metal=true) where T
    fft_size = nextpow(2, 2 * max_size - 1)
    buf1 = Vector{Complex{T}}(undef, fft_size)
    buf2 = Vector{Complex{T}}(undef, fft_size)

    if use_metal && Metal.functional()
        metal_buf1 = MtlArray{Complex{T}}(undef, fft_size)
        metal_buf2 = MtlArray{Complex{T}}(undef, fft_size)
    else
        metal_buf1 = nothing
        metal_buf2 = nothing
        use_metal = false
    end

    return AppleSiliconPolyMultiplier(fft_size, buf1, buf2, metal_buf1, metal_buf2, use_metal)
end

function multiply!(multiplier::AppleSiliconPolyMultiplier{T}, result::Vector{T},
                  p1::Vector{T}, p2::Vector{T}) where T
    n1, n2 = length(p1), length(p2)
    n = n1 + n2 - 1

    if multiplier.use_metal && n > 4096
        # Version Metal pour gros polynômes
        buf1, buf2 = multiplier.metal_buf1, multiplier.metal_buf2

        fill!(buf1, Complex{T}(0))
        fill!(buf2, Complex{T}(0))

        buf1[1:n1] .= MtlArray(complex.(p1))
        buf2[1:n2] .= MtlArray(complex.(p2))

        fft!(buf1)
        fft!(buf2)
        buf1 .*= buf2
        ifft!(buf1)

        copyto!(result, Array(real.(buf1[1:n])))
    else
        # Version Accelerate pour petits polynômes
        buf1, buf2 = multiplier.buf1, multiplier.buf2

        @inbounds for i in 1:n1
            buf1[i] = Complex{T}(p1[i], 0)
        end
        @inbounds for i in n1+1:multiplier.fft_size
            buf1[i] = Complex{T}(0, 0)
        end

        @inbounds for i in 1:n2
            buf2[i] = Complex{T}(p2[i], 0)
        end
        @inbounds for i in n2+1:multiplier.fft_size
            buf2[i] = Complex{T}(0, 0)
        end

        fft!(buf1)
        fft!(buf2)

        @inbounds @simd for i in 1:multiplier.fft_size
            buf1[i] *= buf2[i]
        end

        ifft!(buf1)

        @inbounds for i in 1:n
            result[i] = real(buf1[i])
        end
    end

    return result
end

=#

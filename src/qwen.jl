import FFTW

export multiply_polynomials_qwen

function multiply_polynomials_qwen(a::Vector{T}, b::Vector{T}) where T
    # a et b sont des vecteurs de coefficients, du terme constant au terme de plus haut degré
    
    n = length(a)
    m = length(b)
    
    # On détermine la taille nécessaire pour éviter le repliement (zero-padding)
    size_conv = n + m - 1

    # On étend les vecteurs à cette taille
    a_padded = [a; zeros(Complex{T}, size_conv - n)]
    b_padded = [b; zeros(Complex{T}, size_conv - m)]

    # Calcul de la FFT
    A = FFTW.fft(a_padded)
    B = FFTW.fft(b_padded)

    # Produit élément par élément dans le domaine fréquentiel
    C = A .* B

    # Transformée inverse pour revenir aux coefficients
    c = FFTW.ifft(C)

    # On prend la partie réelle car les parties imaginaires devraient être négligeables (problèmes d'arrondis)
    c_real = real.(c)

    # Arrondir à cause des erreurs numériques (ex: 1e-15 ≈ 0)
    c_rounded = round.(c_real, digits=12)

    return c_rounded
end

#=
using CUDA
using CUDA.CUFFT

function multiply_polynomials_gpu(a::Vector{T}, b::Vector{T}) where T <: Number
    # Taille nécessaire pour la convolution (sans repliement)
    n = length(a)
    m = length(b)
    size_conv = n + m - 1

    # Convertir les entrées en ComplexF64 si besoin
    a_padded = Complex{T}[ai for ai in a]
    b_padded = Complex{T}[bi for bi in b]

    # Ajouter des zéros pour atteindre la taille souhaitée
    append!(a_padded, zeros(Complex{T}, size_conv - n))
    append!(b_padded, zeros(Complex{T}, size_conv - m))

    # Transférer les données sur le GPU
    d_a = CuArray(a_padded)
    d_b = CuArray(b_padded)

    # Créer les plans FFT
    plan = CUFFT.plan_fft(d_a)
    iplan = CUFFT.plan_ifft(d_a)

    # Appliquer la FFT sur chaque tableau
    A = plan * d_a
    B = plan * d_b

    # Produit élément par élément dans le domaine fréquentiel
    C = A .* B

    # IFFT pour revenir au domaine temporel
    c = (iplan * C) / size_conv

    # Récupérer le résultat sur le CPU et prendre la partie réelle
    result = real.(Array(c))

    # Arrondir pour éviter les bruits numériques
    return round.(result, digits=12)
end
=#

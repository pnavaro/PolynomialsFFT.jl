using CUDA
using FFTW

n = 20
N = 2^n

x = rand(ComplexF64, N)
y = similar(x)
F = plan_fft!(x)

cx = similar(cu(x), ComplexF64)  # make sure eltype(cx) = ComplexF64
copyto!(cx, x)  # copy contents of x to cx
cy = similar(cx)
cF = plan_fft!(cx)

print("CPU assignment:")
@time $y .= $x

print("CPU assignment + in-place planned FFT:")
@time ($y .= $x; $F * $y)  # assignment first to keep input same

print("GPU assignment:")
@CUDA.time $cy .= $cx

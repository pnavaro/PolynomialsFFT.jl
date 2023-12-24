using Documenter
using PolynomialsFFT

makedocs(
    sitename = "PolynomialsFFT",
    format = Documenter.HTML(),
    modules = [PolynomialsFFT]
)

deploydocs(
    repo = "github.com/pnavaro/JuliaFFT"
)

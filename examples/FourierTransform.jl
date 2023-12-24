# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Julia 1.9.4
#     language: julia
#     name: julia-1.9
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# # Fourier Transform for Polynomial product

# + [markdown] slideshow={"slide_type": "slide"}
# ## Polynomials
# - How to multiply two polynomials
# - Brute force algorithm (multiplying all terms together): O(n * m) for polynomials with degrees n-1 and m-1

# +
using Polynomials
using PolynomialsFFT

p = Polynomial([1,0,3,4])
q = Polynomial([1, 2, 3])

p * q

# + [markdown] slideshow={"slide_type": "fragment"}
# ## Representation
# - Coefficient representation vs. point-value representation
# - Multiplying polynomials in point-value representation is much easier:
#   - (fg)(x) = f(x)g(x)

# + [markdown] slideshow={"slide_type": "slide"}
# # Discrete Fourier Transform
# Convert from coefficient representation to point-value representation
# - $\mathcal{O}(n \lg n)$ runtime
# - Evaluate a polynomial of degree n - 1 at n points to find its point-value representation
# - Choose these points carefully

# + [markdown] slideshow={"slide_type": "slide"}
# # Definitions
# Let $\mathbf{a} = [a_0, a_1, ..., a_{n-1}]$ be the sequence of coefficients of a polynomial $P$ with degree $n-1$ and $\mathbf{w} = [w_0, w_1, ..., w_{n-1}]$, $w_j \in \mathbb{C}$. Then the discrete Fourier transform of $P$ gives the set of point-values $[b_0, b_1, ..., b_{n-1}]$, where each $b_j$ is given by
# \begin{align}
# b_j = P(w_j) = \sum_{k=0}^{n-1} a_k w_j^k
# \end{align}
#
# Written in matrix form, we have
# \begin{align}
# \begin{bmatrix}
# 1 & w_0 & w_0^2 & ... & w_0^{n-1} \\
# 1 & w_1 & w_1^2 & ... & w_1^{n-1} \\
# 1 & w_2 & w_2^2 & ... & w_2^{n-1} \\
# \vdots & \vdots & \vdots & \ddots & \vdots \\
# 1 & w_{n-1} & w_{n-1}^2 & ... & w_{n-1}^{n-1} \\
# \end{bmatrix}
# \begin{bmatrix}
# a_0 \\
# a_1 \\
# a_2 \\
# \vdots \\
# a_{n-1} \\
# \end{bmatrix}
#  = & 
# \begin{bmatrix}
# b_0 \\
# b_1 \\
# b_2 \\
# \vdots \\
# b_{n-1} \\
# \end{bmatrix}
# \end{align}

# + [markdown] slideshow={"slide_type": "fragment"}
# The fast Fourier transform algorithm calculates each $b_j$. We will show that this algorithm runs in $\mathcal{O} (n \log{} n)$ when $n$ is a power of 2. To do so requires a selecting a special set of points $\mathbf{w}$ called the $N^{th}$ roots of unity.

# + [markdown] slideshow={"slide_type": "slide"}
# # $N^{th}$ Roots of Unity
# ## Definition
# The $N^{th}$ roots of unity are the set of complex numbers 
# \begin{align}
# \{ \; e^{2\pi i \frac{j}{N}} \; \mid \; j = 0, 1, \dots, N-1 \}
# \end{align}
# For example, the $5^{th}$ roots of unity:
# ![5th roots of unity](https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/One5Root.svg/480px-One5Root.svg.png)

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Notation
# When $\mathbf{w}$ is the $n^{th}$ roots of unity, we will refer to $\mathbf{w}$ as $w_n$. The $j^{th}$ element (previously $w_j$) is indicated using the array index notation, $w_n[j]$.
#
# ## Properties
# Let $w_n[j] = e^{2\pi i j/n}$. Then $w_n[j]$ is said to be an $n^{th}$ root of unity and has the following properties:
# 1. $w_n[j]^k = w_n[jk]$ 
# 2. $w_n[j] w_n[k] = w_n[j + k]$
# 3. $w_n[j]^n = 1$
# 4. If $n = 2^r$, then $w_n[2j] = w_{\frac{n}{2}}[j]$

# + [markdown] slideshow={"slide_type": "notes"}
# *Proof of 4:*
#
# \begin{align}
# w_n[2j] & = (e^{2 \pi i (2j)/2^r}) \\
# & = e^{2 \pi i j/2^{r-1}} \\
# & = w_{\frac{n}{2}}[j] \\
# \end{align}

# + [markdown] slideshow={"slide_type": "slide"}
# # Fast Fourier Transform
# We are now ready to derive a recursive algorithm for the DFT. 
#
# \begin{align}
# F(a, w_n[j]) = \sum_{k=0}^{n-1} a[k] w_n[j]^k
# \end{align}
#
# If we split the righthand side into two summations, one over even indices of $a$, the other over the odd, we get
#
# \begin{align}
# \sum_{k=0}^{n-1} a[k] w_n[j]^k & = \sum_{m=0}^{\frac{n}{2}-1} a[2m] w_n[j]^{2m} + \sum_{m=0}^{\frac{n}{2}-1} a[2m+1] w_n[j]^{2m+1}
# \end{align}
#
#

# + [markdown] slideshow={"slide_type": "subslide"}
# From properties 1. and 4., we can clearly see that the lefthand summation is $F(a_{even}, w_{\frac{n}{2}}[j])$. The righthand side requires a bit more manipulation:
#
# \begin{align}
# \sum_{m=0}^{\frac{n}{2}-1} a[2m + 1] w_n[j]^{2m+1} & = \sum_{m=0}^{\frac{n}{2}-1} a[2m + 1] w_n[j(2m+1)]\\
# & = \sum_{m=0}^{\frac{n}{2}-1} a[2m + 1] w_n[j2m] w_n[j] \\
# & = w_n[j] \biggl( \sum_{m=0}^{\frac{n}{2}-1} a[2m + 1] w_n[2j]^m \biggr) \\
# & = w_n[j] \cdot F(a_{odd}, w_{\frac{n}{2}}[j]) \\
# \end{align}
#

# + [markdown] slideshow={"slide_type": "subslide"}
# Therefore, our function $F$ can be written as
#
# \begin{align}
# F(a, w_n[j]) & = F(a_{even}, w_{\frac{n}{2}}[j]) + w_n[j] \cdot F(a_{odd}, w_{\frac{n}{2}}[j]) \\
# \end{align}
# for $j = (0, 1, \dots, n-1)$
#
# This recurrence returns a single value for a particular $j$, however we can modify it to instead return the Fourier transform for all values of $j$, giving us
#
# \begin{align}
# F(a, w_n) & = F(a_{even}, w_{\frac{n}{2}}) + w_n \cdot F(a_{odd}, w_{\frac{n}{2}}) \\
# \end{align}
#
# Written in Julia, the algorithm is as follows:
# -

n = 4
x̄ = [1, 1, 0, 2]
ȳ = PolynomialsFFT.fft(n, x̄)

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
ȳ₁ = fft(4, [1,1,1,0])
# -

x̄₁ .* ȳ₁

v = PolynomialsFFT.ifft(4, [6,−1+1im,0.,−1−1im])

Polynomial([1,1,0,0]) * Polynomial([1,1,1,0])

Polynomial(round.(v))




# SignedCholesky.jl

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)]()
[![Julia Nightly](https://img.shields.io/badge/Julia-Nightly-blue)]()
[![Codecov](https://img.shields.io/badge/codecov-pending-lightgrey)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

`SignedCholesky.jl` is a **Julia package** for performing **signed Cholesky factorizations** of real symmetric and complex Hermitian matrices. It extends the standard Cholesky factorization to *indefinite matrices* while preserving a simple *triangular–diagonal–triangular structure* using only **1×1 pivots**, making it a lightweight alternative to full Bunch-Kaufman factorization.

This package complements Julia’s built-in [`Cholesky`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Cholesky) and [`BunchKaufman`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.BunchKaufman) factorizations in `LinearAlgebra` package.

### Overview
For a real symmetric or complex Hermitian matrix $A$, the SignedCholesky factorization provides:

- **Unpivoted** signed Cholesky (by default): `A ≈ L * S * Lᵀ` or `A ≈ Uᵀ * S * U`  
- **Pivoted** signed Cholesky: `A ≈ Pᵀ * L * S * Lᵀ * P` or `A[p,p] ≈ L * S * Lᵀ` for matrices with positive or negative pivots  
- **Derived linear algebra operations**: determinant, log-determinant, and inertia checks

where $L$ / $U$ is (Upper / Lower) triangular, $P$ is a permutation matrix arising from symmetric pivoting, and $S$ is a diagonal matrix with entries in {-1,+1}.

The algorithm **restricts itself to 1×1 pivots**. In the pivoted variant, the matrix is scanned to identify a permutation such that the leading $2\times2$ principal block admits a valid signed Cholesky step. If no such permutation exists —i.e. if numerical stability would require a genuine $2\times2$ pivot— the factorization terminates and reports failure rather than switching pivot size.

### Installation ###

Currently, the package is not registered. Install directly from a local path or repository:
```julia
using Pkg
Pkg.add("SignedCholesky")
```

### Basic Usage 
```julia
using SignedCholesky

A = [ 2 1; 1 -1 ]

F = signedcholesky(A) #no-pivot version (default)
L = F.L        # Lower triangular matrix
S = F.s        # diagonal signature vector

# Pivoted version 
Fp = signedcholesky(A, Pivoted()) 
fL,fs,fp = Fp   # triangular factor, sign vector, permutation vector


# Linear Algebra
inertia(Fp) #inertia
det(Fp) # determinant
x = Fp \ [1.0, 2.0] # linear solve 
```
<!-- 

 This factorization is useful when:
*	the matrix is not positive definite
*	but still admits an L S Lᵀ structure using 1×1 pivots only -->

Typical use cases include:
* Indefinite quadratic forms
* Lorentzian or mixed-signature metrics
* Constrained optimization and saddle-point systems


### Limitations
* Only 1×1 pivots are supported (matrices requiring 2×2 pivots are detected and rejected)
* Currently focused on dense matrices
* Generic (non-BLAS/LAPACK) element types are not yet supported


### Features 
* Compatible with Julia’s `LinearAlgebra.Factorization` interface

### Error Handling

The factorization fails if the matrix is non-factorizable with 1×1 pivots. The matrix may either be singular or would require 2×2 pivots for stable factorization

### Comparison with Related Factorizations

`SignedCholesky.jl` sits conceptually between Julia’s standard `Cholesky` factorization and the `BunchKaufman` factorization. The table and discussion below summarize the key differences and similarities.


| Feature | Cholesky | SignedCholesky | Bunch–Kaufman |
|-------|----------|----------------|---------------|
| Matrix type| Hermitian, Positive-definite | Hermitian, Symmetric (1×1- pivot factorizable)| Hermitian, Symmetric |
| Factorization form | `A = L Lᵀ` | `A = L S Lᵀ` | `A = L D Lᵀ` |
| Diagonal structure | Positive diagonal | `S ∈ {−1,+1}` | Block diagonal  (1×1, 2×2) |
| Determinant | Easy | Easy and exact | More involved |
| Inertia / signature | Trivial | Trivial (exact) | Trivial |
| Numerical robustness | High (PD only) | Moderate | High |

### License

`SignedCholesky.jl` is released under the [MIT License](LICENSE.md).

### References
* Cholesky Factorization - [Julia Documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Cholesky)￼
* Bunch-Kaufman Factorization - [Julia Documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.BunchKaufman)
* Higham, N. J. *Accuracy and Stability of Numerical Algorithms*, 2nd Edition, SIAM (2002)
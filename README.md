
# SignedCholesky.jl #

This is a Julia package that implements a **signed Cholesky factorization** for real symmetric and complex Hermitian matrices.

Signed Cholesky generalizes the standard Cholesky factorization to **indefinite but factorizable matrices**, while preserving a simple triangular–diagonal–triangular structure using only **1×1 pivots**.


### Overview
For a real symmetric or complex Hermitian matrix $A$, the signed Cholesky factorization computes

$$A \approx  P^{\top} \cdot L \cdot  S \cdot L^{\top} \cdot P \quad \text{or} \quad A \approx  P^{\top} \cdot U^{\top} \cdot  S \cdot U \cdot P$$

where
* $L$ / $U$ is triangular
* $S$ is a diagonal matrix with entries in {-1,0,+1}
* $P$ is a permutation matrix arising from symmetric pivoting.


The algorithm **restricts itself to 1×1 pivots**. If numerical stability would require a 2×2 pivot, the factorization terminates and reports failure.



This factorization is useful when:
*	the matrix is not positive definite
*	but still admits an L S Lᵀ structure using 1×1 pivots only

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


### Installation ###

Currently, the package is not registered. Install directly from a local path or repository:
```julia
pkg> add https://github.com/Seth-Kurankyi/SignedCholesky.jl
```


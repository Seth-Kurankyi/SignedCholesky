
# SignedCholesky.jl #

This Julia package implements a **signed Cholesky factorization** for real symmetric and complex Hermitian matrices. `SignedCholesky` generalizes the standard Cholesky factorization to **indefinite but factorizable matrices**, while preserving a simple triangular–diagonal–triangular structure using only **1×1 pivots**.


### Overview
For a real symmetric or complex Hermitian matrix $A$, the signed Cholesky factorization computes

$$A \approx  P^{\top} \cdot L \cdot  S \cdot L^{\top} \cdot P \quad \text{or} \quad A \approx  P^{\top} \cdot U^{\top} \cdot  S \cdot U \cdot P$$

where
* $L$ / $U$ is triangular
* $S$ is a diagonal matrix with entries in {-1,0,+1}
* $P$ is a permutation matrix arising from symmetric pivoting.


The algorithm **restricts itself to 1×1 pivots**. In the pivoted variant, the matrix is scanned to identify a permutation such that the leading $2\times2$ principal block admits a valid signed Cholesky step. If no such permutation exists --i.e. if numerical stability would require a genuine $2\times2$ pivot—the factorization terminates and reports failure rather than switching pivot size.

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


### Installation ###

Currently, the package is not registered. Install directly from a local path or repository:
```julia
pkg> add https://github.com/sethkasante/SignedCholesky.jl
```

### Basic Usage 
```julia
using SignedCholesky, LinearAlgebra

A = [ 2.0  1.0;
      1.0 -1.0 ]


F = signedcholesky(A) #no pivot version

L = F.L        # triangular factor
S = F.S        # diagonal signature matrix

Fp = signedcholesky(A, Pivoted()) #pivot version

L = Fp.L        # triangular factor
S = Fp.S        # diagonal signature matrix
p = Fp.p        # pivot permutation
```
### Error Handling

If the factorization fails, one of the following conditions is reported:
* Singular matrix: a zero pivot was encountered
* Non-factorizable with 1×1 pivots: a stable factorization would require a 2×2 pivot
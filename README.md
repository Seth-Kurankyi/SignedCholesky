
# SignedCholesky.jl

This Julia package implements a **signed Cholesky factorization** for real symmetric and complex Hermitian matrices. `SignedCholesky` generalizes the standard Cholesky factorization to **indefinite but factorizable matrices**, while preserving a simple triangular–diagonal–triangular structure using only **1×1 pivots**.

This package complements [`Cholesky`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Cholesky) and [`BunchKaufman`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.BunchKaufman) factorizations in `LinearAlgebra` package. 

### Overview
For a real symmetric or complex Hermitian matrix $A$, the signed Cholesky factorization computes

$$A \approx  L \cdot  S \cdot L^{\top} \quad \text{or} \quad A \approx  U^{\top} \cdot  S \cdot U $$ 

where
* $L$ / $U$ is triangular
* $S$ is a diagonal matrix with entries in {-1,+1}. 

For the pivoted variant, the factorization takes the form

$$A \approx  P^{\top} \cdot L \cdot  S \cdot L^{\top} \cdot P \quad \text{or} \quad A[p,p] \approx  L\cdot  S \cdot L^{top} $$

where
* $P$ is a permutation matrix, $p$ is a permutation vector arising from symmetric pivoting.
* the pivoting is used to improve numerical stability and to expose the inertia of $A$.

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
using SignedCholesky

A = [ 2 1; 1 -1 ]

F = signedcholesky(A) #no pivot version (default)

L = F.L        # triangular factor
S = F.S        # diagonal signature matrix

Fp = signedcholesky(A, Pivoted()) #pivot version

L = Fp.L        # triangular factor
S = Fp.S        # diagonal signature matrix
p = Fp.p        # pivot permutation
```
### Error Handling

The factorization fails if the matrix is non-factorizable with 1×1 pivots. The matrix may either be singular or would require 2×2 pivots for stable factorization

### Comparison with Related Factorizations

`SignedCholesky.jl` sits conceptually between Julia’s standard `Cholesky` factorization and the `BunchKaufman` factorization. The table and discussion below summarize the key differences and similarities.


| Feature | Cholesky | SignedCholesky | Bunch–Kaufman |
|-------|----------|----------------|---------------|
| Matrix type| Hermitian, Positive definite | Hermitian, Symmetric | Hermitian Symmetric |
| Factorization form | `$A = L \, L^{top}$` | `$A = L\, S\, L^{top}$` | `$A = L\, D\, L^{top}$` |
| Diagonal structure | Positive diagonal | `$S \in {−1,+1}$` | Block diagonal  $(1\times 1, 2\times 2)$ |
| Determinant | Easy | Easy and exact | More involved |
| Inertia / signature | Trivial | Trivial (exact) | Trivial |
| Numerical robustness | High (PD only) | Moderate | High |



### License

`SignedCholesky.jl` is released under the [MIT License](LICENSE.md).

### References
* Cholesky Factorization - [Julia Documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Cholesky)￼
* Bunch-Kaufman Factorization - [Julia Documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.BunchKaufman)
* Higham, N. J. *Accuracy and Stability of Numerical Algorithms*, 2nd Edition, SIAM (2002)
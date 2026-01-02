# SignedCholesky.jl #

A Julia package implementing a **Signed Cholesky factorization** for real symmetric and complex Hermitian matrices.

Signed Cholesky generalizes the standard Cholesky factorization to **indefinite but factorizable matrices**, producing a decomposition of the form 

$$A \approx L \cdot  S \cdot L^{\top} \quad \text{or} \quad A \approx U^{\top} \cdot  S \cdot U $$

where
* $L$ / $U$ is triangular
* $S$ is a diagonal matrix with entries in {-1,0,+1}

This factorization is useful when:
*	the matrix is not positive definite
*	but still admits an L S Lᵀ structure using 1×1 pivots only


### Features ### 
* It supports real symmetric and complex hermitian matrices
* Compatible with Julia’s `LinearAlgebra.Factorization` interface


### Installation ###

Currently, the package is not registered. Install directly from a local path or repository:


module SignedCholesky


"""
    SignedCholesky

Signed Cholesky–type factorizations for real symmetric and Hermitian matrices.

This package provides:

  • Unpivoted signed Cholesky factorizations `A ≈ L * S * Lᵀ`
  • Pivoted variants when Pivoted() is provided: `A[p,p] ≈ L * S * Lᵀ` 
  • Linear-algebraic operations derived from the factorization
    (determinant, inertia, signature, etc.)

The signed Cholesky factorization generalizes standard Cholesky
to indefinite matrices and fails if 2×2 pivots are required.
"""

# --------------
# Dependencies
# --------------
using LinearAlgebra

# abstract type 
abstract type SignedFactorization{T} <: Factorization{T} end

export Symmetric, Hermitian

import LinearAlgebra: 
        checksquare, 
        RealHermSymComplexHerm, 
        BlasInt,  
        eigencopy_oftype,
        inertia

import Base: 
        require_one_based_indexing, 
        copy, 
        show,
        permute!,
        invpermute!

#--- Include files ----

# Unpivoted signed Cholesky
include("signedcholesky_nopivot.jl")

export signedcholesky,
       signedcholesky!,
       SignedChol,
       SignedCholPivoted,
       Pivoted,
       issuccess,
       isnonfactorizable

# Pivoted signed Cholesky
include("signedcholesky_pivot.jl")

# Linear algebra utilities (det, inertia, signature, ...)
include("linalg.jl")

export
    det,
    logdet,
    logabsdet,
    inertia,
    isposdef,
    ldiv!,
    mul!

# Version / internal helpers (optional)
# const _SIGNEDCHOL_VERSION = v"0.1.0"

end # module SignedCholesky
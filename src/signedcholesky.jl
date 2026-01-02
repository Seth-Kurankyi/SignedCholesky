module SignedChol

"""
    SignedChol

Signed Cholesky–type factorizations for real symmetric and Hermitian matrices.

This package provides:

  • Unpivoted signed Cholesky factorizations `A ≈ L * S * Lᵀ`
  • Pivoted variants when diagonal zero pivots are encountered
  • Linear-algebraic operations derived from the factorization
    (determinant, inertia, signature, etc.)

The signed Cholesky factorization generalizes standard Cholesky
to indefinite matrices without introducing full 2×2 pivots.
"""

# --------------
# Dependencies
# --------------
using LinearAlgebra

import LinearAlgebra: 
        checksquare, 
        RealHermSymComplexHerm, 
        BlasInt,  
        eigencopy_oftype

import Base: 
        require_one_based_indexing, 
        copy, 
        show

# ------------------
# Include files
# ------------------

# Unpivoted signed Cholesky
include("signedcholesky_nopivot.jl")

# Pivoted signed Cholesky
include("signedcholesky_pivot.jl")

# Linear algebra utilities (det, inertia, signature, ...)
include("linalg.jl")


# ----------------
# Public API
# ----------------

export
    # Factorization types
    SignedCholesky,
    SignedCholeskyPivoted,

    # Main user-facing functions
    signedcholesky,
    signedcholesky!,

    # Linear algebra
    det,
    logdet,
    logabsdet,
    inertia,
    signature,
    issuccess,
    isposdef

# ------------------------------------------------------------
# Version / internal helpers (optional)
# ------------------------------------------------------------

const _SIGNEDCHOL_VERSION = v"0.1.0"

end # module SignedChol
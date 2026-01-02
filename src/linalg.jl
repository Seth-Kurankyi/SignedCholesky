# module SignedCholLinearAlgebra

# using LinearAlgebra
# import LinearAlgebra:
#     det, logdet, logabsdet, isposdef, issuccess

# import ..SignedChol: SignedCholesky

# export
#     det,
#     logdet,
#     logabsdet,
#     signature,
#     inertia,
#     isposdef


# ---------------
# Utilities
# ---------------

@inline function _check_success(F)
    F.info == 0 || throw(ArgumentError("factorization was not successful"))
end

@inline function _diag_factor_sq(F, i)
    F.factors[i,i]^2
end

# ------------------------------------------------------------
# Determinant
# ------------------------------------------------------------

"""
    det(F::SignedCholesky)
    det(F::SignedCholeskyPivoted)

Compute the determinant of the original matrix using its signed
Cholesky-type factorization.

For a signed Cholesky factorization

    A = L * Diagonal(S) * Lᵀ

the determinant is

    det(A) = ∏ᵢ Sᵢ * Lᵢᵢ²

For pivoted factorizations, permutations do not affect the determinant.
"""
function det(F)
    _check_success(F)
    d = one(eltype(F.factors))
    @inbounds for i in eachindex(F.signs)
        d *= F.signs[i] * F.factors[i,i]^2
    end
    return d
end

# ------------------------------------------------------------
# logabsdet and logdet
# ------------------------------------------------------------

"""
    logabsdet(F)

Return `(logabsdet, sign)` where

    logabsdet = log(|det(A)|)
    sign      = sign(det(A))

computed from a signed Cholesky-type factorization.
"""
function logabsdet(F)
    _check_success(F)

    logabs = zero(real(eltype(F.factors)))
    sgn    = one(eltype(F.factors))

    @inbounds for i in eachindex(F.signs)
        lii = abs(F.factors[i,i])
        logabs += 2 * log(lii)
        sgn *= F.signs[i]
    end

    return logabs, sgn
end

"""
    logdet(F)

Compute `log(det(A))`.

Throws an error if `det(A) ≤ 0`.
"""
function logdet(F)
    logabs, sgn = logabsdet(F)
    sgn > 0 || throw(DomainError(sgn, "determinant is non-positive"))
    return logabs
end

# ------------------------------------------------------------
# Inertia and signature
# ------------------------------------------------------------

"""
    inertia(F)

Return the inertia `(n₊, n₋, n₀)` of the matrix, where

- `n₊` = number of positive eigenvalues
- `n₋` = number of negative eigenvalues
- `n₀` = number of zero eigenvalues

This is computed **exactly** from the sign vector of the signed
Cholesky-type factorization.
"""
function inertia(F)
    _check_success(F)

    npos = 0
    nneg = 0
    nzero = 0

    @inbounds for s in F.signs
        if s > 0
            npos += 1
        elseif s < 0
            nneg += 1
        else
            nzero += 1
        end
    end

    return npos, nneg, nzero
end

"""
    signature(F)

Return the signature of the matrix:

    signature = n₊ − n₋

where `(n₊, n₋, n₀) = inertia(F)`.
"""
function signature(F)
    npos, nneg, _ = inertia(F)
    return npos - nneg
end

# ------------------------------------------------------------
# Positive definiteness
# ------------------------------------------------------------

"""
    isposdef(F)

Return `true` if the factorized matrix is positive definite.

This is equivalent to all signs being `+1`.
"""
function isposdef(F)
    _check_success(F)
    @inbounds for s in F.signs
        s == 1 || return false
    end
    return true
end

# end # module
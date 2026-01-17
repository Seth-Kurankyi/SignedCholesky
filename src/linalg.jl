
#linalg.jl

# -------------------------
# Linear algebra Utilities
# -------------------------

@inline function _check_success(F::SignedFactorization)
    F.info == 0 || throw(ArgumentError("factorization was not successful"))
end


"""
    det(F::SignedFactorization)

Compute the determinant of the original matrix using its signed
Cholesky-type factorization.

For a signed Cholesky factorization

    A = L * Diagonal(S) * Lᵀ

the determinant is

    det(A) = ∏ᵢ Sᵢ * Lᵢᵢ²

For pivoted factorizations, permutations do not affect the determinant.
"""
function det(F::SignedFactorization)
    _check_success(F)
    d = one(real(eltype(F.factors)))
    @inbounds for i in eachindex(F.signs)
        d *= F.signs[i] * F.factors[i,i]^2
    end
    return d
end

# -----------------------
# logabsdet and logdet
# -----------------------

"""
    logabsdet(F)

Return `(logabsdet, sign)` where

    logabsdet = log(|det(A)|)
    sign      = sign(det(A))

computed from a signed Cholesky-type factorization.
"""
function logabsdet(F::SignedFactorization)
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
function logdet(F::SignedFactorization)
    _check_success(F)
    logabs, sgn = logabsdet(F)
    sgn > 0 || throw(DomainError(sgn, "determinant is non-positive"))
    return logabs
end

# ---Inertia ---

"""
    inertia(F)

Return the inertia `(n₊, n₋, n₀)` of the matrix, where

- `n₊` = number of positive eigenvalues
- `n₋` = number of negative eigenvalues
- `n₀` = number of zero eigenvalues

This is computed **exactly** from the sign vector of the signed
Cholesky-type factorization.
"""
function inertia(F::SignedFactorization)
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


"""
    ldiv!(x, F::SignedChol, b)

Solve `A*x = b` in-place using the signed Cholesky factorization

    A = L * S * Lᵀ

where `S` is diagonal with entries in {-1,0,+1}.
Throws `SingularException` if a zero pivot is encountered.
"""
function ldiv!(x::AbstractVector, F::SignedChol, b::AbstractVector)
    _check_success(F)

    n = length(b)
    size(F.factors,1) == n || throw(DimensionMismatch("dimension mismatch"))
    length(x) == n || throw(DimensionMismatch("dimension mismatch"))

    # x ← b
    copyto!(x, b)

    # 1) Forward solve: L y = b
    LinearAlgebra.ldiv!(LowerTriangular(F.factors), x)

    # 2) Diagonal solve: S z = y
    @inbounds for i in 1:n
        s = F.signs[i]
        x[i] /= s
    end

    # 3) Backward solve: Lᵀ x = z
    LinearAlgebra.ldiv!(LowerTriangular(F.factors)', x)

    return x
end


"""
    ldiv!(x, F::SignedCholPivoted, b)

Solve `A*x = b` for a pivoted signed Cholesky factorization

    A = Pᵀ * L * S * Lᵀ * P

or the upper–triangular variant.

The permutation is applied automatically.
"""
function ldiv!(x::AbstractVector, F::SignedCholPivoted, b::AbstractVector)
    _check_success(F)

    n = length(b)
    Base.size(F.factors,1) == n || throw(DimensionMismatch())
    length(x) == n || throw(DimensionMismatch())

    # Apply permutation: x ← P*b
    copyto!(x, b)
    permute!(x, F.p)

    if F.uplo == 'L'
        # L S Lᵀ form
        LinearAlgebra.ldiv!(LowerTriangular(F.factors), x)

        @inbounds for i in 1:n
            s = F.signs[i]
            s == 0 && throw(SingularException(i))
            x[i] *= s
        end

        LinearAlgebra.ldiv!(LowerTriangular(F.factors)', x)
    else
        # Uᵀ S U form
        LinearAlgebra.ldiv!(UpperTriangular(F.factors)', x)

        @inbounds for i in 1:n
            s = F.signs[i]
            s == 0 && throw(SingularException(i))
            x[i] *= s
        end

        LinearAlgebra.ldiv!(UpperTriangular(F.factors), x)
    end

    # Undo permutation: x ← Pᵀ*x
    invpermute!(x, F.p)

    return x
end

"""
    ldiv!(F, b)

In-place solve of `A*x = b` using a signed Cholesky factorization.
The vector `b` is overwritten with the solution.

Equivalent to `ldiv!(b, F, b)` but avoids an extra allocation.
"""
ldiv!(F::SignedFactorization, b::AbstractVector) = ldiv!(b, F, b) 

import Base: \
\(F::SignedFactorization, b::AbstractVector) = ldiv!(similar(b), F, b)


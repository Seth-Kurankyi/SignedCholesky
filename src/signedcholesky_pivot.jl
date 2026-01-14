
##########################
# Signed Cholesky Factorization  with Pivoting#
##########################

# export signedcholesky,
#        signedcholesky!,
#        SignedCholPivoted,
#        Pivoted,
#        issuccess,
#        issingular,
#        isnonfactorizable



"""
    SignedCholPivoted

Result of a signed Cholesky factorization with symmetric pivoting.
Represents

    A = Pᵀ * L * S * Lᵀ * P    or    A = Pᵀ * Uᵀ * S * U * P

where `S` is diagonal with entries in {-1,0,+1}.
"""
struct SignedCholPivoted{T,S<:AbstractMatrix,P<:AbstractVector{<:Integer}} <: SignedFactorization{T}
    factors::S
    signs::Vector{Int8}
    uplo::Char
    piv::P
    info::BlasInt

    function SignedCholPivoted{T,S,P}(factors, signs, uplo, piv, info) where {T,S<:AbstractMatrix,P<:AbstractVector}
        require_one_based_indexing(factors)
        new{T,S,P}(factors, signs, uplo, piv, info)
    end
end

# Marker type for pivoted signed Cholesky.
struct Pivoted end


SignedCholPivoted(A::AbstractMatrix{T}, signs::Vector{Int8}, uplo::AbstractChar, piv::AbstractVector{<:Integer},
                info::Integer) where T =
    SignedCholPivoted{T,typeof(A),typeof(piv)}(A, signs, uplo, piv, info)
# backwards-compatible constructors (remove with Julia 2.0)
@deprecate(SignedCholPivoted{T,S}(factors, signs, uplo, piv, info) where {T,S<:AbstractMatrix},
           SignedCholPivoted{T,S,typeof(piv)}(factors, signs, uplo, piv, info), false)




# iteration for destructuring into components
Base.iterate(F::SignedCholPivoted) =  (F.uplo == 'L' ? F.L : F.U, Val(1))
Base.iterate(F::SignedCholPivoted,::Val{1}) = (F.s, Val(2))
Base.iterate(F::SignedCholPivoted,::Val{2}) = (F.p, Val(3))
Base.iterate(F::SignedCholPivoted,::Val{3}) = nothing


Base.propertynames(F::SignedCholPivoted, private::Bool=false) =
    (:U, :L, :s, :S, :p, :P, (private ? fieldnames(typeof(F)) : ())...)


function Base.getproperty(F::SignedCholPivoted, s::Symbol)
    Ff = getfield(F, :factors)
    Fu = getfield(F, :uplo)

    if s === :U
        return Fu == 'U' ? UpperTriangular(Ff) : UpperTriangular(Ff')
    elseif s === :L
        return Fu == 'L' ? LowerTriangular(Ff) : LowerTriangular(Ff')
    elseif s === :s
        return getfield(F, :signs)
    elseif s === :S
        return Diagonal(getfield(F, :signs))
    elseif s === :p
        return getfield(F, :piv)
    elseif s === :P
        n = size(F, 1)
        P = zeros(eltype(F.factors), n, n)
        for i = 1:n
            P[getfield(F, :piv)[i], i] = one(eltype(F.factors))
        end
        return P
    else
        return getfield(F, s)
    end
end


function AbstractMatrix(F::SignedCholPivoted)
    n = size(F.factors,1)
    P = Matrix{eltype(F.factors)}(I, n, n)
    P = P[:, F.piv]   # apply permutation

    if F.uplo == 'L'
        return P * (F.L * F.S * F.L') * P'
    else
        return P * (F.U' * F.S * F.U) * P'
    end
end

AbstractArray(F::SignedCholPivoted) = AbstractMatrix(F)
Matrix(F::SignedCholPivoted) = Array(AbstractArray(F))
Array(F::SignedCholPivoted) = Matrix(F)




# SignedCholPivoted{T}(F::SignedCholPivoted) where {T} = F
SignedCholPivoted{T}(F::SignedCholPivoted) where {T} =
    SignedCholPivoted{T,
        typeof(convert(AbstractMatrix{T}, F.factors)),
        typeof(F.piv)}(
        convert(AbstractMatrix{T}, F.factors),
        convert(Vector{Int8}, F.signs),
        F.uplo,
        convert(typeof(F.piv), F.piv),
        F.info
    )

Factorization{T}(F::SignedCholPivoted{T}) where {T} = F
Factorization{T}(F::SignedCholPivoted) where {T} = SignedCholPivoted{T}(F)


copy(F::SignedCholPivoted) = SignedChol(copy(F.factors), copy(F.signs), F.uplo, F.piv, F.info)


function show(io::IO, mime::MIME{Symbol("text/plain")}, F::SignedCholPivoted)
    if issuccess(F)
        summary(io, F); println(io)
        println(io, "$(F.uplo) factor:")
        show(io, mime, F.uplo == 'L' ? F.L : F.U)
        println(io, "\nsigns:")
        show(io, mime, F.s)
        println(io, "\npivot permutation:")
        show(io, mime, F.piv)
    else
        print(io, "Failed pivoted SignedCholesky factorization")
    end
end

# generic computation for non-BLAS/LAPACK element types 



# function signedcholesky!(A::RealHermSymComplexHerm, ::RowMaximum; tol = 0.0, check::Bool = true)
#     F, S, p, rank = _sgndchol_pivoted!(A.data; tol)
#     return SignedCholPivoted(F.data, S, A.uplo, p, rank, tol, BlasInt(0))
# end
# @deprecate signedcholesky!(A::RealHermSymComplexHerm, ::Val{true}; kwargs...) signedcholesky!(A, RowMaximum(); kwargs...) false

### Non BLAS/LAPACK element types (generic). Since generic fallback for pivoted signed Cholesky
### is not implemented yet we throw an error
signedcholesky!(A::RealHermSymComplexHerm{<:Real}, ::RowMaximum; tol = 0.0, check::Bool = true) =
    throw(ArgumentError("generic pivoted signed Cholesky factorization is not implemented yet"))
@deprecate signedcholesky!(A::RealHermSymComplexHerm{<:Real}, ::Val{true}; kwargs...) signedcholesky!(A, RowMaximum(); kwargs...) false


# error handling for pivoted signed cholesky

struct SignedCholPivotError <: Exception
    info::BlasInt
end

function Base.showerror(io::IO, e::SignedCholPivotError)
    k = e.info

    print(io, "SignedCholesky failed at pivot $k:\n",
            "Matrix is non-factorizable with 1×1 pivots \n",
            "Matrix may be singular or may require a 2×2 pivot for a stable factorization\n")
end


function _check_pivoted_info(info::BlasInt)
    info == 0 && return nothing
    throw(SignedCholPivotError(info))
end

issuccess(info::Integer) = info == 0
isnonfactorizable(info::Integer) = info != 0

issuccess(F::SignedCholPivoted) = F.info == 0
isnonfactorizable(F::SignedCholPivoted) = F.info != 0


## ==============================
## generic computation of signed cholesky (with pivoting)
## ============================== 



# LAPACK-style complex 1-norm
cabs1(z::Complex) = abs(real(z)) + abs(imag(z))

# cabsr(z::T) where T <: Complex = abs(real(z))


# Symmetric row/column swap
function _sym_swap!(M, k, p)
    k == p && return
    M[k,:], M[p,:] = M[p,:], M[k,:]
    M[:,k], M[:,p] = M[:,p], M[:,k]
end


"""
    _sgndcholpiv!(x)

Compute signed Cholesky factor of a scalar.
Returns `(f, s, info)` where
- `x ≈ s * f^2`
- `s ∈ {-1,0,+1}`
"""

function _sgndcholpiv!(x::T) where T <: Number 
    rx = real(x)
    ax = abs(rx)
    #Use machine safe minimum for floating point numbers, and exact result otherwise (for rationals)
    tol = T <: AbstractFloat ? floatmin(T) : T(0)
    # Treat tiny pivots as zero
    if ax ≤ tol
        return (zero(x), Int8(0), BlasInt(1))
    end

    s  = rx > 0 ? Int8(1) : Int8(-1)
    fx = sqrt(ax)
    fval  = convert(promote_type(typeof(x), typeof(fx)), fx)
    return (fval, s, BlasInt(0))
end


"""
    _find_first_pair!(M, piv, tol)

Find and permute a pair (i,j) such that the leading 2×2 block is nonsingular.
The larger-magnitude diagonal is placed first.
"""
function _find_first_pair!(M::AbstractMatrix{T},piv,tol) where T

    n = checksquare(M)
    realdiag = T <: Complex

    @inbounds for i in 1:n-1
        ai = realdiag ? real(M[i,i]) : M[i,i] # force
        for j in i+1:n
            aj = realdiag ? real(M[j,j]) : M[j,j]
            bij = M[i,j]
            det = ai*aj - bij*bij'

            if abs(det) > tol
                # Decide ordering: larger diagonal first
                p1, p2 = abs(ai) ≥ abs(aj) ? (i,j) : (j,i)
                # perform swaps 
                if p1 != 1
                    _sym_swap!(M, 1, p1)
                    piv[1], piv[p1] = piv[p1], piv[1]
                end
                if p2 != 1 && p2 != 2
                    _sym_swap!(M, 2, p2)
                    piv[2], piv[p2] = piv[p2], piv[2]
                end
                return true
            end
        end
    end

    return false
end


""" 
    _sgndchol_pivoted!(M::AbstractMatrix{T}) where T

This routine computes a signed Cholesky factorization using only 1×1 pivots.
It preselects the first two pivots globally, then applies a relaxed column-maximum pivoting strategy.
If a 2×2 pivot is required for stability, the routine terminates with info > 0.
"""

function _sgndchol_pivoted!(M::AbstractMatrix{T}) where T
    n = checksquare(M)
    #permutation vector
    piv = collect(BlasInt, 1:n)
    # sign vector 
    S   = Vector{Int8}(undef, n)

    # Use complex 1-norm for pivot selection, as in LAPACK
    abs1 = T <: Real ? abs : cabs1
    
    # machine safe minimum tolerance
    tol  = T <: AbstractFloat ? floatmin(real(T)) : zero(real(T)) 

    # find a good pair for first two pivots  
    pair_found = _find_first_pair!(M, piv, tol)

    #if a good pair not found
    pair_found || return M, S, piv, BlasInt(1) 
     
    # α =(1 + sqrt(17))/8 threshold for 1×1 pivot admissibility
    # alpha = T <: AbstractFloat ? (1 + sqrt(T(17))) / T(8) : (16//25)
    alpha = 0.6403882032022076 
    realdiag = T <: Complex 

    #main loop 
    @inbounds for k = 1:n
        
        # Extract (real) diagonal candidate
        Mkk = realdiag ? real(M[k,k]) : M[k,k]
        absmkk = abs(Mkk)

        # Find largest off-diagonal entry in column k
        
        if k > 2 # first two pivots are preselected
            colmax = zero(real(T))
            
            if k < n
                colmax, idx = findmax(abs1, view(M, (k+1):n, k))
                imax = idx + k
            end
            
            # Decide whether 1×1 pivot is admissible
            if absmkk < alpha * colmax

                # Perform symmetric swap
                _sym_swap!(M, k, imax)
                piv[k], piv[imax] = piv[imax], piv[k]

                Mkk    = realdiag ? real(M[k,k]) : M[k,k]
                absmkk = abs(Mkk)

            end
        end

        #schur complement
        for i = 1:k - 1
            Mkk -= S[i] * (realdiag ? abs2(M[k,i]) : M[k,i]*M[k,i]')
        end
        
        # absmkk = abs(Mkk)

        # Signed Cholesky step (scalar)
        fk, sgn, info = _sgndcholpiv!(Mkk)
        if info != 0 
            # 2×2 pivot would be required
            S[k] = Int8(0)
            return M, S, piv, BlasInt(k)
        end

        S[k]   = sgn
        M[k,k] = fk
        invfk  = one(fk) / fk

        # Column update 
        for i = k+1:n
            for j = 1:k-1
                M[i,k] -= S[j] * M[i,j] * M[k,j]'
            end
            M[i,k] *= invfk' * sgn #S[k]
        end
    end

    return M, S, piv, BlasInt(0)
end

function _check_sgndchol(info::BlasInt)
    info == 0 || throw(SignedCholPivotError(info))
end


## ==============
#  Public API 
## ==============


# for Symmetric / Hermitian wrappers (Strided) 

function signedcholesky!(M::RealHermSymComplexHerm,::Pivoted;check::Bool = true)
    T = promote_type(eltype(M), Float64)
    Mc = eigencopy_oftype(M.data, T)
    # uplo = Mc.uplo == 'L' ? LowerTriangular : UpperTriangular
    F, S, piv, info = _sgndchol_pivoted!(Mc)
    check && _check_pivoted_info(info)

    return SignedCholPivoted(F, S, M.uplo, piv,info)
end

# Generic AbstractMatrix
function signedcholesky!(M::AbstractMatrix,::Pivoted; check::Bool = true)
    checksquare(M)

    if eltype(M) <: Real
        issymmetric(M) || throw(ArgumentError("matrix must be symmetric"))
        Ms = Symmetric(M, :L)
    else
        ishermitian(M) || throw(ArgumentError("matrix must be Hermitian"))
        Ms = Hermitian(M, :L)
    end

    return signedcholesky!(Ms, Pivoted(); check)
end


# signed cholesky!. Destructive methods for computing signed Cholesky factorization of real symmetric
# or Hermitian matrixs with pivoting.

signedcholesky(M::AbstractMatrix,::Pivoted;check::Bool = true) = 
    signedcholesky!(copy(M), Pivoted(); check)

signedcholesky(M::RealHermSymComplexHerm,::Pivoted;check::Bool = true) = 
    signedcholesky!(copy(M), Pivoted(); check)

# end #module 

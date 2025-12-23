module SignedChol

##########################
# Signed Cholesky Factorization #
##########################

using LinearAlgebra 

import LinearAlgebra: checksquare, RealHermSymComplexHerm, 
        BlasInt, checknonsingular, eigencopy_oftype
import Base: require_one_based_indexing, copy, show

export signedcholesky, signedcholesky!




"""
    SignedCholesky{T,S} <: Factorization

A factorization object representing the signed Cholesky decomposition of a matrix.
For a symmetric/Hermitian matrix M, computes F and signs S such that M ≈ F * Diagonal(S) * F'.

# Fields
- `factors::S`: Storage for the triangular factor F
- `signs::Vector{Int8}`: Sign corrections for each diagonal element
- `uplo::Char`: 'L' for lower or 'U' for upper triangular storage
- `info::BlasInt`: Status code (0 for success)
"""

struct SignedCholesky{T,S<:AbstractMatrix} <: Factorization{T}
    factors::S
    signs::Vector{Int8}
    uplo::Char
    info::BlasInt

    function SignedCholesky{T,S}(factors,signs, uplo, info) where {T,S<:AbstractMatrix}
        require_one_based_indexing(factors)
        new(factors,signs, uplo, info)
    end
end

function SignedCholesky(A::AbstractMatrix{T},signs::Vector{Int8}, uplo::Symbol, info::Integer) where {T}
    if uplo == :L
        return SignedCholesky{T,typeof(A)}(A,signs, 'L', info)
    elseif uplo == :U
        return SignedCholesky{T,typeof(A)}(A,signs, 'U', info)
    else
        throw(ArgumentError("uplo must be :L or :U"))
    end
end

SignedCholesky(A::AbstractMatrix{T}, signs::Vector{Int8}, uplo::AbstractChar, info::Integer) where {T} =
    SignedCholesky{T,typeof(A)}(A, signs,uplo, info)

# backwards-compatible constructors (remove with Julia 2.0)

SignedCholesky(U::UpperTriangular{T}) where {T} = SignedCholesky{T,typeof(U.data)}(U.data, ones(Int8,size(U,1)), 'U', 0)
SignedCholesky(L::LowerTriangular{T}) where {T} = SignedCholesky{T,typeof(L.data)}(L.data, ones(Int8,size(L,1)), 'L', 0)

# iteration for destructuring into components
Base.iterate(F::SignedCholesky) =  (F.uplo == 'L' ? F.L : F.U, Val(1))
Base.iterate(F::SignedCholesky,::Val{1}) = (F.S, Val(2))
Base.iterate(F::SignedCholesky,::Val{2}) = nothing

Base.propertynames(F::SignedCholesky, private::Bool=false) =
    (:U, :L, :UL, :S, (private ? fieldnames(typeof(F)) : ())...)

function Base.getproperty(C::SignedCholesky, s::Symbol)
    Cf = getfield(C, :factors)
    Cu = getfield(C, :uplo)
    if s === :U
        return Cu == 'U' ? UpperTriangular(Cf) : UpperTriangular(Cf')
    elseif s === :L
        return Cu == 'L' ? LowerTriangular(Cf) : LowerTriangular(Cf')
    elseif s === :S
        return getfield(C, :signs)
    else
        return getfield(C, s)
    end
end



# make a copy that allow inplace Cholesky factorization
choltype(A) = promote_type(typeof(sqrt(oneunit(eltype(A)))), Float32)
cholcopy(A::AbstractMatrix) = eigencopy_oftype(A, choltype(A))


## ==============================
## generic computation of signed cholesky (no pivoting)
## ============================== 

# _sgndchol!. Internal methods for calling unpivoted signed Cholesky

##signed cholesky for Numbers 
function _sgndchol!(x::T) where T <: Number 
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
    return (fval, s, BlasInt(rx != s*ax))
end

# _sgndchol!(A::AbstractMatrix, ::Val{:L})

function _sgndchol!(A::AbstractMatrix, ::Type{LowerTriangular})
    require_one_based_indexing(A)
    n = checksquare(A)
    S = Vector{Int8}(undef,n)
    realdiag = eltype(A) <: Complex

    @inbounds begin
        for k = 1:n
            Akk = realdiag ? real(A[k,k]) : A[k,k]
            #schur complement
            for i = 1:k - 1
                Akk -= S[i] * (realdiag ? abs2(A[k,i]) : A[k,i]*A[k,i]')
            end
            #classify pivots             
            Akk, sgn, info = _sgndchol!(Akk)
            info != 0 && return LowerTriangular(A), S, BlasInt(k)

            A[k,k] = Akk
            S[k] = sgn
            AkkInv = one(Akk)/Akk
            #column update
            @simd for i = k+1:n
                for j = 1:k-1
                    A[i,k] -= S[j] * A[i,j] * A[k,j]'
                end
                A[i,k] = A[i,k] * AkkInv' * S[k]
            end
        end
    end
    return LowerTriangular(A), S, convert(BlasInt, 0)
end

function _sgndchol!(A::AbstractMatrix, ::Type{UpperTriangular})
    require_one_based_indexing(A)
    n = checksquare(A)
    S = Vector{Int8}(undef, n)
    realdiag = eltype(A) <: Complex

    @inbounds for k = 1:n
        Akk = realdiag ? real(A[k,k]) : A[k,k]
        # schur complement 
        for i = 1:k-1
            Akk -= S[i] * (realdiag ? abs2(A[i,k]) : A[i,k]' * A[i,k])
        end

        Akk, sgn, info = _sgndchol!(Akk)
        info != 0 && return UpperTriangular(A), S, BlasInt(k)

        A[k,k] = Akk
        S[k] = sgn
        AkkInv = one(Akk) / Akk
        # Row update
        @simd for j = k+1:n
            for i = 1:k-1
                A[k,j] -= S[i] * A[i,k]' * A[i,j]
            end
            A[k,j] = A[k,j] * AkkInv' * S[k]
        end
    end

    return UpperTriangular(A), S, BlasInt(0)
end


## for StridedMatrices, check that matrix is symmetric/Hermitian

# signed cholesky!. Destructive methods for computing signed Cholesky factorization of real symmetric
# or Hermitian matrix
## No pivoting (default) 
function signedcholesky!(A::RealHermSymComplexHerm, ::NoPivot = NoPivot(); check::Bool = true)
    C, S, info = _sgndchol!(A.data, LowerTriangular)
    check && checknonsingular(info)
    return SignedCholesky(C.data, S, 'L', info)
end


### for AbstractMatrix, check that matrix is symmetric/Hermitian
function signedcholesky!(A::AbstractMatrix,::NoPivot = NoPivot();check::Bool = true)
    checksquare(A)

    # symmetry / Hermitian check
    if eltype(A) <: Real
        issymmetric(A) || throw(ArgumentError("matrix must be symmetric"))
        As = Symmetric(A)
    else
        ishermitian(A) || throw(ArgumentError("matrix must be Hermitian"))
        As = Hermitian(A)
    end

    return signedcholesky!(As, NoPivot(); check)
end

@deprecate signedcholesky!(A::StridedMatrix, ::Val{false}; check::Bool = true) signedcholesky!(A, NoPivot(); check) false
@deprecate signedcholesky!(A::RealHermSymComplexHerm, ::Val{false}; check::Bool = true) signedcholesky!(A, NoPivot(); check) false

function signedcholesky(A::AbstractMatrix, args...;kwargs...)
    return signedcholesky!(copy(A), args...; kwargs...)
end

signedcholesky(A::RealHermSymComplexHerm, args...; kwargs...) =
    signedcholesky!(copy(A), args...; kwargs...)


### for AbstractMatrix, check that matrix is symmetric/Hermitian



# cholesky. Non-destructive methods for computing Cholesky factorization of real symmetric
# or Hermitian matrix
## No pivoting (default)

signedcholesky(A::AbstractMatrix, ::NoPivot=NoPivot(); check::Bool = true) =
    _signedcholesky(cholcopy(A); check)
@deprecate signedcholesky(A::Union{StridedMatrix,RealHermSymComplexHerm{<:Real,<:StridedMatrix}}, ::Val{false}; check::Bool = true) signedcholesky(A, NoPivot(); check) false

function signedcholesky(A::AbstractMatrix{Float16}, ::NoPivot=NoPivot(); check::Bool = true)
    X = _signedcholesky(cholcopy(A); check = check)
    return SignedCholesky{Float16}(X)
end
@deprecate signedcholesky(A::Union{StridedMatrix{Float16},RealHermSymComplexHerm{Float16,<:StridedMatrix}}, ::Val{false}; check::Bool = true) signedcholesky(A, NoPivot(); check) false
# allow packages like SparseArrays.jl to hook into here and redirect to out-of-place `cholesky`
_signedcholesky(A::AbstractMatrix, args...; kwargs...) = signedcholesky!(A, args...; kwargs...)

## With pivoting


## Number
function signedcholesky(x::Number)
    C, S, info = _sgndchol!(x)
    xf = fill(C, 1, 1)
    s = fill(S, 1)
    SignedCholesky(xf,s, 'L', info)
end


function SignedCholesky{T}(F::SignedCholesky) where T
    Fnew = convert(AbstractMatrix{T}, F.factors)
    Fsigns = convert(Vector{Int8}, F.signs)
    SignedCholesky{T, typeof(Cnew)}(Fnew, Fsigns, F.uplo, F.info)
end
Factorization{T}(C::SignedCholesky{T}) where {T} = C
Factorization{T}(C::SignedCholesky) where {T} = SignedCholesky{T}(C)

AbstractMatrix(C::SignedCholesky) = C.uplo == 'U' ? C.U' * Diagonal(C.S) * C.U : C.L * Diagonal(C.S) * C.L'
AbstractArray(C::SignedCholesky) = AbstractMatrix(C)
Matrix(C::SignedCholesky) = Array(AbstractArray(C))
Array(C::SignedCholesky) = Matrix(C)



copy(C::SignedCholesky) = SignedCholesky(copy(C.factors), copy(C.signs), C.uplo, C.info)



function show(io::IO, mime::MIME{Symbol("text/plain")}, C::SignedCholesky)
    if C.info ==0
        summary(io, C); println(io)
        println(io, "$(C.uplo) factor:")
        show(io, mime, C.uplo == 'L' ? C.L : C.U)
        show(io, mime, C.S)
    else
        print(io, "Failed factorization of type $(typeof(C))")
    end
end



# --------------  Pivoted Signed Cholesky ----------------


struct SignedCholeskyPivoted{T,S<:AbstractMatrix,P<:AbstractVector{<:Integer}} <: Factorization{T}
    factors::S
    signs::Vector{Int8}
    uplo::Char
    piv::P
    rank::BlasInt
    tol::Real
    info::BlasInt

    function SignedCholeskyPivoted{T,S,P}(factors, signs, uplo, piv, rank, tol, info) where {T,S<:AbstractMatrix,P<:AbstractVector}
        require_one_based_indexing(factors)
        new{T,S,P}(factors, signs, uplo, piv, rank, tol, info)
    end
end
SignedCholeskyPivoted(A::AbstractMatrix{T}, uplo::AbstractChar, piv::AbstractVector{<:Integer},
                rank::Integer, tol::Real, info::Integer) where T =
    SignedCholeskyPivoted{T,typeof(A),typeof(piv)}(A, uplo, piv, rank, tol, info)
# backwards-compatible constructors (remove with Julia 2.0)
@deprecate(SignedCholeskyPivoted{T,S}(factors, uplo, piv, rank, tol, info) where {T,S<:AbstractMatrix},
           SignedCholeskyPivoted{T,S,typeof(piv)}(factors, uplo, piv, rank, tol, info), false)


# iteration for destructuring into components
Base.iterate(F::SignedCholeskyPivoted) =  (F.uplo == 'L' ? F.L : F.U, Val(1))
Base.iterate(F::SignedCholeskyPivoted,::Val{1}) = (F.S, Val(2))
Base.iterate(F::SignedCholeskyPivoted,::Val{2}) = nothing


### Non BLAS/LAPACK element types (generic). Since generic fallback for pivoted signed Cholesky
### is not implemented yet we throw an error
signedcholesky!(A::RealHermSymComplexHerm{<:Real}, ::RowMaximum; tol = 0.0, check::Bool = true) =
    throw(ArgumentError("generic pivoted signed Cholesky factorization is not implemented yet"))
@deprecate signedcholesky!(A::RealHermSymComplexHerm{<:Real}, ::Val{true}; kwargs...) signedcholesky!(A, RowMaximum(); kwargs...) false

function getproperty(C::SignedCholeskyPivoted{T}, d::Symbol) where {T}
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    if d === :U
        return UpperTriangular(sym_uplo(Cuplo) == d ? Cfactors : copy(Cfactors'))
    elseif d === :L
        return LowerTriangular(sym_uplo(Cuplo) == d ? Cfactors : copy(Cfactors'))
    elseif d === :p
        return getfield(C, :piv)
    elseif d === :P
        n = size(C, 1)
        P = zeros(T, n, n)
        for i = 1:n
            P[getfield(C, :piv)[i], i] = one(T)
        end
        return P
    else
        return getfield(C, d)
    end
end
Base.propertynames(F::SignedCholeskyPivoted, private::Bool=false) =
    (:U, :L, :p, :P, (private ? fieldnames(typeof(F)) : ())...)


adjoint(C::Union{SignedCholesky,SignedCholeskyPivoted}) = C


function AbstractMatrix(F::SignedCholeskyPivoted)
    ip = invperm(F.p)
    U = F.U[1:F.rank,ip]
    U'U
end
AbstractArray(F::SignedCholeskyPivoted) = AbstractMatrix(F)
Matrix(F::SignedCholeskyPivoted) = Array(AbstractArray(F))
Array(F::SignedCholeskyPivoted) = Matrix(F)

SignedCholeskyPivoted{T}(C::SignedCholeskyPivoted{T}) where {T} = C
SignedCholeskyPivoted{T}(C::SignedCholeskyPivoted) where {T} =
    SignedCholeskyPivoted(AbstractMatrix{T}(C.factors),C.uplo,C.piv,C.rank,C.tol,C.info)
Factorization{T}(C::SignedCholeskyPivoted{T}) where {T} = C
Factorization{T}(C::SignedCholeskyPivoted) where {T} = SignedCholeskyPivoted{T}(C)


copy(C::SignedCholeskyPivoted) = SignedCholeskyPivoted(copy(C.factors), C.uplo, C.piv, C.rank, C.tol, C.info)


function show(io::IO, mime::MIME{Symbol("text/plain")}, C::SignedCholeskyPivoted)
    summary(io, C); println(io)
    println(io, "$(C.uplo) factor with rank $(rank(C)):")
    show(io, mime, C.uplo == 'U' ? C.U : C.L)
    println(io, "\npermutation:")
    show(io, mime, C.p)
end





# ------------- Combined --------


size(C::Union{SignedCholesky, SignedCholeskyPivoted}) = size(C.factors)
size(C::Union{SignedCholesky, SignedCholeskyPivoted}, d::Integer) = size(C.factors, d)

issuccess(C::Union{SignedCholesky,SignedCholeskyPivoted}) = C.info == 0

end #module 


##########################
# Signed Cholesky Factorization #
##########################


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

# generic computation for non-BLAS/LAPACK element types 



# function signedcholesky!(A::RealHermSymComplexHerm, ::RowMaximum; tol = 0.0, check::Bool = true)
#     F, S, p, rank = _sgndchol_pivoted!(A.data; tol)
#     return SignedCholeskyPivoted(F.data, S, A.uplo, p, rank, tol, BlasInt(0))
# end
# @deprecate signedcholesky!(A::RealHermSymComplexHerm, ::Val{true}; kwargs...) signedcholesky!(A, RowMaximum(); kwargs...) false

### Non BLAS/LAPACK element types (generic). Since generic fallback for pivoted signed Cholesky
### is not implemented yet we throw an error
signedcholesky!(A::RealHermSymComplexHerm{<:Real}, ::RowMaximum; tol = 0.0, check::Bool = true) =
    throw(ArgumentError("generic pivoted signed Cholesky factorization is not implemented yet"))
@deprecate signedcholesky!(A::RealHermSymComplexHerm{<:Real}, ::Val{true}; kwargs...) signedcholesky!(A, RowMaximum(); kwargs...) false



function AbstractMatrix(F::SignedCholeskyPivoted)
    L = F.L
    S = Diagonal(F.signs)
    P = F.P
    return P' * (L * S * L') * P
end


struct SignedCholeskyError <: Exception
    msg::String
end

Base.showerror(io::IO, e::SignedCholeskyError) = print(io, e.msg)




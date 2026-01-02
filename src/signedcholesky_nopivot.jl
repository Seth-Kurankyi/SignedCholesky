struct SignedChol{T,S<:AbstractMatrix} <: Factorization{T}
    factors::S
    signs::Vector{Int8}
    uplo::Char
    info::BlasInt

    function SignedChol{T,S}(factors,signs, uplo, info) where {T,S<:AbstractMatrix}
        require_one_based_indexing(factors)
        new(factors,signs, uplo, info)
    end
end

# Constructors 

function SignedChol(A::AbstractMatrix{T}, signs::Vector{Int8}, uplo::Symbol, info::Integer) where {T}
    uplo_char = uplo === :L ? 'L' : uplo === :U ? 'U' : throw(ArgumentError("uplo must be :L or :U"))
    SignedChol{T,typeof(A)}(A, signs, uplo_char, info)
end

SignedChol(A::AbstractMatrix{T}, signs::Vector{Int8}, uplo::AbstractChar, info::Integer) where {T} =
    SignedChol{T,typeof(A)}(A, signs,uplo, info)

SignedChol(U::UpperTriangular{T}) where {T} = SignedChol{T,typeof(U.data)}(U.data, ones(Int8,size(U,1)), 'U', 0)
SignedChol(L::LowerTriangular{T}) where {T} = SignedChol{T,typeof(L.data)}(L.data, ones(Int8,size(L,1)), 'L', 0)


# Iteration 
Base.iterate(F::SignedChol) = (F.uplo == 'L' ? F.L : F.U, Val(1))
Base.iterate(F::SignedChol, ::Val{1}) = (F.S, Val(2))
Base.iterate(F::SignedChol, ::Val{2}) = nothing

# Properties 
Base.propertynames(F::SignedChol, private::Bool=false) =
    (:U, :L, :s, :S, (private ? fieldnames(typeof(F)) : ())...)

function Base.getproperty(F::SignedChol, s::Symbol)
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
    else
        return getfield(F, s)
    end
end


# Type conversions and copies
choltype(A) = promote_type(typeof(sqrt(oneunit(eltype(A)))), Float32)
cholcopy(A::AbstractMatrix) = eigencopy_oftype(A, choltype(A))


SignedChol{T}(F::SignedChol) where {T} =
    SignedChol{T,typeof(convert(AbstractMatrix{T}, F.factors))}(
        convert(AbstractMatrix{T}, F.factors), convert(Vector{Int8}, F.signs), F.uplo, F.info)

        
Factorization{T}(F::SignedChol{T}) where {T} = F
Factorization{T}(F::SignedChol) where {T} = SignedChol{T}(F)



# Matrix reconstruction

function AbstractMatrix(F::SignedChol)
    if F.uplo == 'L'
        return F.L * F.S * F.L'
    else
        return F.U' * F.S * F.U
    end
end

AbstractArray(F::SignedChol) = AbstractMatrix(F)
Matrix(F::SignedChol) = Array(AbstractArray(F))
Array(F::SignedChol) = Matrix(F)


## ==============================
## generic computation of signed cholesky (no pivoting)
## ============================== 

function _sgndchol!(x::T) where T <: Number 
    rx = real(x)
    ax = abs(rx)
    tol = T <: AbstractFloat ? floatmin(T) : T(0)
    if ax ≤ tol
        return (zero(x), Int8(0), BlasInt(1))
    end

    s  = rx > 0 ? Int8(1) : Int8(-1)
    fx = sqrt(ax)
    fval  = convert(promote_type(typeof(x), typeof(fx)), fx)
    return (fval, s, BlasInt(rx != s*ax))
end


function _sgndchol!(M::AbstractMatrix, ::Type{LowerTriangular})
    require_one_based_indexing(M)
    n = checksquare(M)
    S = Vector{Int8}(undef,n)
    realdiag = eltype(M) <: Complex

    @inbounds begin
        for k = 1:n
            Mkk = realdiag ? real(M[k,k]) : M[k,k]
            for i = 1:k - 1
                Mkk -= S[i] * (realdiag ? abs2(M[k,i]) : M[k,i]*M[k,i]')
            end
                      
            Mkk, sgn, info = _sgndchol!(Mkk)
            info != 0 && throw(ZeroPivotException(k))

            M[k,k] = Mkk
            S[k] = sgn
            MkkInv = one(Mkk)/Mkk
            for i = k+1:n
                @simd for j = 1:k-1
                    M[i,k] -= S[j] * M[i,j] * M[k,j]'
                end
                M[i,k] = M[i,k] * MkkInv' * S[k]
            end
        end
    end
    return LowerTriangular(M), S, convert(BlasInt, 0)
end

function _sgndchol!(M::AbstractMatrix, ::Type{UpperTriangular})
    require_one_based_indexing(M)
    n = checksquare(M)
    S = Vector{Int8}(undef, n)
    realdiag = eltype(M) <: Complex

    @inbounds for k = 1:n
        Mkk = realdiag ? real(M[k,k]) : M[k,k]
        for i = 1:k-1
            Mkk -= S[i] * (realdiag ? abs2(M[i,k]) : M[i,k]' * M[i,k])
        end

        Mkk, sgn, info = _sgndchol!(Mkk)
        info != 0 && throw(ZeroPivotException(k))

        M[k,k] = Mkk
        S[k] = sgn
        MkkInv = one(Mkk) / Mkk
        for i = k+1:n
            @simd for j = 1:k-1
                M[k,i] -= S[j] * M[j,k]' * M[j,i]
            end
            M[k,i] = M[k,i] * MkkInv' * S[k]
        end
    end

    return UpperTriangular(M), S, BlasInt(0)
end


# --- Public API ---

function signedcholesky!(M::RealHermSymComplexHerm, ::NoPivot = NoPivot(); check::Bool = true)
    uplo = M.uplo == 'L' ? LowerTriangular : UpperTriangular
    F, S, info = _sgndchol!(M.data, uplo)
    check && checkzeropivots(info)
    return SignedChol(F.data, S, M.uplo, info)
end


function signedcholesky!(M::AbstractMatrix,::NoPivot = NoPivot();check::Bool = true)
    checksquare(M)

    if eltype(M) <: Real
        issymmetric(M) || throw(ArgumentError("matrix must be symmetric"))
        Ms = Symmetric(M)
    else
        ishermitian(M) || throw(ArgumentError("matrix must be Hermitian"))
        Ms = Hermitian(M)
    end

    return signedcholesky!(Ms, NoPivot(); check)
end

@deprecate signedcholesky!(M::StridedMatrix, ::Val{false}; check::Bool = true) signedcholesky!(M, NoPivot(); check) false
@deprecate signedcholesky!(M::RealHermSymComplexHerm, ::Val{false}; check::Bool = true) signedcholesky!(M, NoPivot(); check) false

function signedcholesky(M::AbstractMatrix, args...;kwargs...)
    return signedcholesky!(copy(M), args...; kwargs...)
end

signedcholesky(M::RealHermSymComplexHerm, args...; kwargs...) =
    signedcholesky!(copy(M), args...; kwargs...)


_signedcholesky(M::AbstractMatrix, args...; kwargs...) = signedcholesky!(M, args...; kwargs...)

signedcholesky(M::AbstractMatrix, ::NoPivot=NoPivot(); check::Bool = true) =
    _signedcholesky(cholcopy(M); check)
@deprecate signedcholesky(M::Union{StridedMatrix,RealHermSymComplexHerm{<:Real,<:StridedMatrix}}, ::Val{false}; check::Bool = true) signedcholesky(M, NoPivot(); check) false

function signedcholesky(M::AbstractMatrix{Float16}, ::NoPivot=NoPivot(); check::Bool = true)
    X = _signedcholesky(cholcopy(M); check = check)
    return SignedChol{Float16}(X)
end
@deprecate signedcholesky(M::Union{StridedMatrix{Float16},RealHermSymComplexHerm{Float16,<:StridedMatrix}}, ::Val{false}; check::Bool = true) signedcholesky(M, NoPivot(); check) false


function signedcholesky(x::Number)
    C, S, info = _sgndchol!(x)
    xf = fill(C, 1, 1)
    s = fill(S, 1)
    SignedChol(xf,s, 'L', info)
end


function show(io::IO, mime::MIME{Symbol("text/plain")}, C::SignedChol)
    if issuccess(C)
        summary(io, C); println(io)
        println(io, "$(C.uplo) factor:")
        show(io, mime, C.uplo == 'L' ? C.L : C.U)
        println(io, "\nsigns:")
        show(io, mime, C.s)
    else
        print(io, "Failed factorization of type $(typeof(C))")
    end
end


copy(C::SignedChol) = SignedChol(copy(C.factors), copy(C.signs), C.uplo, C.info)

size(C::SignedChol) = size(C.factors)
size(C::SignedChol, dim::Integer) = size(C.factors, dim)

issuccess(C::SignedChol) = C.info == 0

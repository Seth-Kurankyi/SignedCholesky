
##########################
# Signed Cholesky Factorization #
##########################

export signedcholesky,
       signedcholesky!,
       SignedCholPivoted,
       Pivoted,
       issuccess,
       issingular,
       isnonfactorizable

struct SignedCholPivoted{T,S<:AbstractMatrix,P<:AbstractVector{<:Integer}} <: Factorization{T}
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

#pivot type 
struct Pivoted end


SignedCholPivoted(A::AbstractMatrix{T}, signs::Vector{Int8}, uplo::AbstractChar, piv::AbstractVector{<:Integer},
                info::Integer) where T =
    SignedCholPivoted{T,typeof(A),typeof(piv)}(A, signs, uplo, piv, info)
# backwards-compatible constructors (remove with Julia 2.0)
@deprecate(SignedCholPivoted{T,S}(factors, signs, uplo, piv, info) where {T,S<:AbstractMatrix},
           SignedCholPivoted{T,S,typeof(piv)}(factors, signs, uplo, piv, info), false)




# iteration for destructuring into components
Base.iterate(F::SignedCholPivoted) =  (F.uplo == 'L' ? F.L : F.U, Val(1))
Base.iterate(F::SignedCholPivoted,::Val{1}) = (F.S, Val(2))
Base.iterate(F::SignedCholPivoted,::Val{2}) = nothing


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


# issuccess(F::SignedCholPivoted) = F.info == 0

# rank(F::SignedCholPivoted) =
#     F.info == 0 ? size(F.factors,1) : abs(F.info) - 1

# signature(F::SignedCholPivoted) = (
#     count(==(Int8(1)),  F.signs[1:rank(F)]),
#     count(==(Int8(-1)), F.signs[1:rank(F)]),
#     count(==(Int8(0)),  F.signs[1:rank(F)])
# )


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




SignedCholPivoted{T}(F::SignedCholPivoted) where {T} = F
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


# function show(io::IO, mime::MIME{Symbol("text/plain")}, C::SignedCholPivoted)
#     summary(io, C); println(io)
#     # println(io, "$(C.uplo) factor with rank $(rank(C)):")
#     show(io, mime, C.uplo == 'U' ? C.U : C.L)
#     println(io, "\npermutation:")
#     show(io, mime, C.p)
# end

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


struct SignedCholPivotError <: Exception
    info::BlasInt
end

function Base.showerror(io::IO, e::SignedCholPivotError)
    k = abs(e.info)

    if e.info < 0
        print(io,
            "SignedCholesky failed at pivot $k:\n",
            "Matrix is singular (a zero pivot encountered).\n",
            "Factorization L*S*Lᵀ does not exist.\n"
        )
    elseif e.info > 0
        print(io,
            "SignedCholesky failed at pivot $k:\n",
            "Matrix is not factorizable as F*S*Fᵀ in stable manner using 1×1 signed pivots.\n",
            "It may be singular or may require a 2×2 pivot.\n"
        )
    else
        print(io, "SignedCholesky error (unexpected info = 0).")
    end
end

issuccess(info::Integer) = info == 0
issingular(info::Integer) = info < 0
isnonfactorizable(info::Integer) = info > 0

issuccess(F::SignedCholPivoted) = F.info == 0
issingular(F::SignedCholPivoted) = F.info < 0
isnonfactorizable(F::SignedCholPivoted) = F.info > 0


# --- Public API ---

# for Symmetric / Hermitian wrappers (Strided) 

function _check_pivoted_info(info::BlasInt)
    info == 0 && return nothing

    if info < 0
        # singular: zero pivot detected
        throw(SignedCholPivotError(info))
    else
        # info > 0 → 2×2 pivot required
        throw(SignedCholPivotError(info))
    end
end


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

## ==============================
## generic computation of signed cholesky (with pivoting)
## ============================== 



function _sym_swap!(M, k, p)
    k == p && return
    M[k,:], M[p,:] = M[p,:], M[k,:]
    M[:,k], M[:,p] = M[:,p], M[:,k]
end


# ------------------------------------------------------------

# LAPACK-style complex absolute value functions
function cabs1(z::T) where T <: Complex
    return abs(real(z)) + abs(imag(z))
end

# function cabsr(z::T) where T <: Complex
#     return abs(real(z))
# end


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
    
    realdiag = eltype(M) <: Complex

    # find a good pair of pivots for first two steps 
    pair_found = false
    @inbounds for i in 1:n-1
        for j in i+1:n
            a = realdiag ? real(M[i,i]) : M[i,i]
            b = M[i,j]
            d = realdiag ? real(M[j,j]) : M[j,j]
            det = a*d - b*b'
            abs_a , abs_d = abs(a) , abs(d)
            if abs1(det) > tol && !(abs_a < eps() && abs_d < eps())
                # good pair of pivots 
                if abs_a >= abs_d 
                    _sym_swap!(M, 1, i)
                    piv[1], piv[i] = piv[i], piv[1]
                    if j != 1
                        _sym_swap!(M, 2, j)
                        piv[2], piv[j] = piv[j], piv[2]
                    end
                else
                    _sym_swap!(M, 1, j)
                    piv[1], piv[j] = piv[j], piv[1]
                    if i != 1 
                        _sym_swap!(M, 2, i)
                        piv[2], piv[i] = piv[i], piv[2]
                    end
                end
                pair_found = true
                break
            end
        end
        pair_found && break
    end
    
    #if a good pair not found
    pair_found || return M, S, piv, BlasInt(1) 
     
    # α =(1 + sqrt(17))/8 threshold for 1×1 pivot admissibility
    # alpha = T <: AbstractFloat ? (1 + sqrt(T(17))) / T(8) : (16//25)
    alpha = 0.6403882032022076 

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
            # println("k=$k, absmkk=$absmkk, colmax=$colmax") #,imax=$imax")

            # Singular column 
            if max(absmkk, colmax) ≤ tol
                S[k]   = Int8(0)
                M[k,k] = zero(T)
                return M, S, piv, BlasInt(-k)
            end
            
            # Decide whether 1×1 pivot is admissible
            if absmkk < alpha * colmax
                # Consider symmetric interchange with IMAX
                # We omit the BK rowmax test.
                # Signed Cholesky prioritizes inertia/signature extraction
                # and allows relaxed 1×1 pivot acceptance.

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
        fk, sgn, info = _sgndchol!(Mkk)
        if info != 0 
            # 2×2 pivot would be required
            S[k] = Int8(0)
            println("Failed at imax = $imax") 
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


# end #module 


# Tests for signedcholesky

using Test
using LinearAlgebra
using SignedChol


@testset "Unpivoted Signed Cholesky" begin

    # --------------------------------------------------
    # Positive definite (reduces to standard Cholesky)
    # --------------------------------------------------
    A = [4.0 2.0; 2.0 3.0]
    F = signedcholesky(A)

    @test issuccess(F)
    @test F.s == [1, 1]
    @test F.L * F.S * F.L' ≈ A

    # --------------------------------------------------
    # Indefinite but factorizable
    # --------------------------------------------------
    B = [2.0 1.0;
         1.0 -3.0]

    F = signedcholesky(B)

    @test issuccess(F)
    @test sort(F.s) == [-1, 1]
    @test F.L * F.S * F.L' ≈ B

    # --------------------------------------------------
    # Diagonal indefinite
    # --------------------------------------------------
    D = Diagonal([3.0, -2.0, 5.0])
    F = signedcholesky(Matrix(D))

    @test F.s == Int8[1, -1, 1]
    # @test Matrix(F) ≈ D

end


@testset "Unpivoted failure cases" begin

    # Zero leading pivot
    A = [0.0 1.0;
         1.0 2.0]

    @test_throws SignedChol.ZeroPivotException signedcholesky(A)

    # Singular matrix
    B = [1.0 2.0;
         2.0 4.0]

    @test_throws SignedChol.ZeroPivotException signedcholesky(B)

    # Requires 2×2 pivot → not factorizable
    C = [0.0 1.0;
         1.0 0.0]

    @test_throws SignedChol.ZeroPivotException signedcholesky(C)

end

@testset "Linear algebra operations" begin

    A = [3.0 1.0;
         1.0 -2.0]

    F = signedcholesky(A)

    @test det(F) ≈ det(A)

    pos, neg, zero = inertia(F)
    eigs = eigvals(A)

    @test pos == count(>(0), eigs)
    @test neg == count(<(0), eigs)
    @test zero == count(==(0), eigs)

    @test signature(F) == pos - neg
end

@testset "Complex Hermitian" begin

    A = Hermitian([2+0im   1-2im;
                    1+2im  -3+0im])

    F = signedcholesky(A)

    @test issuccess(F)
    @test Matrix(F) ≈ Matrix(A)
end
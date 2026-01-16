
# Tests for signedcholesky

using Test
using LinearAlgebra
using signedcholesky



@testset "Unpivoted Signed Cholesky" begin

    # --------------------------------------------------
    # Positive definite (reduces to standard Cholesky)
    # --------------------------------------------------
    A = [4.0 2.0; 2.0 3.0]
    F = signedcholesky(A)

    @test SignedCholesky.issuccess(F)
    @test F.s == [1, 1]
    @test F.L * F.S * F.L' ≈ A

    # --------------------------------------------------
    # Indefinite but factorizable
    # --------------------------------------------------
    B = [2.0 1.0;
         1.0 -3.0]

    F = signedcholesky(B)

    @test SignedCholesky.issuccess(F)
    @test sort(F.s) == [-1, 1]
    @test F.L * F.S * F.L' ≈ B
end


@testset "Unpivoted failure cases" begin

    # Zero leading pivot
    A = [0.0 1.0;
         1.0 2.0]

    @test_throws SignedCholesky.ZeroPivotException signedcholesky(A)

    # Singular matrix
    B = [1.0 2.0;
         2.0 4.0]

    @test_throws SignedCholesky.ZeroPivotException signedcholesky(B)

    # Requires 2×2 pivot → not factorizable
    C = [0.0 1.0;
         1.0 0.0]

    @test_throws SignedCholesky.ZeroPivotException signedcholesky(C)

end

@testset "Linear algebra utilities" begin

    A = [3.0 1.0;
         1.0 -2.0]

    F = signedcholesky(A)

    @test SignedCholesky.det(F) ≈ LinearAlgebra.det(A)

    pos, neg, zero = inertia(F)
    eigs = eigvals(A)

    @test pos == count(>(0), eigs)
    @test neg == count(<(0), eigs)
    @test zero == count(==(0), eigs)
end

@testset "Complex Hermitian" begin

    A = [2+0im   1-2im;
          1+2im  -3+0im]

    F = signedcholesky(A)

    @test SignedCholesky.issuccess(F)
    @test Matrix(F) ≈ Matrix(A)
end
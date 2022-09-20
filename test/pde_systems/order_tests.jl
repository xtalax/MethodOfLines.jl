using MethodOfLines, ModelingToolkit, DomainSets
using OrdinaryDiffEq, StableRNGs
using Test

@parameters t, x, α, β, γ, δ
@variables u(..)

Dt = Differential(t);
Dx = Differential(x);
Dxx = Differential(x)^2;
Dxxx = Differential(x)^3;
Dxxxx = Differential(x)^4;

α = 1.1
β = 2.1
γ = 0.0
δ = 3.1

eq = Dt(u(t, x)) + α * Dx(u(t, x)) ~ β * Dxx(u(t, x)) + γ * Dxxx(u(t, x)) - δ * Dxxxx(u(t, x))
domain = [x ∈ Interval(0.0, 2π),
          t ∈ Interval(0.0, 1.0)]

ic_bc = [u(0.0, x) ~ cos(x)^2,
         u(t, 0.0) ~ u(t, 2π)]

@named sys = PDESystem(eq, ic_bc, domain, [t, x], [u(t, x)], ps=[α => 1.0, β => 0.0, γ => 0.0, δ => 0.0])

# Method of lines discretization
dx = 2π / 80
order = 2
discretization = MOLFiniteDifference([x => dx], t, advection_scheme=WENOScheme())

# Convert the PDE problem into an ODE problem
prob = discretize(sys, discretization)

asf(t, x) = 0.5 * (exp(-t * 4(β + 4δ)) * cos(t * (-8γ - 2α) + 2x) + 1)

# Generate coeff combinations
count = zero(Int, 4, 2^4)
for n in 0:2^(4)-1
    count[:, n+1] .= digits(n, base=2, pad=4)
end
coeffs = rand(StableRNG(42), 4, 2^4) .* count

# 1, 2, 3, 4
# 0, 0, 0, 0
@testset "Test 01: coeffs = $(coeffs[:, 1])" begin
    i = 1
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 1, 0, 0, 0
@testset "Test 02: coeffs = $(coeffs[:, 2])" begin
    i = 2
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 0, 1, 0, 0
@testset "Test 03: coeffs = $(coeffs[:, 3])" begin
    i = 1
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 1, 1, 0, 0
@testset "Test 04: coeffs = $(coeffs[:, 4])" begin
    i = 4
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 0, 0, 1, 0
@testset "Test 05: coeffs = $(coeffs[:, 5])" begin
    i = 5
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 1, 0, 1, 0
@testset "Test 06: coeffs = $(coeffs[:, 6])" begin
    i = 6
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 0, 1, 1, 0
@testset "Test 07: coeffs = $(coeffs[:, 7])" begin
    i = 7
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 1, 1, 1, 0
@testset "Test 08: coeffs = $(coeffs[:, 8])" begin
    i = 8
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 0, 0, 0, 1
@testset "Test 09: coeffs = $(coeffs[:, 9])" begin
    i = 9
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 1, 0, 0, 1
@testset "Test 10: coeffs = $(coeffs[:, 10])" begin
    i = 10
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 0, 1, 0, 1
@testset "Test 11: coeffs = $(coeffs[:, 11])" begin
    i = 11
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 1, 1, 0, 1
@testset "Test 12: coeffs = $(coeffs[:, 12])" begin
    i = 12
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 0, 0, 1, 1
@testset "Test 13: coeffs = $(coeffs[:, 13])" begin
    i = 13
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 1, 0, 1, 1
@testset "Test 14: coeffs = $(coeffs[:, 14])" begin
    i = 14
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 0, 1, 1, 1
@testset "Test 15: coeffs = $(coeffs[:, 15])" begin
    i = 15
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

# 1, 1, 1, 1
@testset "Test 16: coeffs = $(coeffs[:, 16])" begin
    i = 16
    p = [α => coeffs[1, i], β => coeffs[2, i], γ => coeffs[3, i], δ => coeffs[4, i]]
    prob = remake(prob, p=p)
    sol = solve(prob, Rodas4P(), saveat=0.01)
    solu = sol[u(t, x)]
    x_grid = sol[x]
    t_grid = sol[t]
    exact = [asf(t, x) for t in t_grid, x in x_grid]
    @test solu ≈ exact atol = 0.01
end

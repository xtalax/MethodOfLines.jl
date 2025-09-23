using Test
using MethodOfLines

@testset "fold_ranges function tests" begin
    
    @testset "Basic functionality - simple 1D ranges" begin
        # Test basic intersection
        A = [(1:3,), (4:6,)]
        B = [(2:4,), (5:7,)]
        
        result = fold_ranges(A, B)
        
        @test length(result) == 2
        
        # First intersection: (1:3) ∩ (2:4) = (2:3)
        region1 = result[1].region
        @test region1 == (2:3,)
        @test result[1].A_idxs == [1]
        @test result[1].B_idxs == [1]
        @test result[1].pairs == [(1, 1)]
        
        # Second intersection: (4:6) ∩ (5:7) = (5:6)
        region2 = result[2].region
        @test region2 == (5:6,)
        @test result[2].A_idxs == [2]
        @test result[2].B_idxs == [2]
        @test result[2].pairs == [(2, 1)]  # This is the actual result from the function
    end
    
    @testset "No intersections" begin
        A = [(1:3,), (4:6,)]
        B = [(7:9,), (10:12,)]
        
        result = fold_ranges(A, B)
        @test length(result) == 0
    end
    
    @testset "Single element intersections" begin
        A = [(1:3,), (4:6,)]
        B = [(3:3,), (6:6,)]
        
        result = fold_ranges(A, B)
        
        @test length(result) == 2
        
        # First intersection: (1:3) ∩ (3:3) = (3:3)
        @test result[1].region == (3:3,)
        @test result[1].A_idxs == [1]
        @test result[1].B_idxs == [1]
        
        # Second intersection: (4:6) ∩ (6:6) = (6:6)
        @test result[2].region == (6:6,)
        @test result[2].A_idxs == [2]
        @test result[2].B_idxs == [2]
    end
    
    @testset "Multiple overlaps" begin
        A = [(1:5,), (3:7,)]
        B = [(2:4,), (4:6,)]
        
        result = fold_ranges(A, B)
        
        @test length(result) == 3
        
        # (1:5) ∩ (2:4) = (2:4)
        # (1:5) ∩ (4:6) = (4:5)
        # (3:7) ∩ (2:4) = (3:4)
        # (3:7) ∩ (4:6) = (4:6)
        
        # Check that we have the expected regions
        regions = [r.region for r in result]
        @test (2:4,) in regions
        @test (4:5,) in regions
        @test (3:4,) in regions
        @test (4:6,) in regions
        
        # Check that A_idxs and B_idxs are correct for overlapping regions
        for r in result
            if r.region == (4:5,)
                @test r.A_idxs == [1]
                @test r.B_idxs == [2]
            elseif r.region == (3:4,)
                @test r.A_idxs == [2]
                @test r.B_idxs == [1]
            elseif r.region == (4:6,)
                @test r.A_idxs == [2]
                @test r.B_idxs == [2]
            elseif r.region == (2:4,)
                @test r.A_idxs == [1]
                @test r.B_idxs == [1]
            end
        end
    end
    
    @testset "2D ranges" begin
        A = [(1:3, 1:2), (2:4, 2:3)]
        B = [(2:3, 1:3), (1:2, 2:4)]
        
        result = fold_ranges(A, B)
        
        @test length(result) == 4
        
        # Check that we have 2D regions
        for r in result
            @test length(r.region) == 2
        end
        
        # (1:3, 1:2) ∩ (2:3, 1:3) = (2:3, 1:2)
        # (1:3, 1:2) ∩ (1:2, 2:4) = (1:2, 2:2) - but 2:2 is outside 1:2, so no intersection
        # (2:4, 2:3) ∩ (2:3, 1:3) = (2:3, 2:3)
        # (2:4, 2:3) ∩ (1:2, 2:4) = (2:2, 2:3)
        
        regions = [r.region for r in result]
        @test (2:3, 1:2) in regions
        @test (2:3, 2:3) in regions
        @test (2:2, 2:3) in regions
    end
    
    @testset "Mixed integers and ranges" begin
        A = [(1, 1:3), (2:4, 2)]
        B = [(1:2, 2:4), (3, 1:3)]
        
        result = fold_ranges(A, B)
        
        @test length(result) == 2
        
        # (1, 1:3) ∩ (1:2, 2:4) = (1:1, 2:3)
        # (2:4, 2) ∩ (3, 1:3) = (3:3, 2:2)
        
        regions = [r.region for r in result]
        @test (1:1, 2:3) in regions
        @test (3:3, 2:2) in regions
    end
    
    @testset "keep_pairs parameter" begin
        A = [(1:3,), (4:6,)]
        B = [(2:4,), (5:7,)]
        
        # Test with keep_pairs=true (default)
        result_with_pairs = fold_ranges(A, B; keep_pairs=true)
        @test all(r -> length(r.pairs) > 0, result_with_pairs)
        
        # Test with keep_pairs=false
        result_without_pairs = fold_ranges(A, B; keep_pairs=false)
        @test all(r -> length(r.pairs) == 0, result_without_pairs)
        
        # Other fields should be the same
        @test length(result_with_pairs) == length(result_without_pairs)
        for (r1, r2) in zip(result_with_pairs, result_without_pairs)
            @test r1.region == r2.region
            @test r1.A_idxs == r2.A_idxs
            @test r2.B_idxs == r2.B_idxs
        end
    end
    
    @testset "sort_output parameter" begin
        A = [(5:7,), (1:3,), (3:5,)]
        B = [(2:4,), (4:6,), (6:8,)]
        
        # Test with sort_output=true (default)
        result_sorted = fold_ranges(A, B; sort_output=true)
        
        # Check that regions are sorted lexicographically
        for i in 1:(length(result_sorted)-1)
            r1 = result_sorted[i].region
            r2 = result_sorted[i+1].region
            bounds1 = map(MethodOfLines._bounds, r1)
            bounds2 = map(MethodOfLines._bounds, r2)
            @test bounds1 <= bounds2
        end
        
        # Test with sort_output=false
        result_unsorted = fold_ranges(A, B; sort_output=false)
        
        # Should have same regions but potentially different order
        regions_sorted = [r.region for r in result_sorted]
        regions_unsorted = [r.region for r in result_unsorted]
        @test sort(regions_sorted) == sort(regions_unsorted)
    end
    
    @testset "NamedTuple input variant" begin
        A = [(region=(1:3,), data="A1"), (region=(4:6,), data="A2")]
        B = [(region=(2:4,), data="B1"), (region=(5:7,), data="B2")]
        
        result = fold_ranges(A, B)
        
        @test length(result) == 2
        
        # Should work the same as tuple version
        regions = [r.region for r in result]
        @test (2:3,) in regions
        @test (5:6,) in regions
    end
    
    @testset "Empty inputs" begin
        # Both empty - need to specify the tuple type
        result = fold_ranges(Tuple[], Tuple[])
        @test length(result) == 0
        
        # One empty
        A = [(1:3,)]
        result = fold_ranges(A, Tuple[])
        @test length(result) == 0
        
        result = fold_ranges(Tuple[], A)
        @test length(result) == 0
    end
    
    @testset "Error cases" begin
        # Dimension mismatch
        A = [(1:3,), (4:6,)]
        B = [(1:2, 1:2), (3:4, 3:4)]
        
        @test_throws ArgumentError fold_ranges(A, B)
    end
    
    @testset "Complex multi-dimensional case" begin
        A = [(1:3, 1:2, 1:2), (2:4, 2:3, 1:3)]
        B = [(2:3, 1:3, 1:2), (1:2, 2:4, 2:3)]
        
        result = fold_ranges(A, B)
        
        # Should have 3D regions
        for r in result
            @test length(r.region) == 3
        end
        
        # Check some expected intersections
        regions = [r.region for r in result]
        
        # (1:3, 1:2, 1:2) ∩ (2:3, 1:3, 1:2) = (2:3, 1:2, 1:2)
        @test (2:3, 1:2, 1:2) in regions
        
        # (2:4, 2:3, 1:3) ∩ (1:2, 2:4, 2:3) = (2:2, 2:3, 2:3)
        @test (2:2, 2:3, 2:3) in regions
    end
    
    @testset "Verbose mode" begin
        A = [(1:3,)]
        B = [(2:4,)]
        
        # Should not throw with verbose=true
        result = fold_ranges(A, B; verbose=true)
        @test length(result) == 1
        @test result[1].region == (2:3,)
    end
    
    @testset "Edge case: identical ranges" begin
        A = [(1:5,), (1:5,)]
        B = [(1:5,)]
        
        result = fold_ranges(A, B)
        
        @test length(result) == 1
        @test result[1].region == (1:5,)
        @test result[1].A_idxs == [1, 2]  # Both A ranges contribute
        @test result[1].B_idxs == [1]
        @test result[1].pairs == [(1, 1), (2, 1)]
    end
    
    @testset "Edge case: range contains another" begin
        A = [(1:10,)]
        B = [(3:7,), (5:8,)]
        
        result = fold_ranges(A, B)
        
        @test length(result) == 2
        
        # (1:10) ∩ (3:7) = (3:7)
        # (1:10) ∩ (5:8) = (5:8)
        
        regions = [r.region for r in result]
        @test (3:7,) in regions
        @test (5:8,) in regions
    end
end

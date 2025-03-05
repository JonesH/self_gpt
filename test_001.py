from problem_001 import sum_of_multiples

def test_sum_of_multiples():
    # Test case from the problem description
    assert sum_of_multiples(10) == 23, "Test case for limit=10 failed"

    print("All test cases passed!")

# Run the test
test_sum_of_multiples()

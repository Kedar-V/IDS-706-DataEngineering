# -------------------------------
# Import functions to test
# -------------------------------
from hello import say_hello, add, hello_world


# -------------------------------
# Test: say_hello function
# -------------------------------
def test_say_hello():
    """
    Test the say_hello function with normal and edge case inputs.
    """
    # Test multiple typical names
    names = ['Kedar', 'Sam']
    for name in names:
        # Check that the output matches the expected greeting
        assert (
            say_hello(name)
            == f"Hello, My name is {name}, and I am excited to join "
            "Data Engineering Systems (IDS 706)!"
        )

    # Edge case: empty string as name
    assert say_hello("") == (
        "Hello, My name is , and I am excited to join "
        "Data Engineering Systems (IDS 706)!"
    )


# -------------------------------
# Test: add function
# -------------------------------
def test_add():
    """
    Test the add function with normal and edge case inputs.
    """
    # Normal case
    assert add(2, 3) == 5

    # Large numbers
    assert add(1_000_000, 2_000_000) == 3_000_000

    # Negative and positive numbers
    assert add(-100, 100) == 0


# -------------------------------
# Test: hello_world function
# -------------------------------
def test_hello_world_intro():
    """
    Test hello_world function without timestamp.
    Ensures formatted introduction matches the expected string.
    """
    result = hello_world(
        name="Kedar Vaidya",
        role="Data Scientist",
        skills=["Computer Vision", "NLP", "Machine Learning"],
        education="a Master’s degree in AI in the U.S.",
        timestamp=False,
    )

    # Expected output string
    expected = (
        "Hello, my name is Kedar Vaidya. 👋\n"
        "I am a Data Scientist with expertise in Computer Vision, NLP, "
        "Machine Learning.\n"
        "Currently, I am pursuing a Master’s degree in AI in the U.S."
    )

    # Assert that the function output matches expected
    assert result == expected

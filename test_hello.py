from hello import say_hello, add, hello_world
# import subprocess
# import sys
# import os


def test_say_hello():
    names = ['Kedar', 'Sam']
    for name in names:
        assert (
            say_hello(name)
            == f"Hello, My name is {name}, and I am excited to join Data Engineering Systems (IDS 706)!"
        )
        assert (
            say_hello(name)
            == f"Hello, My name is {name}, and I am excited to join Data Engineering Systems (IDS 706)!"
        )
    # Edge cases
    assert say_hello("") == "Hello, My name is , and I am excited to join Data Engineering Systems (IDS 706)!"


def test_add():
    assert add(2, 3) == 5
    # Edge cases
    assert add(1_000_000, 2_000_000) == 3_000_000
    assert add(-100, 100) == 0


from hello import hello_world


def test_hello_world_intro():
    result = hello_world(
        name="Kedar Vaidya",
        role="Data Scientist",
        skills=["Computer Vision", "NLP", "Machine Learning"],
        education="a Master’s degree in AI in the U.S.",
        timestamp=False,
    )

    expected = (
        "Hello, my name is Kedar Vaidya. 👋\n"
        "I am a Data Scientist with expertise in Computer Vision, NLP, Machine Learning.\n"
        "Currently, I am pursuing a Master’s degree in AI in the U.S."
    )

    assert result == expected
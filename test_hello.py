from hello import say_hello, add
# import subprocess
# import sys
# import os


def test_say_hello():
    assert (
        say_hello("Annie")
        == "Hello, Annie, welcome to Data Engineering Systems (IDS 706)!"
    )
    assert (
        say_hello("Sam") == "Hello, Sam, welcome to Data Engineering Systems (IDS 706)!"
    )
    # Edge cases
    assert say_hello("") == "Hello, , welcome to Data Engineering Systems (IDS 706)!"


def test_add():
    assert add(2, 3) == 5
    # Edge cases
    assert add(1_000_000, 2_000_000) == 3_000_000
    assert add(-100, 100) == 0


# def test_main_block(capsys):
#     # Run the main block by executing the script as a subprocess
#     result = subprocess.run(
#         [sys.executable, os.path.join(os.path.dirname(__file__), "hello.py")],
#         capture_output=True,
#         text=True,
#         check=True,
#     )
#     output = result.stdout.strip().splitlines()
#     assert (
#         output[0] == "Hello, Prof. Yu, welcome to Data Engineering Systems (IDS 706)!"
#     )
#     assert output[1] == "2 + 3 = 5"

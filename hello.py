from datetime import datetime
from typing import List


# -------------------------------
# Function: say_hello
# -------------------------------
def say_hello(name: str) -> str:
    """
    Return a greeting message to students in the IDS class.

    Args:
        name (str): The name of the student.

    Returns:
        str: A greeting message including the student's name.
    """
    # Using f-strings to format a multi-line greeting
    return (
        f"Hello, My name is {name}, and I am excited to join Data "
        f"Engineering Systems (IDS 706)!"
    )


# -------------------------------
# Function: add
# -------------------------------
def add(a: int, b: int) -> int:
    """
    Return the sum of two numbers.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: The sum of a and b.
    """
    return a + b


# -------------------------------
# Function: hello_world
# -------------------------------
def hello_world(
    name: str,
    role: str,
    skills: List[str],
    education: str,
    timestamp: bool = False,
) -> str:
    """
    Generate a formatted self-introduction message.

    Args:
        name (str): Your full name.
        role (str): Current or target role (e.g., Data Scientist).
        skills (List[str]): List of key skills to highlight.
        education (str): Education background.
        timestamp (bool, optional): Include a timestamp. Default to False.

    Returns:
        str: A formatted introduction message.
    """
    # Build the introduction string with name, role, skills, and education
    intro = (
        f"Hello, my name is {name}. 👋\n"
        f"I am a {role} with expertise in "
        f"{', '.join(skills)}.\n"
        f"Currently, I am pursuing {education}"
    )

    # Optionally append a timestamp if requested
    if timestamp:
        intro += (
            f"\nGenerated at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        )
    return intro


# -------------------------------
# Main Execution Block
# -------------------------------
if __name__ == "__main__":
    # Print a greeting using the say_hello function
    print(say_hello("Kedar Vaidya"))

    # Print a detailed self-introduction with timestamp
    print(
        hello_world(
            name="Kedar Vaidya",
            role="Data Scientist",
            skills=[
                "Computer Vision",
                "NLP",
                "Machine Learning",
            ],
            education="a Master’s degree in AI in the U.S.",
            timestamp=True,  # Include timestamp in the output
        )
    )

    # Print the result of a simple addition
    print("5 + 7 =", add(5, 7))

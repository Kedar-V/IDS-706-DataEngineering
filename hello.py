def say_hello(name: str) -> str:
    """Return a greeting message to students in the IDS class."""
    return f"Hello, My name is {name}, and I am excited to join Data Engineering Systems (IDS 706)!"

def add(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b

from datetime import datetime
from typing import List

from datetime import datetime
from typing import List

def hello_world(
    name: str,
    role: str,
    skills: List[str],
    education: str,
    timestamp: bool = False
) -> str:
    """
    A Function to Introduce Myself
    """
    print("About Me")
    intro = (
        f"Hello, my name is {name}. 👋\n"
        f"I am a {role} with expertise in {', '.join(skills)}.\n"
        f"Currently, I am pursuing {education}"
    )

    if timestamp:
        intro += f"\nGenerated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

    return intro


if __name__ == "__main__":
    print(say_hello("Kedar Vaidya"))
    print(hello_world(
        name="Kedar Vaidya",
        role="Data Scientist",
        skills=["Computer Vision", "Natural Language Processing", "Machine Learning"],
        education="a Master’s degree in AI in the U.S.",
        timestamp=True
    ))
    print("5 + 7 =", add(5, 7))

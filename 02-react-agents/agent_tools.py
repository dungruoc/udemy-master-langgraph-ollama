from langchain_core.tools import tool
import requests
import re


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.
    Used for querying about weather, temperature, or climate conditions in any city.
    Examples: "weather in Paris", "temperator in Tokyo", "is it raining in London".

    Args:
        location: city name (e.g. "New York", "London", "Hanoi")
    
    Returns:
        Current weather information of the provided location.
    """
    try:
        location = re.sub(r'\s+', '+', location.strip())
        print("[TOOL] get_weather for", location)
        url = f"https://wttr.in/{location}?format=j1"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json()['current_condition']
    except:
        pass

    return "Failed to get weather"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    USE THIS TOOL FOR:
    - Any mathematical calculations or arithmetic operations
    - Queries involving numbers and operators (+, -, *, /, **, %)
    - Questions asking to compute, calculate, or solve math problems
    - Evaluating mathematical expressions
    
    EXAMPLE QUERIES:
    - "What is 2 + 2? so call calculate with expression '2 + 2'"
    - "Calculate 15 times 7. so call calculate with expression '15 * 7'"
    - "Solve 100 / 4. so call calculate with expression '100 / 4'"
    - "What's 5 to the power of 3? so call calculate with expression '5 ** 3'"
    - "Compute 45 * 12 + 30. so call calculate with expression '45 * 12 + 30'"
    
    DO NOT USE FOR:
    - Non mathematical question.
    - Word problems without explicit expressions (extract the math first)
    - Questions about mathematical concepts or theory

    Args:
        expression: Mathemathical expression in Python code, like "2 + 2" or "15 * 7"
    """

    try:
        result = eval(expression)
        print(f"[TOOL] calculate ('{expression}') -> '{result}'")
    except Exception as e:
        print(f"Exception has occured with error: {e}")
        return f"Exception has occured with error: {e}"

    return result

if __name__ == "__main__":
    print(get_weather.invoke({"location": "San  Francisco "}))
    print(calculate.invoke({"expression": "45 * 12 + 30"}))
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
import json

console = Console(markup=True)


def pprint(content, title=None, is_json=False, **kwargs):
    if is_json:
        content = JSON(json.dumps(content))
    if title is not None:
        content = Panel(content, title=title)
    console.print(content, **kwargs)


if __name__ == "__main__":
    x = [{"a": 1, "b": 2, "c": 3}]
    pprint(x, is_json=True)

    x = "Hello, world!"
    pprint(x)
    pprint(x, title="Hello")

    x = 123
    pprint(x)

    x = [1, 2, 3]
    pprint(x)

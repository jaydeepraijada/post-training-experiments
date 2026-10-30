import openai
import time
import dotenv
import random

dotenv.load_dotenv()
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
import json
import sys
from print_utils import pprint
from rich.rule import Rule

data = json.load(open(sys.argv[1]))
model_name = sys.argv[2]
correct_str = "[bold green] Correct! [/bold green]"
incorrect_str = "[bold red] Incorrect! [/bold red]"

c = 0
for i in range(20):
    question = data[i]["question"]
    response = data[i]["response"]
    answer = data[i]["answer"]

    correctness = data[i]["correctness"]
    if correctness != 1:
        if random.uniform(0, 1) > 0:
            continue
    if correctness > 10:
        break
    c += 1
    console = Console()
    console.clear()
    console.print(Rule(f"Example: {c}"))
    # Reasoning

    if "Here is the question" in question:
        question = question.split("Here is the question:")[1]
    pprint(question, title="Question")
    reasoning_text = Text("", justify="left")
    reasoning_panel = Panel(
        reasoning_text,
        title=f"[bold green]{model_name} Reasoning[/bold green]",
        border_style="green",
    )

    with Live(
        reasoning_panel, console=console, screen=False, refresh_per_second=10
    ) as live:
        for r in response.split():
            time.sleep(0.02)
            if r == "<answer>":
                break
            reasoning_text.append(r + " ", style="italic dim")

    # Response
    response_text = Text("", justify="left")
    response_panel = Panel(
        response_text,
        title=f"[bold magenta]{model_name} Answer[/bold magenta]",
        border_style="magenta",
    )

    with Live(
        response_panel, console=console, screen=False, refresh_per_second=10
    ) as live:
        for r in answer.split():
            time.sleep(0.1)
            response_text.append(r + " ")

    reward_str = correct_str if correctness == 1 else incorrect_str
    pprint(reward_str, title="Reward")
    time.sleep(2)
    console.clear()

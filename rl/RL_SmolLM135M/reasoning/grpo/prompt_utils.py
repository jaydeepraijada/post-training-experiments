system_prompt = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags. After </think>,
provide only the final answer with no additional commentary or formatting.

Format:
<think> reasoning process here </think>
final answer here

Important:
- Generate your reasoning inside <think> and </think> tags.
- After </think>, output ONLY the answer — no Markdown fences, no extra text.
- Your answer after </think> will be scored against the ground truth.
- Failing to follow this format will result in a penalty.
"""

assistant_prefix_prompt = "<think>"

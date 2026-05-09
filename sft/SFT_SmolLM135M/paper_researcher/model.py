from __future__ import annotations

from .backends import load_backend, Backend
from .tasks import (
    SYSTEM_PROMPT,
    BULLETS_INSTRUCTION,
    QA_PAIRS_INSTRUCTION,
    QUESTION_FROM_PASSAGE_INSTRUCTION,
    FACT_FROM_PASSAGE_INSTRUCTION,
    QA_ANSWER_INSTRUCTION,
    REPHRASE_INSTRUCTION,
    CONTINUATION_INSTRUCTION,
    TRIPLETS_INSTRUCTION,
    COMPARISON_INSTRUCTION,
    RETRIEVAL_INSTRUCTION,
    build_qa_answer_input,
    build_comparison_input,
    build_retrieval_input,
)
from .parsing import parse_bullets, parse_qa_pairs, parse_triplets, parse_retrieval
from .types import QAPair, Triplet, RetrievalResult


class PaperResearcher:
    def __init__(self, model_path: str, mlx: bool = False):
        self._backend: Backend = load_backend(model_path, mlx=mlx)

    def _build_prompt(self, instruction: str, user_input: str) -> str:
        tokenizer = self._backend.tokenizer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\n{user_input}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _preprocess(self, text: str) -> str:
        return " ".join(text.split())

    def _run(self, instruction: str, user_input: str, **kwargs) -> str:
        prompt = self._build_prompt(instruction, self._preprocess(user_input))
        return self._backend.generate(prompt, **kwargs)

    def extract_bullets(self, passage: str, **kwargs) -> list[str]:
        return parse_bullets(self._run(BULLETS_INSTRUCTION, passage, **kwargs))

    def generate_qa_pairs(self, passage: str, **kwargs) -> list[QAPair]:
        return parse_qa_pairs(self._run(QA_PAIRS_INSTRUCTION, passage, **kwargs))

    def generate_question(self, passage: str, **kwargs) -> str:
        return self._run(QUESTION_FROM_PASSAGE_INSTRUCTION, passage, **kwargs)

    def extract_fact(self, passage: str, **kwargs) -> str:
        return self._run(FACT_FROM_PASSAGE_INSTRUCTION, passage, **kwargs)

    def answer(self, question: str, passage: str, **kwargs) -> str:
        user_input = build_qa_answer_input(self._preprocess(passage), self._preprocess(question))
        return self._backend.generate(self._build_prompt(QA_ANSWER_INSTRUCTION, user_input), **kwargs)

    def rephrase(self, passage: str, **kwargs) -> str:
        return self._run(REPHRASE_INSTRUCTION, passage, **kwargs)

    def continue_from(self, passage_start: str, **kwargs) -> str:
        return self._run(CONTINUATION_INSTRUCTION, passage_start, **kwargs)

    def extract_triplets(self, passage: str, **kwargs) -> list[Triplet]:
        return parse_triplets(self._run(TRIPLETS_INSTRUCTION, passage, **kwargs))

    def compare(self, passage_a: str, passage_b: str, **kwargs) -> str:
        user_input = build_comparison_input(self._preprocess(passage_a), self._preprocess(passage_b))
        return self._backend.generate(self._build_prompt(COMPARISON_INSTRUCTION, user_input), **kwargs)

    def find_relevant(self, question: str, passages: list[str], **kwargs) -> RetrievalResult:
        user_input = build_retrieval_input(self._preprocess(question), [self._preprocess(p) for p in passages])
        raw = self._backend.generate(self._build_prompt(RETRIEVAL_INSTRUCTION, user_input), **kwargs)
        return parse_retrieval(raw, num_passages=len(passages))

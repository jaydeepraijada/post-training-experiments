import ast
import json
from dataclasses import dataclass
from typing import Any, Literal

from markdown_it import MarkdownIt


OutputKind = Literal["json", "python_list", "markdown", "prose"]

_MARKDOWN = MarkdownIt("commonmark")


@dataclass(frozen=True)
class OutputSchema:
    kind: OutputKind
    shape: Any
    structurally_valid: bool = True


@dataclass(frozen=True)
class SchemaMatch:
    reference: OutputSchema
    response: OutputSchema
    datatype_matches: bool
    schema_matches: bool


def _json_value(text: str) -> tuple[bool, Any]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False, None
    return True, value


def _python_list(text: str) -> list[Any] | None:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, list) else None


def _value_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return (
            "object",
            tuple(
                sorted(
                    (str(key), _value_shape(item))
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, list):
        return (
            "array",
            frozenset(_value_shape(item) for item in value),
        )
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _normalized_items(values: list[Any]) -> set[str]:
    return {
        json.dumps(value, sort_keys=True, ensure_ascii=True, default=repr)
        for value in values
    }


def _strong_labels(inline_token) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    children = inline_token.children or []
    for index, child in enumerate(children):
        if child.type != "strong_open":
            continue
        text_parts = []
        for nested in children[index + 1:]:
            if nested.type == "strong_close":
                break
            if nested.type == "text":
                text_parts.append(nested.content)
        label = "".join(text_parts).strip().rstrip(":").casefold()
        if label in {"question", "answer"}:
            trailing_text = []
            after_close = False
            for nested in children[index + 1:]:
                if nested.type == "strong_close":
                    after_close = True
                    continue
                if after_close and nested.type in {"text", "code_inline"}:
                    trailing_text.append(nested.content)
            labels[label] = bool("".join(trailing_text).strip())
    return labels


def _markdown_list_items(tokens) -> list[str]:
    items: list[str] = []
    current: list[str] | None = None
    for token in tokens:
        if token.type == "list_item_open":
            current = []
        elif token.type == "list_item_close" and current is not None:
            items.append(" ".join(current).strip().casefold())
            current = None
        elif token.type == "inline" and current is not None:
            current.append(token.content)
    return items


def _markdown_schema(text: str) -> tuple[tuple[Any, ...] | None, bool]:
    tokens = _MARKDOWN.parse(text)
    token_types = [token.type for token in tokens]
    list_items = token_types.count("list_item_open")
    list_kinds = frozenset(
        kind
        for kind, token_type in (
            ("unordered_list", "bullet_list_open"),
            ("ordered_list", "ordered_list_open"),
        )
        if token_type in token_types and list_items > 1
    )
    labels: dict[str, bool] = {}
    has_unlabeled_top_level_paragraph = False
    for token in tokens:
        if token.type == "inline":
            token_labels = _strong_labels(token)
            labels.update(token_labels)
            if token.level == 1 and not token_labels:
                has_unlabeled_top_level_paragraph = True

    heading_levels = frozenset(
        token.tag for token in tokens if token.type == "heading_open"
    )
    root_blocks = frozenset(
        token.type.removesuffix("_open")
        for token in tokens
        if token.nesting == 1 and token.level == 0
    )
    has_question_answer = {"question", "answer"}.issubset(labels)
    features = (
        ("lists", list_kinds),
        ("question_answer", has_question_answer),
        ("heading_levels", heading_levels),
        ("fenced_code", "fence" in token_types),
        ("blockquote", "blockquote_open" in token_types),
        ("root_blocks", root_blocks),
        ("unlabeled_paragraph", has_unlabeled_top_level_paragraph),
    )
    has_explicit_markdown = bool(
        list_kinds
        or has_question_answer
        or heading_levels
        or "fence" in token_types
        or "blockquote_open" in token_types
    )
    list_values = _markdown_list_items(tokens)
    lists_are_valid = not list_kinds or (
        len(list_values) > 1 and len(set(list_values)) > 1
    )
    qa_is_valid = not has_question_answer or all(
        labels[label] for label in ("question", "answer")
    )
    return (
        features if has_explicit_markdown else None,
        lists_are_valid and qa_is_valid,
    )


def infer_output_schema(text: str) -> OutputSchema:
    stripped = str(text or "").strip()

    is_json, json_value = _json_value(stripped)
    if is_json:
        valid = not isinstance(json_value, list) or (
            len(json_value) > 1
            and len(_normalized_items(json_value)) > 1
        )
        return OutputSchema("json", _value_shape(json_value), valid)

    python_list = _python_list(stripped)
    if python_list is not None:
        valid = (
            len(python_list) > 1
            and len(_normalized_items(python_list)) > 1
        )
        return OutputSchema("python_list", _value_shape(python_list), valid)

    markdown_shape, markdown_valid = _markdown_schema(stripped)
    if markdown_shape is not None:
        return OutputSchema("markdown", markdown_shape, markdown_valid)

    return OutputSchema("prose", "nonempty" if stripped else "empty")


def compare_output_schema(reference: str, response: str) -> SchemaMatch:
    reference_schema = infer_output_schema(reference)
    response_schema = infer_output_schema(response)
    datatype_matches = reference_schema.kind == response_schema.kind
    schema_matches = (
        datatype_matches
        and reference_schema.structurally_valid
        and response_schema.structurally_valid
        and reference_schema.shape == response_schema.shape
    )
    return SchemaMatch(
        reference=reference_schema,
        response=response_schema,
        datatype_matches=datatype_matches,
        schema_matches=schema_matches,
    )

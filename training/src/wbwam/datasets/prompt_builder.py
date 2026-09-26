from typing import Optional

DEFAULT_PROMPT = "A video of a robot doing: {task}"

DEFAULT_DEPLOYMENT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

DEFAULT_EMBODIMENT_PROMPT = "A video of a robot doing: {task}.\nRobot Embodiment: {embodiment}."

DEFAULT_WB_INSTRUCTION_TEMPLATE = "The task is: {task}\n{visual_description}\n{control_description}"


def _normalize_sentence(value: str, *, field_name: str) -> str:
    text = " ".join(str(value).replace("_", " ").split())
    if not text:
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    if text[-1] not in ".!?":
        text += "."
    return text


def build_configured_instruction(
    task: str,
    *,
    visual_description: str,
    control_description: str,
    template: str = DEFAULT_WB_INSTRUCTION_TEMPLATE,
) -> str:
    """Build a natural-language WB instruction from config-owned descriptions."""
    values = {
        "task": _normalize_sentence(task, field_name="task"),
        "visual_description": _normalize_sentence(
            visual_description,
            field_name="visual_description",
        ),
        "control_description": _normalize_sentence(
            control_description,
            field_name="control_description",
        ),
    }
    template_text = str(template).strip()
    if not template_text:
        raise ValueError("`instruction_template` must be a non-empty string.")
    try:
        rendered = template_text.format(**values)
    except KeyError as exc:
        allowed = ", ".join(sorted(values))
        raise ValueError(
            f"Unknown instruction-template placeholder {exc.args[0]!r}; allowed placeholders are: {allowed}."
        ) from exc
    return "\n".join(line.strip() for line in rendered.splitlines() if line.strip())


def build_text_prompt(
    task: str,
    *,
    embodiment_type: Optional[str] = None,
    include_embodiment: bool = False,
) -> str:
    task_text = str(task)
    if not include_embodiment:
        return DEFAULT_PROMPT.format(task=task_text)

    if embodiment_type is None or str(embodiment_type).strip() == "":
        raise ValueError("`embodiment_type` is required when include_embodiment=True.")
    return DEFAULT_EMBODIMENT_PROMPT.format(
        embodiment=str(embodiment_type),
        task=task_text,
    )

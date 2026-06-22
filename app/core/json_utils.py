from __future__ import annotations

import json
import re


def fix_json(text: str) -> str:
    """Repair common LLM JSON formatting issues enough for routing output."""

    value = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        value = fence.group(1).strip()

    first_obj = min([idx for idx in (value.find("{"), value.find("[")) if idx >= 0], default=-1)
    last_obj = max(value.rfind("}"), value.rfind("]"))
    if first_obj >= 0 and last_obj >= first_obj:
        value = value[first_obj : last_obj + 1]

    value = (
        value.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    value = re.sub(r",(\s*[}\]])", r"\1", value)
    value = re.sub(r"([{,]\s*)([A-Za-z_][\w-]*)(\s*:)", r'\1"\2"\3', value)

    try:
        json.loads(value)
        return value
    except json.JSONDecodeError:
        return json.dumps({"content": text}, ensure_ascii=False)

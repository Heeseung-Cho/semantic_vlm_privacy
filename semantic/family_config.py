from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_FAMILY_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'family_category_direct_v1.json'

_ACTIVE_FAMILY_CONFIG_PATH = DEFAULT_FAMILY_CONFIG_PATH
_ACTIVE_FAMILY_CONFIG: dict | None = None


def _normalize_text(text: str) -> str:
    return ' '.join((text or '').strip().replace('_', ' ').lower().split())


def _validate_family_config(payload: dict, path: Path) -> dict:
    families = payload.get('families')
    if not isinstance(families, dict) or not families:
        raise ValueError(f'Invalid family config at {path}: missing non-empty "families" object')
    normalized_names: dict[str, str] = {}
    normalized_families: dict[str, dict[str, object]] = {}
    for family_name, spec in families.items():
        categories: list[str]
        description: str | None = None
        if isinstance(spec, list):
            categories = spec
        elif isinstance(spec, dict):
            categories = spec.get('categories')
            description = spec.get('description')
            if description is not None and not isinstance(description, str):
                raise ValueError(
                    f'Invalid family spec for {family_name!r} at {path}: "description" must be str|null'
                )
        else:
            raise ValueError(
                f'Invalid family spec for {family_name!r} at {path}: expected list[str] or object'
            )
        if not isinstance(categories, list):
            raise ValueError(
                f'Invalid family spec for {family_name!r} at {path}: expected "categories" list[str]'
            )
        normalized_family = _normalize_text(family_name)
        if normalized_family in normalized_names:
            raise ValueError(f'Duplicate normalized family name {family_name!r} in {path}')
        normalized_names[normalized_family] = family_name
        normalized_families[family_name] = {
            'categories': categories,
            'description': description.strip() if isinstance(description, str) else None,
        }
    payload['families'] = normalized_families
    return payload


def load_family_config(path: str | Path | None = None) -> dict:
    config_path = Path(path or DEFAULT_FAMILY_CONFIG_PATH).resolve()
    payload = json.loads(config_path.read_text(encoding='utf-8'))
    payload = _validate_family_config(payload, config_path)
    payload['_config_path'] = str(config_path)
    return payload


def set_active_family_config(path: str | Path | None = None) -> dict:
    global _ACTIVE_FAMILY_CONFIG_PATH, _ACTIVE_FAMILY_CONFIG
    _ACTIVE_FAMILY_CONFIG_PATH = Path(path or DEFAULT_FAMILY_CONFIG_PATH).resolve()
    _ACTIVE_FAMILY_CONFIG = load_family_config(_ACTIVE_FAMILY_CONFIG_PATH)
    return _ACTIVE_FAMILY_CONFIG


def get_active_family_config() -> dict:
    global _ACTIVE_FAMILY_CONFIG
    if _ACTIVE_FAMILY_CONFIG is None:
        _ACTIVE_FAMILY_CONFIG = load_family_config(_ACTIVE_FAMILY_CONFIG_PATH)
    return _ACTIVE_FAMILY_CONFIG


def get_active_family_config_path() -> str:
    return str(get_active_family_config().get('_config_path', DEFAULT_FAMILY_CONFIG_PATH))


def get_family_names() -> list[str]:
    return list(get_active_family_config()['families'].keys())


def canonicalize_family_name(name: str) -> str:
    normalized_name = _normalize_text(name)
    if not normalized_name:
        return ''
    for canonical_name in get_family_names():
        if _normalize_text(canonical_name) == normalized_name:
            return canonical_name
    return name.strip()


def get_family_categories(name: str) -> list[str]:
    canonical_name = canonicalize_family_name(name)
    spec = get_active_family_config()['families'].get(canonical_name, {})
    categories = spec.get('categories', []) if isinstance(spec, dict) else []
    return [str(category) for category in categories if str(category).strip()]


def get_family_description(name: str) -> str | None:
    canonical_name = canonicalize_family_name(name)
    spec = get_active_family_config()['families'].get(canonical_name, {})
    if not isinstance(spec, dict):
        return None
    description = spec.get('description')
    if not isinstance(description, str):
        return None
    stripped = description.strip()
    return stripped or None


def render_family_description_block() -> str:
    lines = []
    for family_name in get_family_names():
        description = get_family_description(family_name)
        if description:
            lines.append(f'- {family_name}: {description}')
    return '\n'.join(lines)


def render_prompt_with_family_config(prompt_text: str) -> str:
    family_list = ', '.join(get_family_names())
    family_descriptions = render_family_description_block()
    if '{{family_list}}' in prompt_text:
        prompt_text = prompt_text.replace('{{family_list}}', family_list)
    if '{{family_descriptions}}' in prompt_text:
        prompt_text = prompt_text.replace('{{family_descriptions}}', family_descriptions)
    pattern = re.compile(r'<family>one .*? chosen from:.*?</family>', flags=re.IGNORECASE | re.DOTALL)
    replacement = f'<family>one mid-level family chosen from: {family_list}</family>'
    if pattern.search(prompt_text):
        prompt_text = pattern.sub(replacement, prompt_text, count=1)
    if '{{family_descriptions}}' in prompt_text:
        prompt_text = prompt_text.replace('{{family_descriptions}}', '')
    prompt_text = re.sub(
        r'\nFamily guidance:\n(?:[ \t]*\n)?(?=\nRules:)',
        '\n',
        prompt_text,
        flags=re.IGNORECASE,
    )
    if pattern.search(prompt_text) or '{{family_list}}' in prompt_text:
        return prompt_text
    return '\n'.join([prompt_text.rstrip(), '', f'Allowed family set: {family_list}'])

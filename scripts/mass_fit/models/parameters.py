"""Load decay-local parameter definitions and create zfit parameters."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from numbers import Real
import operator
from pathlib import Path

import zfit


PARAMETER_CONFIG_DIRECTORY = Path(__file__).with_name("parameter_configs")
AVAILABLE_DECAYS = frozenset(
    {"Bu2JpsiK", "Bd2JpsiKst", "Bs2JpsiKst", "Bs2DsPi"}
)
TAIL_PARAMETER_KEYS = ("alphaL", "nL", "alphaR", "nR")


@dataclass(frozen=True)
class ParameterSpec:
    """Resolved configuration needed to create one zfit parameter."""

    name: str
    value: float
    lower: float | None
    upper: float | None

    def create(self, *, value: float | None = None, floating: bool = True):
        initial_value = self.value if value is None else value
        if self.lower is None and self.upper is None:
            return zfit.Parameter(self.name, initial_value, floating=floating)
        return zfit.Parameter(
            self.name,
            initial_value,
            self.lower,
            self.upper,
            floating=floating,
        )


@lru_cache(maxsize=None)
def load_decay_parameter_config(decay: str) -> dict:
    """Read and minimally validate one decay's parameter configuration."""

    if decay not in AVAILABLE_DECAYS:
        supported = ", ".join(sorted(AVAILABLE_DECAYS))
        raise ValueError(f"Unknown decay {decay!r}; expected one of: {supported}")
    filename = PARAMETER_CONFIG_DIRECTORY / f"{decay}.json"
    with filename.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Parameter configuration {filename} must contain an object")
    if config.get("decay") != decay:
        raise ValueError(
            f"Parameter configuration {filename} declares decay {config.get('decay')!r}"
        )
    return config


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _resolve_number(value, proxies: Mapping[str, float], location: str) -> float:
    """Resolve a number or a restricted arithmetic expression such as ``0.1*n``."""

    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        try:
            expression = ast.parse(value, mode="eval")
        except SyntaxError as error:
            raise ValueError(f"Invalid expression at {location}: {value!r}") from error

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, Real)
                and not isinstance(node.value, bool)
            ):
                return float(node.value)
            if isinstance(node, ast.Name) and node.id in proxies:
                return proxies[node.id]
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
                return _BINARY_OPERATORS[type(node.op)](
                    evaluate(node.left),
                    evaluate(node.right),
                )
            if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
                return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
            raise ValueError(
                f"Unsupported expression at {location}: {value!r}. "
                "Only numbers, n, parentheses, +, -, * and / are allowed."
            )

        result = float(evaluate(expression))
    else:
        raise ValueError(f"Expected a number or expression at {location}, got {value!r}")
    if not math.isfinite(result):
        raise ValueError(f"Expression at {location} did not produce a finite number")
    return result


def _parameter_specs(decay: str, *path: str, n: int | None = None):
    config = load_decay_parameter_config(decay)
    section = config
    for part in path:
        try:
            section = section[part]
        except (KeyError, TypeError) as error:
            dotted_path = ".".join(path)
            raise KeyError(
                f"Missing parameter section {dotted_path!r} in {decay}.json"
            ) from error
    if not isinstance(section, dict):
        dotted_path = ".".join(path)
        raise ValueError(f"Parameter section {dotted_path!r} must be an object")

    proxies = {} if n is None else {"n": float(n)}
    specs = {}
    for key, definition in section.items():
        location = f"{decay}.{'.'.join(path)}.{key}"
        if not isinstance(definition, dict):
            raise ValueError(f"Parameter definition at {location} must be an object")
        name = definition.get("name", key)
        if not isinstance(name, str) or not name:
            raise ValueError(f"Parameter at {location} must have a non-empty name")
        if "start" not in definition or "limits" not in definition:
            raise ValueError(f"Parameter at {location} must define start and limits")
        start = _resolve_number(definition["start"], proxies, f"{location}.start")
        limits = definition["limits"]
        if limits is None:
            lower = upper = None
        elif isinstance(limits, list) and len(limits) == 2:
            lower = _resolve_number(limits[0], proxies, f"{location}.limits[0]")
            upper = _resolve_number(limits[1], proxies, f"{location}.limits[1]")
            if lower > upper:
                raise ValueError(f"Lower limit exceeds upper limit at {location}")
            if not lower <= start <= upper:
                raise ValueError(f"Start value lies outside the limits at {location}")
        else:
            raise ValueError(f"Limits at {location} must be null or a two-item list")
        specs[key] = ParameterSpec(name, start, lower, upper)
    return specs


def _create_parameters(specs, overrides=None):
    overrides = {} if overrides is None else dict(overrides)
    parameters = {
        key: overrides[key] if key in overrides else spec.create()
        for key, spec in specs.items()
    }
    parameters.update({key: value for key, value in overrides.items() if key not in specs})
    return parameters


def signal_parameter_specs(decay: str):
    return _parameter_specs(decay, "signal")


def create_signal_parameters(decay: str, overrides=None):
    return _create_parameters(signal_parameter_specs(decay), overrides)


def create_background_parameters(decay: str, is_selected: bool, overrides=None):
    selection = "selected" if is_selected else "unselected"
    specs = _parameter_specs(decay, "background", selection)
    return _create_parameters(specs, overrides)


def create_component_parameter(decay: str, key: str):
    return _parameter_specs(decay, "components")[key].create()


def create_yield_parameter(decay: str, key: str, n: int):
    return _parameter_specs(decay, "yields", n=n)[key].create()


def load_fixed_signal_parameters(
    filename: str | Path,
    decay: str,
    keys: tuple[str, ...] | None = None,
    *,
    require_error: bool = True,
):
    """Create fixed signal parameters from a fit-result JSON file."""

    with Path(filename).open(encoding="utf-8") as parameter_file:
        payload = json.load(parameter_file)
    if not isinstance(payload, dict):
        raise ValueError("Shape-parameter JSON must contain an object at its root")

    specs = signal_parameter_specs(decay)
    requested_keys = tuple(specs) if keys is None else keys
    unknown_keys = set(requested_keys).difference(specs)
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown shape parameters for {decay}: {unknown}")

    fixed_parameters = {}
    missing = []
    for key in requested_keys:
        spec = specs[key]
        json_key = key if key in payload else spec.name
        if json_key not in payload:
            missing.append(f"{key} (or {spec.name})")
            continue
        entry = payload[json_key]
        required_fields = {"value", "error"} if require_error else {"value"}
        if not isinstance(entry, dict) or not required_fields.issubset(entry):
            raise ValueError(
                f"Shape parameter {json_key!r} must contain "
                + " and ".join(repr(field) for field in sorted(required_fields))
            )
        if isinstance(entry["value"], bool) or not isinstance(entry["value"], Real):
            raise ValueError(f"Shape parameter {json_key!r} must have a numeric value")
        if require_error and (
            isinstance(entry["error"], bool) or not isinstance(entry["error"], Real)
        ):
            raise ValueError(f"Shape parameter {json_key!r} must have a numeric error")
        fixed_parameters[key] = spec.create(
            value=float(entry["value"]),
            floating=False,
        )

    if missing:
        raise ValueError(
            f"Missing shape parameters for {decay}: {', '.join(missing)}"
        )
    return fixed_parameters

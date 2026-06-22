# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables for the speech audio OpenAI entrypoint."""

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    VLLM_OMNI_AUDIO_SPEECH_DEFAULT_STREAM_FORMAT: bool = False

logger = logging.getLogger(__name__)
_warned_invalid_envs: set[tuple[str, str]] = set()
_AUDIO_SPEECH_DEFAULT_STREAM_FORMAT = "VLLM_OMNI_AUDIO_SPEECH_DEFAULT_STREAM_FORMAT"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    warning_key = (name, value)
    if warning_key not in _warned_invalid_envs:
        logger.warning("%s=%s not recognized; falling back to %r", name, value, default)
        _warned_invalid_envs.add(warning_key)
    return default


environment_variables: dict[str, Callable[[], bool]] = {
    _AUDIO_SPEECH_DEFAULT_STREAM_FORMAT: lambda: _bool_env(_AUDIO_SPEECH_DEFAULT_STREAM_FORMAT),
}


def audio_speech_sse_streaming_required() -> bool:
    return environment_variables[_AUDIO_SPEECH_DEFAULT_STREAM_FORMAT]()


def __getattr__(name: str) -> bool:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
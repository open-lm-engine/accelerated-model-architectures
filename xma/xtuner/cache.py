# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

import yaml

from ..utils import get_boolean_env_variable
from .config import XTuneConfig


_LOAD_XTUNE_CACHE = get_boolean_env_variable("LOAD_XTUNE_CACHE", True)
_XTUNE_CACHE_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache.yml")


class _XTuneCache:
    def __init__(self) -> None:
        self.cache = {}

        if _LOAD_XTUNE_CACHE and os.path.exists(_XTUNE_CACHE_FILENAME):
            cache = yaml.load(open(_XTUNE_CACHE_FILENAME, "r"), yaml.SafeLoader)
            self.cache = self._deserialize(cache)

    def add_config(self, function_hash: str, lookup_key: str, config: XTuneConfig) -> None:
        if function_hash not in self.cache:
            self.cache[function_hash] = {}

        self.cache[function_hash][lookup_key] = config

    def get_config(self, function_hash: str, lookup_key: str) -> XTuneConfig:
        if function_hash in self.cache:
            function_cache = self.cache[function_hash]
            return function_cache.get(lookup_key, None)

        return None

    def save(self) -> None:
        yaml.dump(self._serialize(self.cache), open(_XTUNE_CACHE_FILENAME, "w"))

    def _serialize(self, x: dict) -> dict:
        result = {}

        for function_hash in x:
            function_cache = x[function_hash]
            result[function_hash] = {}

            for lookup_key, config in function_cache.items():
                result[function_hash][lookup_key] = {key: value for key, value in config.get_key_values().items()}

        return result

    def _deserialize(self, x: dict) -> dict:
        result = {}

        for function_hash in x:
            function_cache = x[function_hash]
            result[function_hash] = {}

            for lookup_key, config in function_cache.items():
                result[function_hash][lookup_key] = XTuneConfig({key: value for key, value in config.items()})

        return result


_XTUNE_CACHE = None


def get_xtune_cache() -> _XTuneCache:
    global _XTUNE_CACHE
    if _XTUNE_CACHE is None:
        _XTUNE_CACHE = _XTuneCache()

    return _XTUNE_CACHE

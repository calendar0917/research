"""Shim for the two ``chemprop.utils`` symbols used by ``nn/agg.py``."""
from enum import StrEnum
from typing import Iterator


class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name):
        if isinstance(name, cls):
            return name
        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(f"Unsupported {cls.__name__} member! got: '{name}'")

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)


class ClassRegistry:
    def __class_getitem__(cls, item):
        return cls

    @classmethod
    def register(cls, name):
        def _wrap(obj):
            return obj

        return _wrap


def pretty_shape(shape) -> str:
    return "(" + ", ".join(str(s) for s in shape) + ")"

import pytest

from lightprof.reasoning import ReasoningModule

def test_reasoning_module_instantiation():
    module = ReasoningModule()
    assert isinstance(module, ReasoningModule)
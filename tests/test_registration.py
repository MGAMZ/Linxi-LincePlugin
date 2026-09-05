import importlib

import pytest


pytest.importorskip("linxi")

pytest.importorskip("probeinterface")


from linxi.processor import PROCESS_STAGES
from linxi.processor.registry import get_processor_registration_name, get_registered_probe, get_registered_processors


def _import_plugin_package():
    return importlib.import_module("linxi_linceplugin")



def test_processor_is_registered_after_import():
    _import_plugin_package()

    registered_names = [
        get_processor_registration_name(processor_cls)
        for processor_cls in get_registered_processors(PROCESS_STAGES.PREPROCESS)
    ]

    assert "ExamplePluginProcessor" in registered_names




def test_probe_is_registered_after_import():
    _import_plugin_package()
    assert get_registered_probe("example_linear_probe") is not None




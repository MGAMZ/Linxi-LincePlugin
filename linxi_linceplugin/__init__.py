"""Linxi plugin package template.

Importing this package triggers registration of bundled processors and probes.
"""


from .processor import ExamplePluginProcessor


from .probe import build_example_linear_probe


__all__ = [

    "ExamplePluginProcessor",


    "build_example_linear_probe",

]
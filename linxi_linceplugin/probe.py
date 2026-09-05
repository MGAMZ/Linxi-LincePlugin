import probeinterface as pi

from linxi.processor import register_probe_definition


@register_probe_definition("example_linear_probe")
def build_example_linear_probe():
    """Return a minimal example probe definition."""

    probe = pi.generate_linear_probe(num_elec=4)
    probe.name = "example_linear_probe"
    return probe
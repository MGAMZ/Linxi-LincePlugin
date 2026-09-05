from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor


@register_as_linxi_processor(stage=PROCESS_STAGES.PREPROCESS)
class ExamplePluginProcessor(DefaultProcessor):
    """Example Linxi preprocess processor."""

    def _process(self, context):
        return context
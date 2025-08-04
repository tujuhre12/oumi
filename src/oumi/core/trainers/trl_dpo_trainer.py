from trl import DPOTrainer


class TrlDpoTrainer(DPOTrainer):
    """Light wrapper around the DPOTrainer to handle vision models."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the TrlDpoTrainer."""
        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        return dataset

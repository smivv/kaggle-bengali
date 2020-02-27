try:
    import os

    if os.environ.get("USE_WANDB", "1") == "1":
        from catalyst.dl import SupervisedWandbRunner as Runner
        # from catalyst.dl import SupervisedRunner as Runner
    else:
        from catalyst.dl import SupervisedRunner as Runner
except ImportError:
    from catalyst.dl import SupervisedRunner as Runner

from .datasets.bengali import IMAGE_KEY, INPUT_KEYS, OUTPUT_KEYS


class ModelRunner(Runner):
    def __init__(self,
                 model=None,
                 device=None,
                 input_key=IMAGE_KEY,
                 output_key=None,
                 # output_key=tuple(OUTPUT_KEYS),
                 input_target_key=INPUT_KEYS):
        super().__init__(
            model=model,
            device=device,
            input_key=input_key,
            output_key=output_key,
            input_target_key=input_target_key,
        )

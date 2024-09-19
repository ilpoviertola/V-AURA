from models.data.motionformer_gen_dataset import MotionFormerGenDataset
from models.data.motionformer_datamodule import MotionFormerDatamodule


class MotionFormerGenDatamodule(MotionFormerDatamodule):
    DATASET = MotionFormerGenDataset  # type: ignore

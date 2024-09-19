from models.data.motionformer_dataset import MotionFormerDataset
from models.data.vjepa_datamodule import VJEPADatamodule


class MotionFormerDatamodule(VJEPADatamodule):
    DATASET = MotionFormerDataset

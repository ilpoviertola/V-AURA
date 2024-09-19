from models.data.vjepa_datamodule import VJEPADatamodule
from models.data.vjepa_gen_dataset import VJEPAGenDataset


class VJEPAGenDatamodule(VJEPADatamodule):
    DATASET = VJEPAGenDataset

# V-AURA: Temporally Aligned Audio for Video with Autoregression

The official implementation of V-AURA.

[[`Project Page`](https://v-aura.notion.site/)]

## Environment

This code has been tested on Ubuntu 20.04 with Python 3.8.18 and PyTorch 2.2.1 using CUDA 12.1.
To install the required packages, run the following command:

```bash
conda env create -f conda_env_cuda12.1.yaml
conda activate vaura_cu12.1
```

## Demo

We provide a demo notebook (```demo.ipynb```) to generate samples with the pre-trained model. Simply activate the environment and execute the cells in the notebook to generate samples.

## Data

### VGGSound

Download the full [VGGSound dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/). After downloading, you need to reencode the data. You can use the ```./scripts/reencode_videos.py``` to do so. Just substitute the path in the script with the path to your data.

### VAS

You can run inference on VAS (training is not tested). Download the [VAS dataset](https://drive.google.com/file/d/14birixmH7vwIWKxCHI0MIWCcZyohF59g/view). VAS is encoded on-the-fly during inference, so no need to reencode. Just substitute the VAS base path to ```./data/vas/test/data.jsonl``` file entries.

### VisualSound

To use the novel VisualSound dataset, you need to download the VGGSound dataset and reencode the data. You can use the ```./scripts/reencode_videos.py``` for reencoding. The entries included in VisualSound are defined in ```./data/meta/visualsound/visualsound.csv``` and per split in ```./data/splits/visualsound```. For training, testing, and inference, you can provide a path to full VGGSound dataset (```config.dataloader.data_dir```) and a path to a VisualSound splits (```config.dataloader.split_dir```) in V-AURA configuration. Only files defined in the split-files are used.

## Visual Feature Extractor

We employ [Segment AVCLIP](https://github.com/v-iashin/Synchformer) by Iashin et al. for visual feature extraction. You can download the model pre-trained on VGGSound from [here](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/23-12-22T16-10-50/checkpoints/epoch_best.pt). After downloading, you need to specify the path to the model in the configuration file (```./configs/modules/feature_extractors/avclip_vggsound.yaml```) or in command line (see [training](#training)).

## Models

We provide a checkpoint to the model used to calculate the results in the paper. Download the checkpoint [here](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/v-aura_public/24-08-01T08-34-26.tar.gz).

Extract the TAR-file to your log directory (e.g. ./logs) and provide the path to the experiment directory in the command line (see [generation](#generation-inference)).

## Training

To simplify things we use [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). See PyTorch Lightning Trainer arguments from ```./scripts/train.py``` to customize the training process. To run the training with the default settings, run the following command:

```bash
python main.py
    config=./configs/experiments/vggsound/avclip/9cb-viscond-avclip-channel_concat-llama.yaml
    dataloader.data_dir=/path/to/reencoded_data
    trainer.num_nodes=1  # number of nodes
    trainer.devices=[0]  # GPUs, e.g. 4 or [0, 2]
    feature_extractor_config.params.ckpt_path=/path/to/epoch_best.pt
```

If you want to resume training, just provide the path to the last checkpoint in the **experiment directory** (e.g. YY-MM-DDTHH-MM-SS, which contains checkpoint and logging dirs) and the training will continue from there. Also, logging will be appended to the existing TensorBoard logs:

```bash
python main.py
    config=/path/to/experiment/config.yaml
    dataloader.data_dir=/path/to/reencoded_data
    trainer.num_nodes=1  # number of nodes
    trainer.devices=[0]  # GPUs, e.g. 4 or [0, 2]
    feature_extractor_config.params.ckpt_path=/path/to/epoch_best.pt
    trainer.ckpt_path=/path/to/experiment/last.ckpt
```

By default, TensorBoard is used to log the progress. Start TensorBoard with the following command:

```bash
tensorboard --logdir=./logs
```

## Generation (inference)

To generate VGGSound/VGGSound-Sparse/VisualSound/VAS samples with the model, run the following command:

```bash
python main.py
    config=configs/generate_[vgg, vgg_sparse, vas, visualsound].yaml
    experiment_path=/path/to/experiment  # experiment directory generated during training
    overridden_hparams.feature_extractor_config.params.ckpt_path=/path/to/epoch_best.pt  # path to the visual feature extractor ckpt (if different from training)
    duration=2.56  # duration of the generated audio in seconds (n * 0.64)
    dataloader.data_dir=/path/to/reencoded_data
```

## Evaluation

For evaluation use my [evaluation framework](https://github.com/ilpoviertola/eval_generative_v2a_models).

## Acknowledgements

We would like to thank following open-source repositories for their code and documentation:

- [Synchformer](https://github.com/v-iashin/Synchformer)
- [LlamaGen](https://github.com/FoundationVision/LlamaGen)
- [AudioCraft](https://github.com/facebookresearch/audiocraft)
- [SpecVQGAN](https://github.com/v-iashin/SpecVQGAN)
- [PyTorch](https://github.com/pytorch/pytorch)
- [PyTorchLightning](https://github.com/Lightning-AI/pytorch-lightning)
- NumPy, SciPy, and other Python libraries

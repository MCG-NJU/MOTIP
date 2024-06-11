# Installation

Currently, MOTIP is built upon **Python 3.11, PyTorch 2.2 (recommended)**. 

:warning: As far as I know, due to the use of some new language features in our code, Python version 3.10 or higher is required. For PyTorch, because there have been changes in the type requirements for attention masks, PyTorch version 2.0 or higher is needed.

:construction: We plan to support lower versions of PyTorch in the future, but the exact timeline is yet to be determined.

## Instructions

```shell
conda create -n MOTIP python=3.11		# suggest to use virtual envs
conda activate MOTIP
# PyTorch:
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia		# CUDA version=12.1 is also OK
# Other dependencies:
conda install matplotlib pyyaml scipy tqdm tensorboard seaborn scikit-learn pandas
pip install opencv-python einops wandb pycocotools timm
# Compile the Deformable Attention:
cd models/ops/
sh make.sh
# After compiled, you can use following script to test it:
python test.py		# [Optional]
```




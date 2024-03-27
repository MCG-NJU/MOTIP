# param 1: model outputs dir
# param 2: model filename
# param 3: eval dataset
# param 4: eval split
# param 5: num gpus
python -m torch.distributed.run --nproc_per_node="$5" main.py --mode submit --config-path "$1"train/config.yaml --use-distributed True --inference-config-path "$1"train/config.yaml --inference-model "$1""$2" --outputs-dir "$1" --use-wandb False --inference-dataset "$3" --inference-split "$4" "${@:6}"
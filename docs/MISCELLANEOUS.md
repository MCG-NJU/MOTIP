# MISCELLANEOUS

Some other settings that are unrelated to the specific model structure or hyperparameters. Such as logging, checkpoint saving, *etc*.

## Logging

:hugs: In our codebase, many logging methods are integrated, including tensorboard/wandb/*etc*.

In our default scripts, we set `use_wandb` to `False` to disable wandb logging, because it requires creating an account and making some additional settings, which increases the user's workload. However, if you believe you need to enable wandb logging (which I think is more elegant), you will need to set it up as follows:

1. add `--use-wandb True` to the script.
2. set the `EXP_OWNER` or `--exp-owner`, which is the wandb account name.
3. manually record your current git revision number (`--git-version`, for example, `--git-version d293ceee8c6d208bb4a5d7b6ba92a7b5d7ec4bca`) to ensure you can rollback and reproduce the experiment. I know this is not very elegant, but it is easy enough :stuck_out_tongue:.


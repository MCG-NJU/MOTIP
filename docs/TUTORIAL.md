# Tutorial
In this tutorial, we aim to provide simple, easy-to-understand explanations and guidance to help you better understand, migrate, and improve our model.

## Improvements *vs.* [prev-engine branch](https://github.com/MCG-NJU/MOTIP/tree/prev-engine)
1. We adopt **Accelerate** instead of PyTorch's native DDP framework, making multi-GPU training and mixed-precision training more convenient (though the latter wasn't used in our final experiments).
2. We implement the trajectory augmentation part in data processing (on CPU, together with other image/video augmentation methods), which significantly **improves GPU utilization** during training and accelerates the speed of each iteration.
3. We use different ID assignment groups for the same video clip sample, as specified by the parameter `AUG_NUM_GROUPS: 6`. This can significantly improve data utilization and **accelerate model convergence**.
4. By default, we abandon the Hungarian algorithm in favor of **a simpler runtime ID assignment method** that only selects the highest confidence. We present ablation experiments and explanations in Table 4.

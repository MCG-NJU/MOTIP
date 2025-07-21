# Tutorial
In this tutorial, we aim to provide simple, easy-to-understand explanations and guidance to help you better understand, migrate, and improve our model.

## Improvements *vs.* [prev-engine branch](https://github.com/MCG-NJU/MOTIP/tree/prev-engine)
1. We adopt **Accelerate** instead of PyTorch's native DDP framework, making multi-GPU training and mixed-precision training more convenient (though the latter wasn't used in our final experiments).
2. We implement the trajectory augmentation part in data processing (on CPU, together with other image/video augmentation methods), which significantly **improves GPU utilization** during training and accelerates the speed of each iteration.
3. We use different ID assignment groups for the same video clip sample, as specified by the parameter `AUG_NUM_GROUPS: 6`. This can significantly improve data utilization and **accelerate model convergence**.
4. By default, we abandon the Hungarian algorithm in favor of **a simpler runtime ID assignment method** that only selects the highest confidence. We present ablation experiments and explanations in Table 4.

## Temporal Length (Window)
Like most MOT algorithms, our model only handles target disappearance and re-appearance within a certain tolerance range, which is noted as *T*. If this temporal tolerance is exceeded, even the same target will be assigned a different ID.

Since our model utilizes *long-term sequence training*, *relative temporal position encoding*, and *online inference*, there are some parameters that, together, determine the temporal length. If you need to modify the temporal length we have set, these parameters need to be carefully changed together:
- `SAMPLE_LENGTHS`: The temporal length of the sampled video clip during training.
- `REL_PE_LENGTH`: The max length of the relative temporal position encoding.
- `MISS_TOLERANCE`: The temporal tolerance of re-appear targets during inference.

**A quick and straightforward setting rule is: `SAMPLE_LENGTHS == REL_PE_LENGTH >= MISS_TOLERANCE`.**

## Thresholds
Although our method is end-to-end, we still need some thresholds to control the model's behavior during inference (like DETRs need thresholds to select positive targets). As we decouple the object detection and association processes, the thresholds are also divided into two parts.

1. **Object Detection.** DETR outputs numerous detection results, but we do not need to process all of them in tracking, as this would lead to excessive computational overhead. Therefore, we use the following thresholds to control the process in the current frame:

   1) `DET_THRESH`: A target will only be selected and fed into the ID Decoder if its confidence exceeds this threshold.
   2) `NEWBORN_THRESH`: When a target does not match any historical trajectory, it can be marked as a newborn target only if it exceeds this threshold. This is to make the generation of new trajectories as reliable as possible.

2. **Object Association.** The ID Decoder outputs a probability distribution of a target being assigned to different IDs, so we need a threshold to control the minimum confidence:
   1) `ID_THRESH`: Only when the confidence assigned to an ID is greater than this threshold can it be regarded as a valid allocation.
  

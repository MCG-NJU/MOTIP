# Multiple Object Tracking as ID Prediction

This is the official PyTorch implementation of our paper:

> ***[Multiple Object Tracking as ID Prediction](https://arxiv.org/abs/2403.16848)*** <br>
> :mortar_board: [Ruopeng Gao](https://ruopenggao.com/), Ji Qi, [Limin Wang](https://wanglimin.github.io/) <br>
> :e-mail: Primary contact: ruopenggao@gmail.com

## :mag: Overview

**TL; DR.** We propose a novel perspective to ***regard the multiple object tracking task as an in-context ID prediction problem***. Given a set of trajectories carried with ID information, MOTIP directly decodes the ID labels for current detections, which is straightforward and effective.

![Overview](./assets/overview.png)


## :fire: News

- <span style="font-variant-numeric: tabular-nums;">**2025.03.25**</span>: Our revised paper is released at [arXiv:2403.16848](https://arxiv.org/abs/2403.16848). The latest codebase is being organized :construction:.
- <span style="font-variant-numeric: tabular-nums;">**2025.02.27**</span>: Our paper is accepted by CVPR 2025 :tada: :tada:. The revised paper and a more efficient codebase will be released in March. Almost there :nerd_face: ~
- <span style="font-variant-numeric: tabular-nums;">**2024.03.26**</span>: The first version of our paper is released at [arXiv:2403.16848v1](https://arxiv.org/abs/2403.16848v1) :pushpin:. The corresponding codebase is stored in the [prev-engine branch](https://github.com/MCG-NJU/MOTIP/tree/prev-engine) (No longer maintained starting April 2025 :no_entry:).

## :bouquet: Acknowledgements

This project is built upon [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [MOTR](https://github.com/megvii-research/MOTR), [TrackEval](https://github.com/JonathonLuiten/TrackEval). Thanks to the contributors of these great codebases.

## :pencil2: Citation

If you think this project is helpful, please feel free to leave a :star: and cite our paper:

```tex
@article{MOTIP,
  title={Multiple Object Tracking as ID Prediction},
  author={Gao, Ruopeng and Qi, Ji and Wang, Limin},
  journal={arXiv preprint arXiv:2403.16848},
  year={2024}
}
```

## :star2: Stars

[![Star History Chart](https://api.star-history.com/svg?repos=MCG-NJU/MOTIP&type=Date)](https://star-history.com/#MCG-NJU/MOTIP&Date)

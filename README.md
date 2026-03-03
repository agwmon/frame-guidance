# Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models
> ### TL;DR:
> We present Frame Guidance, a **training-free** framework that supports **diverse control tasks** using **frame-level** signals.

This is an official implementation of paper 'Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models'.

**[ICLR 2026]**- **[Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](https://arxiv.org/abs/2506.07177)**
<br/>
[Sangwon Jang*](https://agwmon.github.io/), [Taekyung Ki*](https://taekyungki.github.io), [Jaehyeong Jo](http://harryjo97.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Soo Ye Kim](https://sites.google.com/view/sooyekim), [Zhe Lin](https://sites.google.com/site/zhelin625/home), [Sungju Hwang](http://www.sungjuhwang.com/)
<br/>(* indicates equal contribution)

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://frame-guidance-video.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2506.07177-b31b1b.svg)](https://arxiv.org/abs/2506.07177)

## Installation
> 2026.02.11: 🚨 There is an installation error with openai-CLIP. Please refer to: https://github.com/openai/CLIP/issues/528.

> 2026.02.12: 🚨 There is a Wan model loading error with `transformers==5.0.0`. Please use `transformers==4.57.3` until this issue is fixed.


Please refer to `setting.sh` for conda environment setup.

## Inference 
|🧩 Task|🔧 Base model|📂 Code|
|---|---|---|
|🎯Keyframe-guided, Color block, Depth, Sketch|CogX-I2V|`keyframe_cogx.ipynb`|
|🎨Stylized, 🔁Loop|CogX-T2V|`others_cogx.ipynb`|
|**Wan2.1 version will be updated!**|
|🎯Keyframe-guided, Color block, Depth, Sketch|Wan-I2V|`keyframe_wan.ipynb`|
|🎨Stylized, 🔁Loop|Wan-T2V|`others_wan.ipynb`|

|Parameter|Description|Default|
|---|---|---|
|`--video`|Input conditions for guidance (List: `[img0, img1, ... imgL]`)|require for I2V|
|`--guidance_lr`|Schedule for guidance **step size** η|`3e0`|
|`--guidance_step`|Schedule for the number of guidance steps M|see `.ipynb` file|
|`--fixed_frames`|Where to apply frame-guidance (e.g., `[25,48]` means apply guidance on 25th and 48th frame)|require|
|`--strength`|V2V strength (It sometimes help converge faster for keyframe guidance)|`0`|
|`--loss_fn`|Loss design for each task [`frame`, `style`, `depth`, `lineart`, `loop` ...]|require|
|`--travel_time`|When we apply time-travel (stochastic) step|CogX: (5, 20), Wan: (3, 10)|

See details in each task-specific examples.

```
@inproceedings{
  jang2026frame,
  title={Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Model},
  author={Sangwon Jang and Taekyung Ki and Jaehyeong Jo and Jaehong Yoon and Soo Ye Kim and Zhe Lin and Sung Ju Hwang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=y39XbEp1vK}
}
```

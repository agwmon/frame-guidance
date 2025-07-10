# Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Model
> ### TL;DR:
> We present Frame Guidance, a **training-free** framework that supports **diverse control tasks** using **frame-level** signals.

This is an official implementation of paper 'Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models'.

**[Arxiv 2025]**- **[Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](https://arxiv.org/abs/2506.07177)**
<br/>
[Sangwon Jang*](https://agwmon.github.io/), [Taekyung Ki*](https://taekyungki.github.io), [Jaehyeong Jo](http://harryjo97.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Soo Ye Kim](https://sites.google.com/view/sooyekim), [Zhe Lin](https://sites.google.com/site/zhelin625/home), [Sungju Hwang](http://www.sungjuhwang.com/)
<br/>(* indicates equal contribution)

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://frame-guidance-video.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2506.07177-b31b1b.svg)](https://arxiv.org/abs/2506.07177)

## Installation
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
@article{jang2025frame,
  title={Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models},
  author={Jang, Sangwon and Ki, Taekyung and Jo, Jaehyeong and Yoon, Jaehong and Kim, Soo Ye and Lin, Zhe and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2506.07177},
  year={2025}
}
```

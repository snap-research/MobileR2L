## MobileR2L<br><sub>Real-Time Neural Light Field on Mobile Devices</sub>

[arXiv](https://arxiv.org/abs/2212.08057) | [PDF](https://arxiv.org/pdf/2212.08057.pdf) | [Web Page](https://snap-research.github.io/MobileR2L/)


<table cellpadding="0" cellspacing="0" >
Shoe Virtual Try-on
  <tr>
    <td  align="center"> <img src="images/shoe-1_record_mp4.gif" width=210px></td>
    <td  align="center"> <img src="images/shoe_1.gif" width=200px></td>
  </tr>
</table>

>[Real-Time Neural Light Field on Mobile Devices](https://snap-research.github.io/MobileR2L/)<br>
>[Junli Cao](https://www.linkedin.com/in/junli-cao-5165b41a1)<sup>1</sup>, [Huan Wang](http://huanwang.tech)<sup>2</sup>, [Pavlo Chemerys](https://www.linkedin.com/in/pashachemerys/)<sup>1</sup>, [Vladislav Shakhrai](https://www.linkedin.com/in/shakhrayv/?originalSubdomain=uk/)<sup>1</sup>, [Ju Hu](https://www.linkedin.com/in/erichuju/)<sup>1</sup>, [Yun Fu](https://coe.northeastern.edu/people/fu-yun/)<sup>2</sup>, <br>[Denys Makoviichuk](https://www.linkedin.com/in/denys-makoviichuk-2219a72b/)<sup>1</sup>, [Sergey Tulyakov](http://www.stulyakov.com/)<sup>1</sup>, [Jian Ren](https://alanspike.github.io/)<sup>1</sup>  
><sup>1</sup>Snap Inc., <sup>2</sup>Northeastern University



<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
 Recent efforts in Neural Rendering Fields (NeRF) have shown impressive results on novel view
                    synthesis by utilizing implicit neural representation to represent 3D scenes.
                    Due to the process of volumetric rendering, the inference speed for NeRF is extremely slow, limiting
                    the application scenarios of utilizing NeRF on resource-constrained hardware, such as mobile
                    devices.
                    Many works have been conducted to reduce the latency of running NeRF models.
                    However, most of them still require high-end GPU for acceleration or extra storage memory, which is
                    all unavailable on mobile devices.
                    Another emerging direction utilizes the neural light field (NeLF) for speedup, as only one forward
                    pass is performed on a ray to predict the pixel color.
                    Nevertheless, to reach a similar rendering quality as NeRF, the network in NeLF is designed with
                    intensive computation, which is not mobile-friendly.
                    In this work, we propose an efficient network that runs in real-time on mobile devices for neural
                    rendering.
                    We follow the setting of NeLF to train our network.
                    Unlike existing works, we introduce a novel network architecture that runs efficiently on mobile
                    devices with low latency and small size, i.e., saving 15x ~ 24x storage
                    compared with MobileNeRF.
                    Our model achieves high-resolution generation while maintaining real-time inference for both
                    synthetic and real-world scenes on mobile devices, e.g., 18.04ms (iPhone 13) for rendering
                    one 1008x756 image of real 3D scenes.
                    Additionally, we achieve similar image quality as NeRF and better quality than MobileNeRF (PSNR
                    26.15 vs. 25.91 on the real-world forward-facing dataset)
</details>


<br>

## Code Coming Soon



## Acknowledgments
This code follows the implementations of [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [R2L](https://github.com/snap-research/R2L). Great thanks to them!

## Citation

If you find the work helps, please cite our paper:
```BibTeX
@article{cao2022real,
  title={Real-Time Neural Light Field on Mobile Devices},
  author={Cao, Junli and Wang, Huan and Chemerys, Pavlo and Shakhrai, Vladislav and Hu, Ju and Fu, Yun and Makoviichuk, Denys and Tulyakov, Sergey and Ren, Jian},
  journal={arXiv preprint arXiv:2212.08057},
  year={2022}
}
```

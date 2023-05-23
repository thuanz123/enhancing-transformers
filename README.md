<div id="top"></div>
<!--
-->

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#training">Training</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## News
***09/09***
1. The release weight of ViT-VQGAN small which is trained on ImageNet at [here](https://drive.google.com/file/d/1jbjD4q0iJpXrRMVSYJRIvM_94AxA1EqJ/view?usp=sharing)

***16/08***
1. First release weight of ViT-VQGAN base which is trained on ImageNet at [here](https://drive.google.com/file/d/1syv0t3nAJ-bETFgFpztw9cPXghanUaM6/view?usp=sharing)
2. Add an colab notebook at [here](https://colab.research.google.com/drive/1y-PzYhkNQbhKj3i459pWd6TAO28SnF5h?usp=sharing)

<!-- ABOUT THE PROJECT -->
## About The Project

This is an unofficial implementation of both [ViT-VQGAN](https://arxiv.org/abs/2110.04627) and [RQ-VAE](https://arxiv.org/abs/2110.04627) in Pytorch. ViT-VQGAN is a simple ViT-based Vector Quantized AutoEncoder while RQ-VAE introduces a new residual quantization scheme. Further details can be viewed in the papers

<!-- GETTING STARTED -->
## Getting Started

For the ease of installation, you should use [anaconda](https://conda.io/) to setup this repo.

### Installation

A suitable conda environment named `enhancing` can be created and activated with:
```
conda env create -f environment.yaml
conda activate enhancing
```

<!-- USAGE EXAMPLES -->
### Training

Training is easy with one line:
   ```python3 main.py -c config_name -lr learning_rate -e epoch_nums```

<!-- ROADMAP -->
## Roadmap

- [x] Add ViT-VQGAN
    - [x] Add ViT-based encoder and decoder
    - [x] Add factorized codes
    - [x] Add l<sub>2</sub>-normalized codes
    - [x] Replace PatchGAN discriminator with StyleGAN one
- [x] Add RQ-VAE
    - [x] Add Residual Quantizer
    - [x] Add RQ-Transformer
- [x] Add dataloader for some common dataset
    - [x] ImageNet
    - [x] LSUN
    - [x] COCO
        - [x] Add COCO Segmentation
        - [x] Add COCO Caption
    - [x] CC3M
- [ ] Add pretrained models
    - [x] ViT-VQGAN small
    - [x] ViT-VQGAN base
    - [ ] ViT-VQGAN large


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Source code and pretrained weights are distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Thuan H. Nguyen - [@leejohnthuan](https://twitter.com/leejohnthuan) - leejohnthuan@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
This project would not be possible without the generous sponsorship from [Stability AI](https://stability.ai/) and helpful discussion of folks in [LAION discord](https://discord.gg/j5GdN49g)

This repo is heavily inspired by following repos and papers:

* [Taming Transformers](https://github.com/CompVis/taming-transformers)
* [ViT-Pytorch](https://github.com/lucidrains/vit-pytorch)
* [minDALL-E](https://github.com/kakaobrain/minDALL-E)
* [CLIP](https://github.com/openai/CLIP)
* [ViT-VQGAN](https://arxiv.org/abs/2110.04627)
* [RQ-VAE](https://arxiv.org/abs/2110.04627)

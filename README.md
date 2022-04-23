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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is an unofficial implementation of both [ViT-VQGAN](https://arxiv.org/abs/2110.04627) and [RQ-VAE](https://arxiv.org/abs/2110.04627) in Pytorch. ViT-VQGAN is a simple ViT-based Vector Quantized AutoEncoder while RQ-VAE propose a new residual qunatization scheme.Further details can be viewed in the papers

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

For the ease of installation, you should use [anaconda](https://conda.io/) to setup this repo.

### Installation

A suitable conda environment named `enhancing` can be created and activated with:
   ```conda env create -f environment.yaml
      conda activate enhancing
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add ViT-VQGAN
- [x] Add RQ-VAE
    - [x] Add Residual Quantizer
    - [x] Add RQ-Transformer
- [ ] Add dataloader for some common dataset
    - [x] ImageNet
    - [x] LSUN
    - [ ] COCO
    - [ ] CC3M
- [ ] Add pretrained models 

<p align="right">(<a href="#top">back to top</a>)</p>



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

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Thuan H. Nguyen - [@leejohnthuan](https://twitter.com/leejohnthuan) - thuan@goback.world

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This repo is heavily inspired by following repos and papers:

* [Taming Transformers](https://github.com/CompVis/taming-transformers)
* [ViT-Pytorch](https://github.com/lucidrains/vit-pytorch)
* [minDALL-E](https://github.com/kakaobrain/minDALL-E)
* [CLIP]()
* [ViT-VQGAN](https://arxiv.org/abs/2110.04627)
* [RQ-VAE](https://arxiv.org/abs/2110.04627)

<p align="right">(<a href="#top">back to top</a>)</p>
- <p align="center">
    <h1 align="center">Squeezing Multiple Cues from a Single Image for Clothed 3D Human Reconstruction</h1>
    <h4 align="center">
      <a href="http://faculty.ccnu.edu.cn/2012980050/"><strong>Leyuan liu</strong></a>
        .
      <a href="http://sunjc0306.github.io"><strong>Jianchi Sun</strong></a>
        .
      <a href="http://sunjc0306.github.io"><strong>Yunqi Gao</strong></a>
        .
      <a href="http://nercel.ccnu.edu.cn/info/1072/3077.htm"><strong>Jingying Chen</strong></a>
      </h4>
    <h3 align="center">National Engineering Research Center for E-Learning, Central China Normal University</h3>
    <h2 align="center">IEEE Transaction on Multimedia</h2>
  </p>

  This repository contains a PyTorch implementation of "Squeezing Multiple Cues from a Single Image for Clothed 3D Human Reconstruction". Tested with PyTorch 1.4.0 on Ubuntu 18.04, CUDA 10.1. 

    <div align="center">
      <img src="./Assets/overview.png" alt="overview" width="100%">
    </div>

  **NOTE**: The current model is trained in a low-resolution dataset called THuman1.0, which contains simple poses and clothes. Fortunately, our method also achieves state-of-the-art performance for some models with complex poses and loose clothes.

  **The code is released. But it's not complete. I'm still updating it.**

  **NOTE 2022/11/25**:  Our paper has been submitted to *IEEE Transaction on Multimedia*. Good luck. This page shows more results generated by our method from Internet images. 

    <div align="center">
      <img src="./Assets/pinterestShow.png" alt="overview" width="100%">
    </div>
    
    
  ## License

  This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).

  ## Citation

  ```bibtex
  @article{MCHuman,
    title     = {Squeezing Multiple Cues from a Single Image for Clothed 3D Human Reconstruction},
    author    = {Liu, Leyuan and Sun, Jianchi and Gao, Yunqi and Chen, Jingying.},
    booktitle = {IEEE Transaction on Multimedia},
  }
  ```

  ## Installation
  Please run the following commands to download the necessary assets (including the pre-trained models):

  Coming Soon！

  ## Run Demo
  In ```test_example```, we provide two examples, which contain an image with a SMPL model fitted to the image and another image from the Internet. 
  ```bash
  python demo.py - input_dir                        
  ```
  ## Train directly the Network using three cues
  Please run the following command to train the network:
  Coming Soon！
  ```bash
  bash 
  ```

  ## Train incrementally the Network by plugging and unplugging 

  Please run the following command to train the network:
  Coming Soon！

  ```bash
  bash 
  ```

  

  ## Acknowledgment

  Note that some code of this repo is heavily based on GeoPIFu, while the source code of squeezing cues is original from PyMAF, pix2pixHD, and openpose . We thank the authors for their great work!


  ## Contact
  - If you have some troubles when running this repo, please do not hesitate to send an E-mail to me (Jianchi Sun [(sunjc0306@qq.com)](mailto:sunjc0306@qq.com)).

<p align="center">
    <h1 align="center">Squeezing Multiple Cues from a Single Image for Clothed 3D Human Reconstruction</h1>
    <h4 align="center">
      <a href="http://faculty.ccnu.edu.cn/2012980050/"><strong>Leyuan Liu</strong></a>
        .
      <a href="http://sunjc0306.github.io"><strong>Jianchi Sun</strong></a>
        .
      <a href="http://sunjc0306.github.io"><strong>Yunqi Gao</strong></a>
        .
      <a href="http://nercel.ccnu.edu.cn/info/1072/3077.htm"><strong>Jingying Chen</strong></a>
      </h4>
    <h3 align="center">National Engineering Research Center for E-Learning, Central China Normal University</h3>
    <h2 align="center">Submitted to IEEE Transaction on Multimedia</h2>
  </p>

This repository contains a PyTorch implementation of "Squeezing Multiple Cues from a Single Image for Clothed 3D Human Reconstruction". 

<img src="./Assets/overview.png" alt="overview" width="100%">

  **NOTE**: The current model is trained in a low-resolution dataset called THuman1.0, which contains simple poses and clothes. Fortunately, our method achieves state-of-the-art performance on human images with complex poses and loose clothes.

  **The code is released, but not complete. We are still updating it.**

  **NOTE 2022/11/25**:  Our paper has been submitted to *IEEE Transaction on Multimedia*. Good luck. This page shows more results generated by our method on in-the-wild images collected from the Internet. 

  **NOTE 2022/12/17**:  The core code and pre-trained models have been uploaded. They can be downloaded from [here](https://pan.baidu.com/s/16b_Pmu0HVW_QxnP9P0ms1g?pwd=tj6r). Install instructions and commands for training the network (directly or incrementally) will be updated and uploaded soon. 
<img src="./Assets/pinterestShow.png" alt="overview" width="100%">

  ## License

  Code and pre-trained models are available for non-commercial research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).

  ## Citation

  ```bibtex
  @article{MCHuman,
    title     = {Squeezing Multiple Cues from a Single Image for Clothed 3D Human Reconstruction},
    author    = {Liu, Leyuan and Sun, Jianchi and Gao, Yunqi and Chen, Jingying.},
  }
  ```

  ## Environments
  This code has been tested with PyTorch 1.4.0 and CUDA 10.1.  on Ubuntu 18.04.


  ## Installation
  Please run the following commands to download the necessary assets (including the pre-trained models):

  Coming Soon！

You can download our provided pre-trained models from  [Download link](链接：https://pan.baidu.com/s/16b_Pmu0HVW_QxnP9P0ms1g?pwd=tj6r) and put them into the `./checkpoints` directory. 



  ## Run Demo
(1) Testing on a single image without SMPL model
 ```bash
python Demo.py -- input_dir img_path
 ```
e.g., python Demo.py -- input_dir ./img.png

(2) Testing on a single image with the ground-truth SMPL model
 ```bash
python Demo.py -- input_dir img_path -- SMPL_dir gtSMPL_path
 ```
e.g., python Demo.py -- input_dir ./img.png -- SMPL_dir ./gtSMPL.pkl


  ## Train the network directly using all cues
  Please run the following command to train the network:
  Coming Soon！
  ```bash
  bash 
  ```

  ## Train the network incrementally by plugging and unplugging cues 

  Please run the following command to train the network:
  Coming Soon！

  ```bash
  bash 
  ```

  

  ## Acknowledgments

  Note that some code of this repo is based on GeoPIFu, PyMAF, pix2pixHD, and OpenPose. We thank the authors for their great work!


  ## Contact
  - If you have any trouble when using this repo, please do not hesitate to send an E-mail to Jianchi Sun (sunjc0306@qq.com).

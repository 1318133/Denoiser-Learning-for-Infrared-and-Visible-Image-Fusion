# Denoiser-Learning-for-Infrared-and-Visible-Image-Fusion
Deno-Fusion: Denoiser Learning for Infrared and Visible Image Fusion

![2](https://github.com/user-attachments/assets/9dbd3c9d-be43-4725-a3b7-004b0bcf51c3)
Pipeline of the proposed method. An efficient fusion network is used to generate fusion images in the generator, which is constrained by semantic adaptive measurement loss, intensity loss, and structural loss functions. In addition, the fusion image is injected with the noise of different intensities and input into different denoising networks (Denoiser-ir, Denoiser-vi) to restore them to images from different sources.
![3](https://github.com/user-attachments/assets/5fda95cc-a6d1-4bb0-a1dc-33d08cd84b58)
Overview of the proposed method

# Train 
Change the arg"Test" as "False". 

# Test
Change the arg"Test" as "Ture". 

# Weights DownLoad 
Google Drive: https://drive.google.com/file/d/1VDVIVnrXFSLDDuMmS9gf5rfmOKNu4wxb/view?usp=sharing

# Reference
@ARTICLE{10713288,
  author={Liu, Jinyang and Li, Shutao and Tan, Lishan and Dian, Renwei},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Denoiser Learning for Infrared and Visible Image Fusion}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Image fusion;Generators;Semantics;Loss measurement;Feature extraction;Noise reduction;Generative adversarial networks;Training;Learning systems;Data mining;Deep learning;denoiser;infrared image (IR) and visible image (VI) fusion},
  doi={10.1109/TNNLS.2024.3454811}}

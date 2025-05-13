# AP-SAM
We propose a model that integrates a Convolutional Neural Network (CNN) with the Segment Anything Model (SAM) for automatic pore segmentation in rock SEM images. As shown in the figure below, this model is termed Auto-Pore SAM (**AP-SAM**).
AP-SAM is capable of automatically segmenting pores in scanning electron microscope images of rocks for automatic pore analysis. 
It prepares for 3D pore structure reconstruction, calculation of porosity and so on.
 ![AP-SAM](AP-SAM.svg)
AP-SAM architecture
# Training process
Training process on mudstone data,we train 200 epochs on AP-SAM.

Original Mudstone dataset is available on [Digital Rocks Portal](https://www.doi.org/10.17612/BVXS-BC79)

Train loss on train set:
![image](epoch200train_loss.svg)

IoU and Dice Coefficient on vaild set(also test set):
![image](epoch200val_iou_biou.svg)
# TODO 
- [ ] Usage
- [x] Ap-sam weight file
- [ ] Improve code readability

# Usage
## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

After above installation, please install `opencv-python` `pycocotools` `matplotlib` `pillow` `numpy`(you can use requirements.txt install)

**Example:**
In Linux,
Create and activate conda env:

`conda create -n APSAM python==3.10`
Activate it

`conda activate APSAM`
Install pytorch and torchvision (cuda version for GPU)

`pip3 install torch torchvision torchaudio`
Use requirements.txt install other :
`cd AP-SAM`
`pip install -r requirements.txt`






# Fine-tuned weight
We provide the AP-SAM weights file after fine-tuning on the mudstone dataset. The model inputs are at 880x1024 resolution.

Download the fine-tuned AP-SAM in ViT-B version here.[Google Drive](https://drive.google.com/file/d/1qAdj6o_2KQveai4_fK6LrInSiNkt7Ds8/view?usp=sharing)



# Thanks 
We modified [HQ-SAM](https://github.com/SysCV/sam-hq) to AP-SAM
```
@inproceedings{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={NeurIPS},
    year={2023}
}  
```
Segment anything model:[SAM](https://github.com/facebookresearch/segment-anything)
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

Mudstone dataset:[Original Dataset](https://www.doi.org/10.17612/BVXS-BC79)
```
Bihani, Abhishek, Daigle, Hugh, Prodanovic, Masa, Milliken, Kitty L., Landry, Christopher, E. Santos, Javier. "Mudrock images from Nankai Trough." Digital Rocks Portal,  Digital Rocks Portal, 22 Apr 2025, https://www.doi.org/10.17612/BVXS-BC79 Accessed 13 May 2025.
```

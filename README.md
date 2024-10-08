# MSLF-SSRNet: Multi-Subspace Light Field Spatial Super-Resolution Network

## Contributions:
* **We propose a novel extraction strategy with three specialized convolutional filters for the 2D subspaces of LFs,designed to fully capture LF information for spatial super-resolution tasks.**
* **We develop a multi-subspace self-attention feature fusion module (MSSA-FFM) to effectively merge diverse features from different subspaces.**
* **We introduce a combined loss function integrating edge-based and pixel-based losses to mitigate the overly smooth and blurry images reported in previous studies.**
<br><br>
## Codes and Models:
### Dependencies
* **Ubuntu 22.04.4**
* **Python 3.8.19**
* **Pyorch 2.4.0 + torchvision 0.19.0 + cuda 12.1**
* **Matlab**

### Datasets
**We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test. Please first download our dataset via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder **`./datasets/`**.**

### Prepare Training and Test Data
* To generate the training data, please first download the five datasets and run:
  ```matlab
  GenerateTrainingData.m
* To generate the test data, run:
  ```matlab
  GenerateTestData.m
### Train
* **Run train.py to perform network training. Example for training MSLF-SSRNet on 5x5 angular resolution for 4x/2xSR:**
  ```
   python train.py --model_name MSLF-SSRNet --angRes 5 --scale_factor 4 --batch_size 4
   python train.py --model_name MSLF-SSRNet --angRes 5 --scale_factor 2 --batch_size 8
  ```
* **Checkpoint will be saved to **`./log/`**.**  
### Test
* **Run test.py to perform network inference. Example for test LFT on 5x5 angular resolution for 4x/2xSR:**
  ```
  python test.py --model_name MSLF-SSRNet --angRes 5 --scale_factor 4 
  python test.py --model_name MSLF-SSRNet --angRes 5 --scale_factor 2 
  ```
* **The PSNR and SSIM values of each dataset will be saved to **`./Test result/`**.**
<br><br>
## Results:  


## Acknowledgement
Our work and implementations are inspired and based on the following projects: <br> 
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)<br> 
[LF-InterNet](https://github.com/YingqianWang/LF-InterNet)<br> 
We sincerely thank the authors for sharing their code and amazing research work!

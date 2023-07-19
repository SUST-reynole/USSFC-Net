# **USSFC-Net**

---

The implementation for

[Ultralightweight Spatial–Spectral Feature Cooperation Network for Change Detection in Remote Sensing Images](https://ieeexplore.ieee.org/document/10081023)

on *2023 IEEE Transactions on Geoscience and Remote Sensing*.

## **Network**

---

![img](https://img2023.cnblogs.com/blog/2735963/202304/2735963-20230412215057042-638909083.png)

## **Datasets**

---

[**LEVIR-CD**](http://chenhao.in/LEVIR/) is a large public CD dataset covering a variety of complex change features. It contains 637 pairs of remote sensing images of size 1024 × 1024 with 0.5-m resolution. To make full use of GPU memory and prevent overfitting, we crop the images into 13 072 patches of size 256 × 256. Finally, the dataset is divided into three parts: 10 000/1024/2048 for raining/validation/test, respectively. 

[**CDD**](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit) is a public CD dataset of seasonal changes in the same area obtained from Google Earth. It contains 11 pairs of multispectral images with resolutions ranging from 0.03 to 1 m. In all, 16 000 patches of size 256 × 256 are obtained from the original images by cropping and rotation operations. The final dataset is divided into three parts: 10 000/3000/3000 for training/validation/test, respectively. 

[**DSIFN-CD**](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) is a public CD dataset manually collected from Google Earth. It consists of six high-resolution images from different cities in China. The authors provide cropping the Xi’an image pair into 48 patches of size 512 × 512 for model testing. The other five city images were cropped into 3940 patches of the same size for training and validation. The final obtained dataset is divided into three parts: 3600/340/48 for training/validation/test, respectively.

## **Experiments**

![image-20230719114603971](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20230719114603971.png)

![image-20230719114633070](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20230719114633070.png)

## Citation

If you find this work useful for your own research, please consider citing our paper as follow:

```
@ARTICLE{10081023,
  author={Lei, Tao and Geng, Xinzhe and Ning, Hailong and Lv, Zhiyong and Gong, Maoguo and Jin, Yaochu and Nandi, Asoke K.},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Ultralightweight Spatial–Spectral Feature Cooperation Network for Change Detection in Remote Sensing Images}, 
  year={2023},
  volume={61},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2023.3261273}}

```


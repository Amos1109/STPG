# Exploiting Minority Pseudo-Labels for Semi-Supervised Fine-grained Road Scene Understanding

The official Pytorch implementation of "Exploiting Minority Pseudo-Labels for Semi-Supervised Fine-grained Road Scene Understanding" accepted by *IEEE Transactions on Intelligent Transportation Systems*.

## (0) Abstract

In fine-grained road scene understanding, semantic segmentation plays a crucial role in enabling vehicles to perceive and comprehend their surroundings. By assigning a specific class label to each pixel in an image, it allows for precise identification and localization of detailed road features, which is vital for high-quality scene understanding and downstream perception tasks. A key challenge in this domain lies in improving the recognition performance of minority classes while mitigating the dominance of majority classes, which is essential for achieving balanced and robust overall performance. 			However, traditional semi-supervised learning methods often train models overlooking the imbalance between classes. To address this issue, firstly, we propose a general training module that learns from all the pseudo-labels without a conventional filtering strategy. Secondly, we propose a professional training module to learn specifically from reliable minority-class pseudo-labels identified by a novel mismatch score metric. The two modules are crossly supervised by each other so that it reduces model coupling which is essential for semi-supervised learning. During contrastive learning, to avoid the dominance of the majority classes in the feature space, we propose a strategy to assign evenly distributed anchors for different classes in the feature space.  Experimental results on multiple public benchmarks show that our method surpasses traditional approaches in recognizing tail classes.


## (1) Setup

This code has been tested with Python 3.6, PyTorch 1.0.0 on Ubuntu 18.04.

* Setup the environment
  ```bash
  conda create -n STPG python=3.6
  source activate STPG
  conda install pytorch==1.0.0 torchvision==0.2.2
  ```

* Clone the repository

* Install the requirements
  ```bash
  pip install -r requirements.txt
  ```
  
* Download pertained models

  * Download the ResNet-50/ResNet-101 for training and move it to ./DATA/pytorch-weight/

      | Model                    | Baidu Cloud  |
      |--------------------------|--------------|
      | ResNet-50                | [Download](https://pan.baidu.com/s/1agsf6BSvmVVGTvk23JXxaw): skrv |
      | ResNet-101               | [Download](https://pan.baidu.com/s/1PLg22P_Nv9GwR-KEzGdvTA): 0g8u |

## (2) Cityscapes

* Data preparation
  
  Download the "city.zip" followed [CPS](https://pkueducn-my.sharepoint.com/personal/pkucxk_pku_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpkucxk%5Fpku%5Fedu%5Fcn%2FDocuments%2FDATA&ga=1), and move the upcompressed folder to ./DATA/city

* Modify the configuration in [config.py](./exp_city/config.py)
  * Setup the path to the STPG
    ```python
    C.volna = '/Path_to_STPG/'
    ```
    > The recommended nepochs corresponding to the labeled_ratio are listed as below
    > | Dataset    | 1/16 | 1/8  | 1/4  | 1/2  |
    > | ---------- | ---- | ---- | ---- | ---- |
    > | Cityscapes | 128  | 137  | 160  | 240  |
  
* Training
  ```bash
  cd exp_city
  python train_contra.py
  ```

* Evaluation
  ```bash
  cd exp_city
  python eval.py -e $model.pth -d $GPU-ID
  # add argument -s to save demo images
  python eval.py -e $model.pth -d $GPU-ID -s
  ```


## (3) PascalVOC

* Data preparation
  
  Download the "pascal_voc.zip" at [BaiduCloud](https://pan.baidu.com/s/1x86kqXAFU9q3-lYPFN_7dw): o9b3, and move the upcompressed folder to ./DATA/pascal_voc

* Modify the configuration in [config.py](./exp_voc/config.py)
  * Setup the path to the STPG in line 25
    ```python
    C.volna = '/Path_to_STPG/'
    ```
    > The recommended nepochs corresponding to the labeled_ratio are listed as below
    > | Dataset    | 1/16 | 1/8  | 1/4  | 1/2  |
    > | ---------- | ---- | ---- | ---- | ---- |
    > | PascalVOC  | 32   | 34   | 40   | 60   |
  
* Training
  ```bash
  cd exp_voc
  python train_contra.py
  ```

* Evaluation
  ```bash
  cd exp_voc
  python eval.py -e $model.pth -d $GPU-ID
  ```

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{hong2026exploiting,
  title={Exploiting Minority Pseudo-Labels for Semi-Supervised Fine-grained Road Scene Understanding},
  author={Hong, Yuting and Wu, Yongkang and Xiao, Hui and Hao, Huazheng and Qiu, Xiaojie and Yao, Baochen and Peng, Chengbin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026}
}
```

### Acknowledgment

Part of our code refers to the work [CPS](https://github.com/charlesCXK/TorchSemiSeg)



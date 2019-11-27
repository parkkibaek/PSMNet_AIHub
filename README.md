# PSMNet_AIHub

본 저장소는 NIA 인도 보행 공공 데이터의 검수용으로 "[Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)" 기반으로 구축되었습니다.<br/>
인도 보행 데이터는 공공 데이터 구축을 목적으로 하는 [AI Hub](http://www.aihub.or.kr/)에서 제공됩니다.<br/>
인도 보행 공공 데이터는 장애인 인도보행의 어려움과 이동권 문제 해결을 위하여 만들어졌습니다.



This repository contains the inspection of NIA Sidewalk dataset provided by [AI Hub](http://www.aihub.or.kr/).<br/>
Sidewalk dataset is a public dataset to solve that disabled person suffer from the difficulty of mobility in the sidewalk.<br/>
This repository contains the code (in PyTorch) for "[Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)" paper (CVPR 2018) by [Jia-Ren Chang](https://jiarenchang.github.io/) and [Yong-Sheng Chen](https://people.cs.nctu.edu.tw/~yschen/).




## Introduction

PSMNet, a pyramid stereo matching network, consists two main modules: spatial pyramid pooling and 3D CNN.

<img align="center" src="https://user-images.githubusercontent.com/11732099/43501836-1d32897c-958a-11e8-8083-ad41ec26be17.jpg">

## Installation

### Dependencies

- [Python 2.7](https://www.python.org/downloads/)
- [PyTorch (0.4.0+)](http://pytorch.org)
- torchvision 0.2.0 (higher version may cause issues)

For detailed installation, Please refer [INSTALL.md](./INSTALL.md).<br/>
자세한 설치과정은 [INSTALL_kor.md](./INSTALL_kor.md) 을 참고하세요.


### Prepare datasets
Dataset Download : [http://www.aihub.or.kr/content/611](http://www.aihub.or.kr/content/611)<br/>
[main.py](./main.py) reads images from following data structure. The folder name starting with ZED* and test* is training data and test data respectively.

```
${datapath}
├── 1
│   ├── ZED1_1
|   │   ├── ZED1_KSC_000000_right.png
|   │   ├── ZED1_KSC_000000_right.png
|   │   ├── ZED1_KSC_000000_right.png
|   │   ├── ...
│   ├── ZED2_1
|   │   ├── ZED2_KSC_000000_right.png
|   │   ├── ZED2_KSC_000000_right.png
|   │   ├── ZED2_KSC_000000_right.png
│   ├── ...
│   ├── test_ZED4_1
|   │   ├── ZED4_KSC_000000_right.png
|   │   ├── ZED4_KSC_000000_right.png
|   │   ├── ZED4_KSC_000000_right.png
|   │   ├── ...
├── ...
├── 7
```

### Execution

```
sh run.sh
```

In [run.sh](./run.sh), following command are executed. Set --datapath as same as ${datapath} above data structure.

```
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your data folder. for example ./NIA/) \
               --epochs 200 \
               --batchsz 12 \
               --nworker 20 \
               --savemodel (path for saving model)/
```


### Pretrained Model
- NOTE: The pretrained model were saved in .tar; however, you don't need to untar it. Use torch.load() to load it.
- 학습된 모델이 .tar로 저장되나, 압축을 풀지 말고 torch.load()를 사용하면 됩니다.

| KITTI 2015 |  Scene Flow | KITTI 2012| NIA Sidewalk |
|---|---|---|---|
|[Google Drive](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1p4eJ2xDzvQxaqB20A_MmSP9-KORBX1pZ/view)|[Google Drive](https://drive.google.com/open?id=1S8KUz2bCRhBbj_YRyzjB03Ag_S8WJA2I)|


## Results

### Hardware

- 8 NVIDIA Titan Xp GPUs
- Intel Xeon 4210 CPU @ 2.20GHz

### Software

- Ubuntu 16.04
- Python 2.7
- PyTorch 0.4.0
- CUDA 9.0


### Results on NIA Sidewalk dataset

| Network | Language | D1-all (All) | Download |
|--|--|--|--|
| PSMNet | Pytorch 0.4.0 | 4.779 % | [Model](https://drive.google.com/file/d/1ZWhZL_ZiwQub9ZtoNjNzLXjgLHwbF4Wm/view?usp=sharing) |


### Qualitative results
#### Left image
<img align="center" src="https://user-images.githubusercontent.com/36181879/69713212-7ca8aa00-1147-11ea-912f-48a8c000ad40.png">

#### Predicted disparity
<img align="center" src="https://user-images.githubusercontent.com/36181879/69713178-6e5a8e00-1147-11ea-9415-6a0cb5453dc7.png">


## Contacts
parkkibaek@kaist.ac.kr

## License
MIT License

### Citation
```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}
```
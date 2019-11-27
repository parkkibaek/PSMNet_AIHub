 ## Installation


### PSMNet_AIHub 설치하기

a. Conda 가상환경과 dependencies 설치

```shell
conda create -n psmnet_aihub python=2.7 -y
source activate psmnet_aihub
conda install pytorch=0.4.1 cuda90 torchvision  -c pytorch
conda install -c anaconda pillow
conda install ipython pip
pip install matplotlib, mmcv
```

b. PSMNet_AIHub을 다운로드 받고 실행하기

```shell
git clone https://github.com/parkkibaek/PSMNet_AIHub.git
cd PSMNet_AIHub
sh run.sh
```




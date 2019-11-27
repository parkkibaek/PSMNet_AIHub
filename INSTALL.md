 ## Installation


### Install PSMNet_AIHub

a. Install virtual conda environment and dependencies

```shell
conda create -n psmnet_aihub python=2.7 -y
source activate psmnet_aihub
conda install pytorch=0.4.1 cuda90 torchvision  -c pytorch
conda install -c anaconda pillow
conda install ipython pip
pip install matplotlib, mmcv
```

b. Download and execute PSMNet_AIHub

```shell
git clone https://github.com/parkkibaek/PSMNet_AIHub.git
cd PSMNet_AIHub
sh run.sh # execute PSMNet_AIHub
```




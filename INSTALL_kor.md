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

c. 파라미터 설정하기
[run.sh](./run.sh) 내부에서 파라미터를 설정할 수 있습니다.
```
python main.py --maxdisp 192 \ # 최대 disparity를 의미합니다.
               --model stackhourglass \ # 모델 유형을 정의합니다.
               --datapath  \ # 데이터 셋이 들어 있는 폴더를 지정합니다. (예를 들면 ./NIA/)
               --epochs 200 \ # epoch 정의
               --batchsz 12 \ # batch size 정의
               --nworker 20 \ # worker 개수 정의
               --savemodel  # 학습된 모델 저장할 폴더를 지정합니다. (예를 들면 ./trained/)
```


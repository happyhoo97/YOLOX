<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## 과제 개요
### 가. 과제 설계 배경 및 필요성

무인항공기(UAV: Unmanned Aerial Vehicle)은 실제 조종사가 직접 탑승하지 않고, 지상에서
무선으로 조종해 사전 프로그램된 경로에 따라 자동, 반자동으로 날아가는 비행체다. 활용 분야에 따라 다양한 장비(광학, 적외선, 레이다 센서 등)를 탑재하여 감시, 정찰, 정밀공격의 유도, 통신/정보중계 등의 임무를 수행하고 있다. 그러나 UAV 플랫폼에서 실행되는 객체 탐지(Object Detection)를 통한 실시간 장면 분석은
UAV 기체의 제한된 메모리와 컴퓨팅 성능으로 인해 어려움이 있다. 하드웨어와 같은 물리적
제약에 영향을 받지 않도록, 작고 효율적인 객체 탐지 모델이 필요하며, 이와 관련된 연구가
많이 진행되고 있다. 따라서 이번 과제를 통해, 최신 객체 탐지 모델 중, 경량형 모델을 UAV 플랫폼에 적합하도록 학습 및 개선하고자 한다. 최신 객체 탐지 모델 중, 경량형 모델인 YOLOX-Nano를 UAV
플랫폼 최적으로 학습한 후, 선행 연구의 논문에서 제안되었던 SlimYOLOv3 모델과 비교 분석을 진행할 예정이다. 더 나아가, Base Convolution을 사용하는 YOLOX 시리즈 중, 가장 작은 모델인 YOLOX-S를 UAV 플랫폼에 더욱 적합하도록, pruning과 같은 방법을 통해 개선하고자 한다. SlimYOLOv3, YOLOX-Nano, 직접 Pruning한 YOLOX-S 모델의 결과를 비교, 분석하고자 한다.


### 나. 과제 주요내용

본 과제에서는 최신 객체 탐지 모델 중, YOLOX-Nano와 YOLOX-S를 UAV 플랫폼 최적으로 학습한
후, 선행 연구의 논문에서 제안되었던 SlimYOLOv3 모델과 비교 분석을 진행할 예정이다. 더 나아가, YOLOX-S를 Network Slimming 방식의 channel pruning으로 구조적 개선을 하고자 한다. YOLO(You Only Look Once) 시리즈는 real-time application을 위해 속도와 정확성 간의 최적
trade-off를 추구하는 객체 탐지 딥러닝 모델이다. YOLO 시리즈는 YOLOv2, YOLOv3를 거쳐 YOLOv4,
YOLOv5까지 개선, 개발되었다. 선행 연구에서 제안된 SlimYOLOv3는 YOLOv3에 channel pruning을 적용하여, 객체 탐지 성능의 큰
하락 없이 경량화한 모델이다. channel pruning이란 channel의 중요도를 나타내는 scaling factor가 낮은 channel을 제거하는 방식이다. UAV 플랫폼에 적용하고자 하는 YOLOX 모델은 객체 탐지 모델로 유명한 YOLO 시리즈 중, 일부 요소가 적용된 모델이다. 최근 객체 탐지에서 관심이 높은 3가지 요소인, anchor-free detectors,
advanced label assignment strategies, end-to-end (NMS-free) detectors가 YOLOv5까지 적용되지 않았다. 이에 선행 연구의 논문(YOLOX: Exceeding YOLO Series in 2021)에서 3가지 요소를 YOLO 시리즈에 적용한 모델인 YOLOX를 제안하였고, 유의미한 개선 효과를 확인했다. 선행 연구에서 YOLOX 모델을 제안할 때, 다양한 크기의 backbone을 적용하여 여러 버전의 YOLOX를 제안하였는데, 이 중
Depthwise Separable Convolution을 사용한 경량형 모델은 YOLOX-Nano이고, Base Convolution을 사요한 경량형 모델은 YOLOX-S이다. YOLOX-Nano, YOLOX-S를 VisDrone-Det 데이터셋으로 학습하여 UAV가 탐지하고자 하는 객체들을
대상으로 특화하고자 한다. VisDrone-Det은 드론이 다양한 장소, 높이로부터 촬영한 7,019개의 정적인 
이미지로 구성되어 있다. 이미지는 10개의 class(pedestrian, person, car, van, bus, truck, motor,
bicycle, awning-tricycle, tricycle)에 대해 bounding box로 annotation 되어 있다. YOLOX-S의 경우, 모델의 불필요한 구조를 prune하여 모델을 경량화 측면에서 고도화하고자 한다. 최종적으로 학습이 완료된 YOLOX-S, YOLOX-Nano, SlimYOLOv3를 다양한 metric으로 비교 분석한다. 


### 다. 최종결과물의 목표 

선행 연구에서 제안한 YOLOX의 경량모델들인 YOLOX-S와 YOLOX-Nano를 UAV 플랫폼에 특화하여
리소스 대비 성능이 뛰어난 고효율의 경량형 모델로 발전시키고자 한다. 또한 SlimYOLOv3,
YOLOX-Nano, UAV 플랫폼 특화 개량된 YOLOX-S를 네 가지의 metric(mAP, FLOPS, FPS, Model volume)을 기준으로 정량적 비교 분석을 수행한다.

## 과제 수행방법

추진된 내용은 총 6가지로, 1)선행연구 탐구, 2)데이터셋 전처리, 3)딥러닝 학습환경 구축, 4)
VisDrone 데이터셋에 대한 object detection 평가 코드 구현, 5) YOLOX-Nano 모델 학습, 6) YOLOX-S 모델 학습 및 경량화이다.


### 1) 선행연구 탐구

UAV 플랫폼 특화 YOLO 모델을 개발한 선행 연구(SlimYOLOv3: Narrower, Faster and
Better for Real-Time UAV Applications)를 통해, 기존의 YOLOv3 모델을 경량화 하는 방식에
대해 알 수 있었다. 본 선행연구에서는 드론에서 촬영한 항공사진을 데이터셋으로 사용하며, YOLOv3 모델을 학습하되, channel pruning을 통해 경량화를 진행하였다. channel pruning에
대해서는 본 선행연구에서 참조한 또 다른 선행연구(Learning Efficient Convolutional
Networks through Network Slimming)에서 확인할 수 있었다. 선행연구 탐구를 통해, L1 Norm, L1 Loss, L1 Regulation, batch normalization 등에 대한 개념을 학습할 수 있었다. 이를 바탕으로 모델이 학습되는 전반적인 과정도 이해할 수 있었다. 모델에 input이 들어간 후, input 노드에서 가중치가 곱해진 후, batch normalization으로 정규화가 되는데, batch normalization의 수식에서 scaling factor인 감마가 역전파를 통해 계속
업데이트가 될 때, 업데이트를 함에 있어 L1 regularization을 통해 업데이트된다. L1
regularization는 가중치의 크기를 고려한 cost function이기에 가중치가 작아지며, 모델의 성능을 향상한다. 따라서 불필요한 가중치는 0에 가까워지게 작아지거나 0이 되어, 즉 prune되기에 sparse한 모델을 구축할 수 있다. 가중치가 곱해지고, batch normalization가 진행된 데이터는 activation funcion을 거쳐, hidden layor의 첫 번째 노드로 들어간다. 위의 내용을 반복하며 hidden layor의 2, 3, n번째까지 진행하고, output 노드까지 진행된다. 


### 2) 데이터셋 전처리

선행연구(SlimYOLOv3: Narrower, Faster and Better for Real-Time UAV Applications)에서 사용한 데이터셋, VisDrone을 YOLOX-Nano 모델에 사용할 수 있도록 전처리 작업을 수행했다. VisDrone 데이터셋은 드론이 다양한 장소, 높이로부터 촬영한 7,019개의 정적인 이미지로 구성되어 있다. 이미지는 10개의 class(pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, tricycle)에 대해 annotation 되어 있다. YOLOX-Nano 모델은 PASCAL VOC, COCO 포맷의 데이터셋을 지원한다. 이에 VisDrone 데이터셋을 COCO 포맷으로 변환하는 전처리 코드를 작성하여 포맷을 변환했다. 구체적인 내용으로는, annotation 정보가 txt 파일로 저장되어 있던 기존의 VisDrone 데이터셋을 COCO 포맷에 맞춰 annotation 정보를 json 파일로 저장한 후, jpg파일들과 json 파일들을 COCO 포맷에서 사용하는 폴더 구성에 맞춰 재배치하였다. 


### 3) 딥러닝 학습환경 구축

교내 GPU 서버(Seraph)에서 학습을 진행하였다.
수행해야 하는 작업들을 shell script로 작성하고 srun, sbatch와 같은 Slurm 명령어를 통해 코드를 동작했다.


### 4) VisDrone 데이터셋에 대한 object detection 평가 코드 구현

YOLOX를 개발한 선행연구에서 사용한 objection detection metric은 mAP로, general
purpose를 갖는 방법론 연구의 특성상 VisDrone 데이터셋에 대한 objection detection을 평가하기에 부적합하다고 판단한다. 이에 10개의 class에 대한 objection detection의 precision,
recall, F1-Score, mAP를 평가할 수 있는 코드를 구현했다. 또한 선행연구(SlimYOLOv3)의 결과와 본 과제에서의 결과물을 동등하게 비교하기 위해, mAP의 경우 mAP@0.5를 계산하도록
코드를 구현했다. 


### 5) YOLOX-Nano 모델 학습

최종적으로 교내 GPU 서버에 VisDrone 데이터셋과 YOLOX-Nano 모델 구축 코드를 업로드한 후, 선행연구에서 제시한 parameter로 설정한 상태로 학습을 진행했다. 구체적인 내용으로는, backbone으로 사용되는 CSPDarknet의 depth와 width를 각각 0.33, 0.25로 설정하였다.
Input의 경우 832 by 832 사이즈로 설정하였다. Data augmentation의 경우, mixup method
는 사용하지 않고, mosaic method의 경우 scaling range를 줄여서 사용하였다. 총 300epoch
의 학습을 진행했으며, 결과 분석은 가장 성능이 우수했던 epoch의 가중치를 사용했다. 


### 6) YOLOX-S 모델 학습 및 경량화

YOLOX-Nano와 동일한 데이터셋인 VisDrone 데이터셋으로 학습을 진행했다. backbone의 depth와 width는 각각 0.33, 0.50으로 설정했으며, input의 경우 832 by 832 사이즈로 설정하였다. Data augmentation의 경우, mixup method와 mosaic method를 사용했으며, Nano모델 보다는 augmentation을 적극적으로 사용했다. YOLOX-S는 경량화 즉, Network Slimming 방식의 Pruning을 진행했다. Pruning에 대한 threshold는 0.65로 설정했으며, sparsity training에서의 sparsity regularization term λ를 0.0001로 설정했다. 총 300epoch의 학습을 진행했으며, Pruning으로 모델을 경량화 한 후, Pruning된 모델에 대해 Fine-tuning을 진행하여 모델의 완성도를 높이고자 하였다. Fine-tuning 역시 동일한 parameter 조건으로 300 epoch을 진행했다.


## Quick Start

<details>
모든 학습은 GPU서버의 Slurm 환경에서 진행


<summary>YOLOX-Nano 모델 학습</summary>
```shell
sbatch train_script.sh
```

</details>



<details>
<summary>YOLOX-S 모델을 Network Slimming으로 Pruning 진행</summary>
Sparsity training → Pruning → Fine Tuning의 과정을 수행


```shell
sbatch prune_script.sh
```

</details>

<details>
<summary>학습한 모델을 평가</summary>


```shell
sbatch validation.sh
```

## 참고 자료

Zhang, Pengyi, Yunxin Zhong, and Xiaoqiong Li. "SlimYOLOv3: Narrower, faster and better for
real-time UAV applications." Proceedings of the IEEE/CVF International Conference on Computer
Vision Workshops. 2019.

Ge, Zheng, et al. "Yolox: Exceeding yolo series in 2021." arXiv preprint arXiv:2107.08430 (2021).

Liu, Zhuang, et al. "Learning efficient convolutional networks through network slimming."
Proceedings of the IEEE international conference on computer vision. 2017.

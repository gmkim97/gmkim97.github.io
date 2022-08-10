---
title: Setting for PyTorch Learning (feat. M1 Macbook Air)
author: GMKim
date: 2022-08-11 00:50:00 +0900
categories: [Machine Learning, PyTorch]
tags: [Machine Learning, Anaconda, PyTorch, M1, Macbook Air]
---

## Introduction
---
연구실에 Hard-working 용도의 Laptop이 있기에 어쩔 수 없이 기존에 들고 다니는 M1 Macbook Air를 가지고 PyTorch를 공부하게 되었습니다.   
(물론 M1의 성능이 어느 정도인지 궁금한 것도 한 몫 하였지만요.)

다행스럽게도, 최근에는 Apple Silicon이 첫 등장한지 어느 정도 시간이 지났기에 점점 이를 업데이트를 통해서 Native로 호환시켜주는 추세입니다.  
그래서 설정을 하는데 매우 쉬웠으며 그렇게 오랜 시간이 걸리지도 않았습니다. MacOS의 터미널에 친숙하다면 문제 없을 걸로 보입니다.  
거기에 최근 PyTorch의 업데이트로 M1 CPU뿐만 아니라 GPU또한 이용해 성능을 가속시킬 수 있음을 확인하였습니다.

## Environment
---

- M1 Macbook Air 8GB RAM, 256GB SSD
- MacOS Monterey

## Anaconda
---

![Anaconda](/assets/img/anaconda.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### Install

먼저 상황에 맞도록 Python 기반의 가상환경을 만들어주고 관리할 수 있는 Tool인 Anaconda를 설치해보도록 합시다.  
이전에는 M1에 Native하게 지원을 하는 Anaconda가 없어서 miniforge라는 툴을 사용해야 했으나, 이후 2022년 5월 자 업데이트를 통해 가능하게 되었습니다!  
따라서 구글 크롬을 설치하는 것 만큼이나 쉽죠.

1. 다음의 Link를 통해 Anaconda 홈페이지로 갑니다.
    - [Anaconda](https://www.anaconda.com/)
2. Get Additional Installers 밑에 사과 모양을 클릭
3. 중간에 MacOS에서 `64-Bit (M1) Graphical Installer` 을 선택합니다. 
    - `Anaconda3-2022.05-MacOSX-arm64.pkg` 파일이 다운로드될 것입니다.
4. 클릭하여 설치합시다.
5. Done

> 설치를 완료하고 나서 Launchpad를 봐도 Anaconda 아이콘을 찾을 수 없을 것입니다.  
이는 anaconda가 어플로써 설치되는 것이 아니라 터미널 상의 설정으로써 설치되기 때문입니다.  
보통 GUI상의 Anaconda라고 한다면 `Anaconda Navigator`일 것인데, **아직 M1 맥에서는 지원하지 않습니다.**
{: .prompt-info }


### Verification

![Anaconda Installation_1](/assets/img/anaconda_1.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
```console
$ vi ~/.zshrc
# Verification for conda initialization
``` 
![Anaconda Installation_2](/assets/img/anaconda_2.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }  


### Create Environment

PyTorch를 설치하기 이전, 따로 이를 위한 Work Environment를 만들어봅니다. 이 과정은 생략해도 무관합니다. 
```console
$ conda create -n {your_env_name} python==3.8 
# Create new environment based on python version 3.8
```
anaconda가 설치될 때의 python version은 3.9이나, 여기에서는 호환성을 위해 하나 낮추어서 3.8로 하겠습니다.
```console
$ conda env list 
# List all environments
```
```console
$ conda activate {your_env_name}
# Switch to {your_env_name}
```
![Anaconda Installation_3](/assets/img/anaconda_3.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

## PyTorch
---

![PyTorch](/assets/img/pytorch.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### Install
다음으로 PyTorch를 설치해보도록 합시다. 이 또한 M1 Native한 PyTorch의 등장으로 아주 수월해졌습니다. 몇 번의 클릭으로 끝입니다.

1. 다음의 Link를 통해 PyTorch 설치 페이지로 갑니다.
    - [PyTorch Get Started](https://pytorch.org/get-started/locally/)
2. 현재의 상황에 맞게 설정 탭을 클릭합니다.
    - 저희의 경우, `Stable (1.12.1)`, `Mac`, `Conda`, `Python`, `Default`를 선택해 줍니다.
    ![PyTorch_3](/assets/img/pytorch_3.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
    - PyTorch 1.12 버전부터 M1 GPU 가속을 가능케 합니다.
3. 밑에 나오는 링크를 복사합니다.
4. 앞서 띄운 {your_env_name}의 터미널 창에 붙여넣습니다.
    ```console
    $ conda install pytorch torchvision torchaudio -c pytorch
    ```
5. 엔터 그리고 설치되기를 기다립니다.
6. Done

### Verification
```console
$ python
>>> import torch
>>> torch.randint(high=10, size=(2,3))
# Check whether torch is correctly installed
```
![PyTorch_1](/assets/img/pytorch_1.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }  
```console
$ python
>>> import torch
>>> torch.backends.mps.is_available()
# Check whether GPU Acceleration at M1 is possible
```
![PyTorch_2](/assets/img/pytorch_2.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }   

이상입니다. 감사합니다.  
  
## Reference

- [Apple Silicon(M1) 에 Anaconda & PyTorch 설치하기 + 성능 측정](https://velog.io/@bolero2/install-anaconda-pytorch-on-m1)
- [Install Pytorch(GPU) on M1 ver.220624](https://velog.io/@heiswicked/%EB%8B%88%EB%93%A4%EC%9D%B4-mps%EB%A5%BC-%EC%95%84%EB%8A%90%EB%83%90-Install-PytorchGPU-on-M1-ver.220624)
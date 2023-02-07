---
title: Setting for PyTorch Learning (feat. M1 Macbook Air)
author: GMKim
date: 2022-08-11 00:50:00 +0900
categories: [Machine Learning, PyTorch]
tags: [Machine Learning, Anaconda, PyTorch, M1, Macbook Air]
---

## Introduction
---
Since I recently bought M1 Macbook Air, which becomes famous for its performance by Apple's ARM chipset, I try to study and use PyTorch on my laptop.  
Well, I also have computer with Linus OS, and it works really fine, but I'm just curious about its utility on Machine Learning with PyTorch.  
In this page, I just want to share the procedure how I set up my M1 Macbook Air to use PyTorch.  

Fortunately, It has been some time since Apple Silicon first appeared, and it becomes compatible with Apple M1 through various updates.  
So, it won't take much time to prepare for the environment, if you are familiar with MacOS Terminal.  
Plus, I checked that M1's GPU can be used to accelerate the performance by recent update of PyTorch.  

## Environment
---

- M1 Macbook Air 8GB RAM, 256GB SSD
- MacOS Monterey

## Anaconda
---

![Anaconda](/assets/img/anaconda.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### Install

First, I installed Anaconda which can create virtual environments based on Python and manage them for each proper occasions.  
Before the update, I need to utilize the tool called miniforge since there wasn't the Anaconda that supports M1 natively.  
However, after the update on 2022.05., I can install Anaconda working on M1 chip.  

1. Go to Anaconda website via following link.
    - [Anaconda](https://www.anaconda.com/){:target="_blank"}
2. Click the button with apple figure under "Get Additional Installers".
3. Select `64-Bit (M1) Graphical Installer` of MacOS.
    - `Anaconda3-2022.05-MacOSX-arm64.pkg` will be downloaded.
4. Click the pkg file, and follow the instructions.
5. Done

> After you finish the installation, you might not find any icon related to Anaconda on Launchpad.  
This is because the Anaconda is installed as the setting on terminal, not the application.  
Windows, and Linux have `Anaconda Navigator` which shows its operation on GUI, but **It is not supported on M1 Mac yet.**
{: .prompt-info }


### Verification

![Anaconda Installation_1](/assets/img/anaconda_1.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
```console
$ vi ~/.zshrc
# Verification for conda initialization
``` 
![Anaconda Installation_2](/assets/img/anaconda_2.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }  


### Create Environment

Before I install PyTorch, I made its own working environment solely for machine learning.  
You can just skip this part.  

```console
$ conda create -n {your_env_name} python==3.8 
# Create new environment based on python version 3.8
```
Installed Anaconda is based on python version 3.9, but I downgraded the version to 3.8 for its compatibility.  

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
Next, I installed PyTorch on my working environment. This procedure also becomes easy thanks to the appearance of PyTorch which supports M1 natively.  

1. Go to the PyTorch installation website via following link.
    - [PyTorch Get Started](https://pytorch.org/get-started/locally/){:target="_blank"}
2. Select proper tabs depending on your computer settings.
    - In my case, I chose `Stable (1.12.1)`, `Mac`, `Conda`, `Python`, `Default` tabs.
    ![PyTorch_3](/assets/img/pytorch_3.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
    - After PyTorch **version 1.12**, it enables M1 GPU acceleration.
3. Copy the command line that appears on the page.
4. Paste on the terminal of {your_env_name}.
    ```console
    $ conda install pytorch torchvision torchaudio -c pytorch
    ```
5. Press Enter and wait for the installation.
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

Thank you for reading!   
  
## Reference

- [Apple Silicon(M1) 에 Anaconda & PyTorch 설치하기 + 성능 측정](https://velog.io/@bolero2/install-anaconda-pytorch-on-m1)
- [Install Pytorch(GPU) on M1 ver.220624](https://velog.io/@heiswicked/%EB%8B%88%EB%93%A4%EC%9D%B4-mps%EB%A5%BC-%EC%95%84%EB%8A%90%EB%83%90-Install-PytorchGPU-on-M1-ver.220624)
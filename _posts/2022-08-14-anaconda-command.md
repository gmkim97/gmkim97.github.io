---
title: Command for Anaconda (on MacOS)
author: GMKim
date: 2022-08-14 22:09:00 +0900
categories: [Machine Learning, Anaconda]
tags: [Machine Learning, Anaconda, MacOS]
---

## Introduction
---

- 이 노트는 MacOS의 Terminal 상에서 Anaconda를 사용하는데 있어 자주 사용한 Command들을 모아놓은 것들입니다.
<br>  

![Anaconda](/assets/img/anaconda.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }



## Basics
---

### conda env list

- 나의 컴퓨터에 만들어져 있는 가상 환경들(environments)의 list를 보여줍니다.
- 현재 자신이 있는 환경은 별표(*)로 표시됩니다.
![conda_env_list](/assets/img/conda_env_list.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### conda activate {env_name}

- {env_name}의 가상 환경으로 이동합니다.

### conda deactivate

- 현재의 가상 환경에서 나와 default인 base로 돌아갑니다.


## Handling environments
---

### conda create -n {env_name} python={python_version}

- 새로운 가상 환경을 만듭니다. 이때 이름은 {env_name}으로 합니다.
- 특정 버전의 파이썬을 지정하여 설치할 수 있습니다. 여기에서 버전은 {python_version}을 따릅니다.
- `-n` 대신 `--name` 또한 사용 가능합니다.

### conda create --clone {cloned_env} -n {new_env}

- {cloned_env}의 가상환경을 복제하여 {new_env} 이름의 환경을 새로 만듭니다.
- `-n` 대신 `--name` 또한 사용 가능합니다.

### conda env remove -n {env_name}

- {env_name}이라는 이름의 가상 환경과 그 하위 package들 까지 모두 삭제합니다.
- `-n` 대신 `--name` 또한 사용 가능합니다.

## Handling packages
---

### conda list

- 현재의 가상 환경에 설치되어 있는 package들을 list 형태로 보여줍니다.
![conda_list](/assets/img/conda_list.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### conda install {pkg_name}

- **현재 활성화된 가상 환경**에 {pkg_name}라는 이름의 package를 설치합니다.

### conda install -n {env_name} {pkg_name}

- **{env_name}라는 다른 가상 환경**에서 {pkg_name}라는 이름의 package를 설치합니다.
- `-n` 대신 `--name` 또한 사용 가능합니다.

### conda uninstall {pkg_name}

- **현재 활성화된 가상 환경**에 {pkg_name}라는 이름의 package를 삭제합니다.
- `uninstall` 대신 `remove`로 사용 가능합니다.

### conda uninstall -n {env_name} {pkg_name}

- **{env_name}라는 다른 가상 환경**에서 {pkg_name}라는 이름의 package를 삭제합니다.
- `uninstall` 대신 `remove`로 사용 가능합니다.
- `-n` 대신 `--name` 또한 사용 가능합니다.

### conda update {pkg_name}

- 현재 가상 환경에 설치되어 있는 {pkg_name}라는 이름의 package를 업데이트합니다.

### {pkg_name}

- package를 실행합니다.
- 예를 들어, `python`이나 `jupyter-notebook` 등이 있습니다.


## Reference

- [Conda CheatSheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
- [What is the difference between conda uninstall and conda remove?](https://stackoverflow.com/questions/71306374/what-is-the-difference-between-conda-uninstall-and-conda-remove)
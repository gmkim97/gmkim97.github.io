---
title: Command for Anaconda (on MacOS)
author: GMKim
date: 2022-08-14 22:09:00 +0900
categories: [Machine Learning, Anaconda]
tags: [Machine Learning, Anaconda, MacOS]
---

## Introduction
---

- This page includes several Anaconda commands that are frequently used on MacOS terminal. 
<br>  

![Anaconda](/assets/img/anaconda.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }



## Basics
---

### conda env list

- It shows the list of virtual environments which are set on the computer.
- Currently located environment is marked as asterisk(*).
![conda_env_list](/assets/img/conda_env_list.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### conda activate {env_name}

- It activates / moves to the environment called {env_name}.

### conda deactivate

- It deactivates current environment and moves back to base(default).

## Handling environments
---

### conda create -n {env_name} python={python_version}

- It creates new virtual environment with your own name, {env_name}.
- You can install specific version of python. Here, the version follows your input command, {python_version}.
- You can also use `--name` instead of `-n`. Both are valid.

### conda create --clone {cloned_env} -n {new_env}

- It copies the property of given environment, {cloned_env}, and creates new one with name {new_env} depending on copied property.
- You can also use `--name` instead of `-n`. Both are valid.

### conda env remove -n {env_name}

- It removes the environment called {env_name} including its sub-packages.
- You can also use `--name` instead of `-n`. Both are valid.

## Handling packages
---

### conda list

- It shows the list of sub-packages which are installed under your current working environment.
![conda_list](/assets/img/conda_list.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

### conda install {pkg_name}

- It installs the package named {pkg_name} under **current activated environment**.

### conda install -n {env_name} {pkg_name}

- It installs the package named {pkg_name} under **designated environment called {env_name}**.
- You can also use `--name` instead of `-n`. Both are valid.

### conda uninstall {pkg_name}

- It uninstalls / removes the package named {pkg_name} under **current activated environment**.
- You can also use `remove` instead of `uninstall`.

### conda uninstall -n {env_name} {pkg_name}

- It uninstalls / removes the package named {pkg_name} under **designated environment called {env_name}**.
- You can also use `remove` instead of `uninstall`.
- You can also use `--name` instead of `-n`. Both are valid.

### conda update {pkg_name}

- It updates specific package called {pkg_name} which is previously installed on current environment.

### {pkg_name}

- It runs installed package.
- For instance, you can run `python`, or `jupyter-notebook`.

## Reference
---

- [Conda CheatSheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf){:target="_blank"}
- [What is the difference between conda uninstall and conda remove?](https://stackoverflow.com/questions/71306374/what-is-the-difference-between-conda-uninstall-and-conda-remove){:target="_blank"}
---
title: How to handle Git? (1) - Clone, Init, Push
author: GMKim
date: 2023-03-14 15:05:00 +0900
categories: [Git]
tags: [Git, Github]
---

## Introduction

Since it is difficult to handle a lot of codes from different project members, most developers or researchers use [Github](https://github.com/){:target="_blank"} as their remote storage (repository).  
Unlike other cloud services like Google Drive, Onedrive, and Dropbox, Github requires some basic knowledge about Git and its command.  
Thus, throughout several pages, I'd like to simply organize the concepts with command-lines about Git.  
<br>

---

## Structure

At Git's workflow structure, there are 4 stages : Working Directory, Staging Area, Local Repository, Remote Repository.  
From Working Directory to Local Repository, they have their own roles at **local hardware**, and Remote Repository refers to **Github**.  

1. Working Directory
    : Working Directory is similar to our working desk. We can freely modify, move, or delete some files in here. Before we add to Staging Area, Git doesn't track these revisions and recognizes them as untracked ones.

2. Staging Area
    : After you add the files from Working Directory, Your files will be on Staging Area. Before you commit the data to Local Repository, they are on stage tracking by Git.

3. Local Repository
    : Local Repository is the place where the Git truly memorize the history of modification about your files at local. Pushed files of Local Repository finally store in your remote repository.

4. Remote Repository
    : The main copy of your files on Github server. Anyone can pull or clone your files (except for private option) which are stored in this Remote Repository.

![Git_1](/assets/img/git_1.jpeg){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }_Git Workflow Diagram ([Image Source](https://dev.to/tauag/quick-start-guide-to-git-2of5){:target="_blank"})_
<br>

---

## Make your repository at Github

Before you save your files on Github, you have to make the repository for them.

![Git_2](/assets/img/git_2.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }_New Github repository (1)_  

![Git_3](/assets/img/git_3.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }_New Github repository (2)_  

![Git_4](/assets/img/git_4.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }_New Github repository (3)_  

The URL of HTTPS tab is your {remote repository URL}, which is used for several git commands.  
<br>

---

## Initialization - git clone, git init

There are two ways to start using Git.

- Git clone

```console
$ git clone {remote repository URL}
```

- Git init

```console
$ cd Desktop/Git_Test
# Move to the repository you want to work on.
```
```console
$ git init
# Designate the repository as Local Repository
```
<br>
When you activate to watch hidden file, you can find the folder called `.git`
<br>

![Git_5](/assets/img/git_5.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }_Hidden folder of .git_

```console
$ git remote add origin {remote repository URL}
```

Here, using `git remote add` command, your {remote repository URL} is designated to custom name, `origin`.  
Of course, you can change the name instead of "origin", freely.
<br>

---

## Upload your file - git push

```console
$ git status
# To check where your files are.
```

![Git_6](/assets/img/git_6.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }_Git status_

```console
$ git add .
```

Here, `.` refers to `every files`.  
Or, you can choose specific files to upload on your Github repository.  

```console
$ git commit -m "Your message"
```

```console
$ git push origin master
```
Now, the file on local will be pushed to `origin`, your custom name of {remote repository URL} on the branch called `master`.  

![Git_7](/assets/img/git_7.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }_Pushed file on Github repository_
<br>

---
## Reference

- [Quick Start Guide to Git](https://dev.to/tauag/quick-start-guide-to-git-2of5){:target="_blank"}
- [[Git] 구조와 사용법 소개 (KOR)](https://velog.io/@hahaha/Git-%EA%B5%AC%EC%A1%B0%EC%99%80-%EC%82%AC%EC%9A%A9%EB%B2%95-%EC%86%8C%EA%B0%9C){:target="_blank"}
- [[Github] Github에 업로드하는 기본적인 방법 (KOR)](https://victorydntmd.tistory.com/53){:target="_blank"}
---
title: Object Tracker
author: GMKim
date: 2022-01-09 00:00:00 +0900
categories: [Vision]
tags: [Vision, Object Detection, Depth Camera, ROS1, YOLOv3, darknet]
---

## Overview
---
- Object Tracker is a *personal* project I worked with [TaeHyeon Kim](https://github.com/QualiaT){:target="_blank"}.


## Goal
---
- To simultaneously detect various objects and track their location with different distances. 


## Description
---
- The source code is written in Python language and the package is built by CMake.

- This package enables to recognize objects, publish TF topic, and
display distances for each recognized object using Intel Realsense depth camera (D435i) under Linux Ubuntu OS and ROS1.

![obj_tracker_1](/assets/img/object_tracker_1.png){:width="80%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![obj_tracker_2](/assets/img/object_tracker_2.png){:width="80%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![obj_tracker_3](/assets/img/object_tracker_3.png){:width="80%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

## References
---
- [object_tracker](https://github.com/gmkim97/object_tracker){:target="_blank"} [Github Link]

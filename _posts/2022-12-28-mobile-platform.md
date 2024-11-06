---
title: Mobile Robot Platform with ToF Sensors
author: GMKim
date: 2022-12-28 00:00:00 +0900
categories: [Navigation]
tags: [Navigation, Mobile Robot, ToF, Lidar, ROS1, Autodesk Inventor]
---

## Overview
---
- This was a *collaborative* project I worked with Seung Jin Yang under the guidance of the [Robotics Innovatory lab](https://mecha.skku.ac.kr/roboticsinnovatory/index.do){:target="_blank"}.

- It was submitted to AI-ICT Creative Idea Contest hosted by Sungkyunkwan University. - *3rd prize*


## Goal
---
- To design a scalable mobile platform hardware that can support additional sensors, algorithms, and hardware for future use.

- Also, to enable basic navigation using a LiDAR sensor.


## Description
---
- Hardware
    - Designed with reference to various commercial mobile robots, it maintains a low height to allow for additional modules on top and adopts a rounded shape to account for environments with many people nearby.
    - Using aluminum profiles for the internal structure to enhance robot stability.
    - For cost efficiency, the robot’s exterior was designed in AutoDesk Inventor and then 3D printed for assembly.

- Navigation
    - Utilizes a YDLidar sensor to gather surrounding environmental data, enabling distance measurements essential for navigation tasks.
    - Operates in ROS1 Noetic environment.
    - Mapping: Hector mapping
    - Localization: AMCL
    - Global navigation: A* algorithm
    - Local navigation: Dynamic Window Approach (DWA)

![mobile_1](/assets/img/mobile_1.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![mobile_2](/assets/img/mobile_2.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![mobile_3](/assets/img/mobile_3.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![mobile_4](/assets/img/mobile_4.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }


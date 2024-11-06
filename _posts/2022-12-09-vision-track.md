---
title: Vision-based Track Following Vehicle
author: GMKim
date: 2022-12-09 00:00:00 +0900
categories: [Vision]
tags: [Vision, Navigation, Mobile platform, Jetson Nano, Object Detection, YOLOv3, ResNet]
---

## Overview
---
- This was a *collaborative* project I worked with Donghyuk Park, Yunseong Jeon, and Hyeonggeun Hong.

- The project was submitted to Autonomous Driving Capstone Design Contest hosted by Sungkyunkwan University. - *1st prize*

- This contest was linked with the Autonomous Driving Capstone Design (ICE3051-41) course at Sungkyunkwan University, supervised by Professors [Jonghoek Kim](https://home.sejong.ac.kr/~jonghoek/){:target="_blank"}, [Eunbyung Park](https://ice.skku.edu/eng_ice/intro/faculty_elec.do?mode=view&perId=LZStrMwewsgbA6g0gqgeQNQDsDCArCIBOALARQEkcAFAEQHcBeaoA%20&){:target="_blank"}, and [Il Yong Chun](https://ice.skku.edu/eng_ice/intro/faculty_elec.do?mode=view&perId=LZStrEoNQ4gjgDAxgbAMwPIGcCCAvATAdwBwBWALAKYl4DSAigLw1A%20&){:target="_blank"}.


## Goal
---
- To create a car-like model, combining a child’s electric car with a Jetson Nano board, that can drive along a track bordered by red traffic cones and stop upon recognizing a stop sign at the end of the road.


## Description
---
- Car-like model design
    - A Jetson Nano platform with monocular camera was mounted onto a commercially available child’s car to process data and provide control inputs to the car’s motor.

- Track Following
    - Red traffic cones line both sides of the road, defining its boundaries, and the goal is for the car to follow the track without colliding with or veering away from these cones.
    - Using deep learning, the steering angle was trained based on specific track conditions. Input data was a 224x224 image array from the camera, and the model used ResNet18.
    - To improve training efficiency, preprocessing was applied to the image array before feeding it into the model; the non-road upper portion of the image was cropped, and only the red color of the cones was tracked.

- Object Detection
    - At the end of the road, the stop sign is detected, triggering a signal to bring the car’s speed to zero.
    - Deep learning was used to train the model for stop sign detection. The input data was a 224x224 image array from the camera, and the model utilized Yolov5s.

![vision_1](/assets/img/vision_1.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![vision_2](/assets/img/vision_2.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![vision_3](/assets/img/vision_3.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![vision_4](/assets/img/vision_4.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }


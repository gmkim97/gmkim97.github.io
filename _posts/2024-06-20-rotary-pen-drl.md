---
title: Rotary Pendulum with PPO and Domain Randomization
author: GMKim
date: 2024-06-20 00:00:00 +0900
categories: [DRL]
tags: [DRL, Control, Mujoco, Gym, Pytorch, PPO, Domain Randomization]
---

## Overview
---
- Rotary Pendulum with PPO and Domain Randomization is a personal project.

## Goal
---
- To achieve robust control of a rotary (Furuta) pendulum using PPO and Domain Randomization (DR) in simulation, in preparation for real-world implementation where physical inaccuracies may occur.


## Description
---
- The source code is written in Python language and the modeling of rotary pendulum in XML format is from [macstepien](https://github.com/macstepien){:target="_blank"}'s [furuta_pendulum](https://github.com/macstepien/furuta_pendulum/blob/master/furuta_pendulum_rl/model/furuta_pendulum.xml){:target="_blank"} repository.

- The project utilizes PPO algorithm from the Stable-Baselines3 library as DRL framework.

- Simulations are conducted using Mujoco.

- Domain Randomization (DR) is a method that considers the real-world environment as one of many possible random variations, enabling the simulation to learn under diverse physical conditions.

- Three physical properties are randomized: the mass of the pendulum, the length of the pendulum, and the mass of the rod (arm1) connecting the pendulum to the central cylinder.

- The state space includes the angles and angular velocities of arm1 and the pendulum.

- The action space consists of actuator torques ranging from -0.4 to 0.4.

- The reward function is designed to encourage the pendulum to maintain the upright position.


![rotary_1](/assets/img/rotary_1.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![rotary_2](/assets/img/rotary_2.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }
![rotary_3](/assets/img/rotary_3.png){:width="100%" height="auto" :style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }


## References
---
- [rotary_pendulum_ppo](https://github.com/gmkim97/rotary_pendulum_ppo){:target="_blank"} [Github Link]


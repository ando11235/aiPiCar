# aiPiCar - DGMD Fall 2019

## Building a Self-Driving Raspberry Pi Car Capable of Lane Following, Object Recognitions and Obstacle Avoidance
### Project by: Alex Andony

![](Pictures/FinishedCar.jpg)

### Project Intorduction and Overview

Pick up any news source today, and you’ll likely see something about autopilot technology. Whether it’s lane assistance on production cars, self driving fully autonomous vehicles on the streets of LA, delivery drones autopiloting themselves to delivery drops, or worse, a recent accident caused by testing of these emerging technologies, we can be certain that the industry is rapidly advancing towards adoption of autopilot technology. 

Although major corporations like Tesla and Amazon are pouring billions of dollars into this technology, the practical applications can be learned and demonstrated on a much smaller scale. The goal of this project is to demonstrate these technological concepts using a small Robotic Car powered by a Raspberry Pi, at a price tag under $400. 

#### In order to demonstrate the above technology, we outlined the following goals to demonstrate this behavior:
* Assemble the PiCar Kit and Base Raspberry Pi Operating System 
* Demonstrate Autonomous “Lane Following” through OpenCV
* Demonstrate Advanced Autonomous “Lane Following” using Deep Learning and TensorFlow
* Demonstrate Advanced Camera recognition features, including adapting to street signs and basic obstacle avoidance

#### The build was inspired and heavily influenced by the following web pages:
* [Building a Raspberry Pi Car Robot with WiFi and Video](https://www.hanselman.com/blog/BuildingARaspberryPiCarRobotWithWiFiAndVideo.aspx)
* [DeepPiCar — Part 1: How to Build a Deep Learning, Self Driving Robotic Car on a Shoestring Budget](https://towardsdatascience.com/deeppicar-part-1-102e03c83f2c)
* [Road Lane Detection with Raspberry Pi](https://www.hackster.io/Abhinav_Abhi/road-lane-detection-with-raspberry-pi-a4711f)

### Hardware and Software Overview
#### Hardware Used for aiPiCar
* Car Kit Used: [SunFounder Smart Car Kit](https://www.amazon.com/gp/product/B06XWSVLL8/ref=ppx_yo_dt_b_asin_title_o00_s02?ie=UTF8&psc=1)
* CPU: [Raspberry Pi 3 B+ (B Plus)](https://www.amazon.com/gp/product/B07BC6WH7V/ref=ppx_yo_dt_b_asin_title_o00_s02?ie=UTF8&psc=1)
* MicroSD: [Kingston 64GB MicroSD Card](https://www.amazon.com/gp/product/B079GVC5B8/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1)
* Batteries: [Generic 18650 Batteries x4 + Charger](https://www.amazon.com/gp/product/B07T93HQYZ/ref=ppx_yo_dt_b_asin_title_o00_s01?ie=UTF8&psc=1)
* Upgrade 1 - Wide Angle Camera: [WLP Wide Angle USB Camera](https://www.amazon.com/gp/product/B01N07O9CQ/ref=ppx_yo_dt_b_asin_title_o00_s02?ie=UTF8&psc=1)
* Upgrade 2 - CPU Upgrade: [Coral EdgeTPU USB Accelerator](https://www.amazon.com/gp/product/B07S214S5Y/ref=ppx_yo_dt_b_asin_title_o00_s03?ie=UTF8&psc=1)


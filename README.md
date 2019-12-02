# aiPiCar - DGMD Fall 2019

## Building a Self-Driving Raspberry Pi Car Capable of Lane Following, Object Recognitions and Obstacle Avoidance
### Project by: Alex Andony

<img src="Pictures/FinishedCar.jpg" width="75%">

## Project Introduction and Overview

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

## Hardware and Software Overview
#### Hardware Used for aiPiCar:
* Car Kit Used: [SunFounder Smart Car Kit](https://www.amazon.com/gp/product/B06XWSVLL8/ref=ppx_yo_dt_b_asin_title_o00_s02?ie=UTF8&psc=1)
* CPU: [Raspberry Pi 3 B+ (B Plus)](https://www.amazon.com/gp/product/B07BC6WH7V/ref=ppx_yo_dt_b_asin_title_o00_s02?ie=UTF8&psc=1)
* MicroSD: [Kingston 64GB MicroSD Card](https://www.amazon.com/gp/product/B079GVC5B8/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1)
* Batteries: [Generic 18650 Batteries x4 + Charger](https://www.amazon.com/gp/product/B07T93HQYZ/ref=ppx_yo_dt_b_asin_title_o00_s01?ie=UTF8&psc=1)
* Upgrade 1 - Wide Angle Camera: [WLP Wide Angle USB Camera](https://www.amazon.com/gp/product/B01N07O9CQ/ref=ppx_yo_dt_b_asin_title_o00_s02?ie=UTF8&psc=1)
* Upgrade 2 - CPU Upgrade: [Coral EdgeTPU USB Accelerator](https://www.amazon.com/gp/product/B07S214S5Y/ref=ppx_yo_dt_b_asin_title_o00_s03?ie=UTF8&psc=1)

#### Software Used for aiPiCar:
* Raspberry Pi OS: [Rasbpian Buster v. Sept 2019](https://www.raspberrypi.org/downloads/raspbian/)
* Remote File Server: Samba File Server 
``` ~sudo apt-get install samba samba-common-bin -y ```
* Camera Driver Utlity: v4l2 Linux 
``` ~sudo apt-get install v4l-utils ```
* Remote Car Control: SunFounder PiCar ``` ~git clone https://github.com/dctian/SunFounder_PiCar.git ```
* Python v3.7.3
* Open CV v4.1.1
* Matplotlib v3.0.2
* tensorflow v1.14.0
* Keras v2.3.1

#### Hardware/ Software for Laptop Setup:
* Laptop: Apple Macbook Pro
* Software: macOS Catalina v10.15.1
* MicroSD card flasher for Pi: [BalenaEtcher](https://www.balena.io/etcher/)
* Remote Desktop Viewer for Pi: [RealVNC](https://www.realvnc.com/en/connect/download/viewer/)

#### Additional Resources Used:
* Lego Figurines 10 Pack (Amazon)
* Street Signs: [Attatoy Kids Playset Signs](https://www.amazon.com/gp/product/B01A8XTHHA/ref=ppx_yo_dt_b_asin_title_o00_s01?ie=UTF8&psc=1)
* Blue Painters Tape for Lanes (Hardware Store)

## Building the aiPiCar

Bulding the aiPiCar was relatively easy. For the most part, I was able to follow the instructions from the manufacturer, with a few key diviations outlines below.

[Manufacturer Build Instructions](https://www.sunfounder.com/learn/download/X1BWQ19SYXNwYmVycnlfUGlfU21hcnRfVmlkZW9fQ2FyX1YyLjAucGRm/dispi)

<img src="Pictures/components.jpg" width="50%">

###### PiCar Components Laid Out

### Installing Heatsinks

**(Before Page 16, PCB Assembly)** - Install heatsinks on Respberry Pi Board

Before Installing the HATS on the Pi board during install, I needed to install the heatsinks directly to the Raspberry Pi board. To do so, all that was required was to remove the backing strip on the heatsink to expose the adhesive paste, and to push the ehatsinks onto the CPU and network card as shown below:

<img src="Pictures/heatsinks.jpg" width="50%">

###### Heatsinks applied to Pi Board

### Installing Upgraded Camera

**(Before Page 33, Pan-and-Tilt)** - Install upgraded USB Camera on aiPiCar

As recommended on the build site I referenced above, the camera that ships with the PiCar kit is an extremely low quality narrow lens. I ordered a 170 degree wide angle USB camera as a replacement, which was relatively easy to install with slight modification to the stock mounting plate. The experiment below illustrates the difference between the stock and “upgraded” camera.

<img src="Pictures/camerademo.jpg" width="50%">

###### Camera Placed roughly 16" from lego figurines on table

<img src="Pictures/stockcamera.jpg" width="50%">

###### Stock Camera View at 16"

<img src="Pictures/wideanglecamera.jpg" width="50%">

###### Upgraded Camera at 16"

To install the camera, rather than installing the stock pan and tilt servos, I secured the USB Camera directly to the Servo Plate using two M3x8 Cross Screws and M3 Nuts. The camera provided a wide enough lens that the pan and tilt servos were unnecessary.

<img src="Pictures/cameramount.jpg" width="50%">

###### Upgraded Camera Mounted on Servo Plate

### Installing Edge TPU Processor

**(Before Page 33, Pan-and-Tilt)** - Install Google EdgeTPU Processor

In order to take some of the processing load off of the Pi for image processing and running Object Detection in TensorFlow, I installed Google’s Edge Tensor Processing Unit USB. At an extremely low cost, it was an easy upgrade to allow the car to process images much more rapidly while leaving the onboard Pi CPU open for performing navigation.

Physical installation was very easy. I simply threaded one M3x8 screw through the top corner of the edgeTPU and secured it with an M3 nut. 

<img src="Pictures/edgeTPU.jpg" width="50%">

###### Google edgeTPU mounted to Servo Plate behind camera

## Raspberry Pi Software Setup

### RaspianOS Install

I installed Raspbian OS directly from the Raspberry Pi site. I initially tried to use the NOOBS installer package, but ran into issues with a known software bug no allowing the NOOBS installer to display on an HDMI connected monitor. To solve this, I reformatted the microSD card and flashed the RaspianOS directly to the card using BalenaEtcher (linked above). After reconnecting the HDMI, keyboard and mouse, I was able to see the Pi desktop below:

<img src="Pictures/pidesktop.png" width="50%">

### Setup Remote Client to macOS: 

After the RaspianOS was setup on the Pi and I was able to see the desktop, the next goal was to setup a VNC Server on the Pi and VNC Viewer on macOS. I used the free software RealVNC to accomplish this.

Step One: Enable SSH and VNC through the Pi’s Interface Settings

<img src="Pictures/piconfig.png" width="50%">

Step Two: Download RealVNC on Mac and enter the IP address of the Pi

<img src="Pictures/remotedesktop.png" width="50%">

### Setup Remote File Server

In order to be able to easily read/ write files from the Pi without needed to constantly disconnect the SD card, I wanted to setup remote file access to the file system. To do this, I installed SAMBA file server on the Pi using the directions found [here](https://pimylifeup.com/raspberry-pi-samba/). Once the server was setup on the Pi, I was able to connect using the Mac instructions found [here](http://osxdaily.com/2010/09/20/map-a-network-drive-on-a-mac/). 


<img src="Pictures/remotefile.png" width="50%">

###### Screenshot of the file server accessible through the macOS

### SunFounder PiCar-V Software Setup

Unfortunately, the Manufacturer's server code runs on Python 2, while we are running the project on Python 3. Thankfully, the author of one of the tutorials I am following (David Tian, linked above) has updated the repo to be Python 3 compatible. Commands to install his version of the Servewr API are found below:

```
alias python=python3
alias pip=pip3
alias sudo='sudo '
pi@raspberrypi:~ $ git clone https://github.com/dctian/SunFounder_PiCar.git
pi@raspberrypi:~ $ cd ~/SunFounder_PiCar/picar/
pi@raspberrypi:~/SunFounder_PiCar/picar $ git clone https://github.com/dctian/SunFounder_PCA9685.git
pi@raspberrypi:~/SunFounder_PiCar/picar $ cd ~/SunFounder_PiCar/
pi@raspberrypi:~/SunFounder_PiCar $ sudo python setup.py install
pi@raspberrypi:~/SunFounder_PiCar/picar $ cd
pi@raspberrypi:~ $ git clone https://github.com/dctian/SunFounder_PiCar-V.git
pi@raspberrypi:~ $ cd SunFounder_PiCar-V
pi@raspberrypi:~/SunFounder_PiCar-V $ sudo ./install_dependencies
```

### Installing Python and Dependant Packages

Once the OS was up and running, we need to install the other packages we will be utilizing:
* Install Open CV: 

``` pi@raspberrypi:~ $  sudo apt-get install libhdf5-dev -y && sudo apt-get install libhdf5-serial-dev -y && sudo apt-get install libatlas-base-dev -y && sudo apt-get install libjasper-dev -y && sudo apt-get install libqtgui4 -y && sudo apt-get install libqt4-test -y ```

###### NOTE python3-open cv, NOT opencv-python. This caused me lots of headaches and research time to solve initially.

``` pi@raspberrypi:~ $ pip3 install python3-opencv ```

* Install Matplotlib:

``` pi@raspberrypi:~ $ pip3 install matplotlib ```

* Install tensorflow:

``` pi@raspberrypi:~ $ pip3 install tensorflow ```

* Install keras:

``` pi@raspberrypi:~ $ pip3 install keras ```

### Installing Google edgeTPU

Similar to the above mentioned problem with python3 and OpenCV compatibility, there were some issues intially with compatibility between Google's edgeTPU and Rasbpian Buster. The original install instructions for the edgeTPU can be found below.

[Google EdgeTPU Install Instructions](https://coral.withgoogle.com/docs/accelerator/get-started/)

Again, fortunately, I was able to find a github repo with workaround instructions for Raspian and EdgeTPU compatibility. Instructructions followed below, under "Getting the Coral to work with the Pi 4":

[leswright1977/RPi4-Google-Coral](https://github.com/leswright1977/RPi4-Google-Coral)

### Testing Installed Packages:

Once all the packages have installed and are running without errors, I used David Tian's repo and "coco_object_detection" to make sure was running smoothly. In David's words:

> You should see a live video screen coming up, and it will try to identify objects in the screen at around 7–8 Frames/sec. Note that the COCO (Common Object in COntext) object detection model can detect about 100 common objects, like a person, chair, TV, couch, book, laptop, cell phone, etc. Don’t let this simple program fool you, this is Deep Learning at work. The object detection model used in this program is called ssd_mobilenet_coco_v2, and it contains more than 200 layers!

To run the python program:

```
pi@raspberrypi:~ $ git clone https://github.com/dctian/DeepPiCar.git
pi@raspberrypi:~ $ cd ~/DeepPiCar/models/object_detection/
pi@raspberrypi:~/DeepPiCar/models/object_detection $ python3 code/coco_object_detection.py 
```
As you can see, our aiPiCar was able to perform some object recognition:

<img src="Pictures/cocodetection.png" width="50%">

###### In the words of Neo - "I know Kung Fu!"

## Testing the aiPiCar

Now that the hardware and software have been setup for the aiPiCar, I was able to use the SunFounder's built in controller to demo the motor functions and camera features. To do so, we need to first run the PiCar server on the Pi, and then connect locally to the Server in order to operate the car remotely:

Step One: Start aiPiCar Server

```
cd ~/SunFounder_PiCar-V/remote_control 
sudo ./start 
```

Step Two: Connect to localhost from web browser

**http://{pi's IP address}:8000/**

Video of the aiPiCar operating via remote control:

[![Video of the aiPiCar operating via remote control](https://assets.change.org/photos/7/qo/yu/WPqOyUupuKfUyEM-800x450-noPad.jpg?1523045557)](https://www.youtube.com/watch?v=Bf4ltXj-Qyw)



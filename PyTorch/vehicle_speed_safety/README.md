# Vehicle Speed Detection with Safety Check for the passenger

Vehicle Detection right now is a very important application in computer vision and much work is being done on that side. 
The approaches vary and right now detecting speed of vehicles from the camera stream alone is still being worked upon.
There are different approaches that vary from using optical flow techniques to using depth perception in 3-D bounding boxes and more.
An approach without any reference of the road length and other assumtpions is a bit of a challenge compared to having some reference. 

The approaches here would be using Vehicle Detection then tracking the vehicle, since we will be trying to find the speed of the individual vehicles and not random detected ones. 
For this Vehicle tracking is needed. 

The tracking part is being implemented by using the **SORT** Tracker from this repo https://github.com/abewley/sort.
Vehicle detection and other object detections tasks to be taken care of by **YOLOV5** https://github.com/ultralytics/yolov5.


## Vehicle Tracking

1. Track Vehicles using SORT + YOLOv5

```Python
python track1.py --input cars.mp4 --output F
```
![[VIDEO](https://imgur.com/zw0VZkl)](https://i.imgur.com/zw0VZkl.gif)

2. Addition of script to save results to csv

```Python
python track2.py --input cars.mp4 --output results.csv
```
3. Vehicle Speed Detection with Safety Detection to csv

```Python
python speed_det.py --input test2.mp4 --output speed.csv --output2 safe.csv
```
![[VIDEO](https://imgur.com/EFhfaws)](https://i.imgur.com/EFhfaws.gif)

<a href="https://drive.google.com/uc?export=view&id=1p1OwtVQBapetK4RCPF7lI2m2yuioRKSE"><img src="https://drive.google.com/uc?export=view&id=1p1OwtVQBapetK4RCPF7lI2m2yuioRKSE" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>

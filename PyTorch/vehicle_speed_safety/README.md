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

```Python
python track1.py --input cars.mp4 --output F
```
![Video](https://drive.google.com/file/d/1fkt0M97OjrQAcqA8AK0dyT44vwfVuzea/view?usp=drive_link)

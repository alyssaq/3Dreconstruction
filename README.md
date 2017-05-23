# 3d reconstruction

WIP: 3d reconstruction from 2d images pipeline

Steps:
* Projection
* Detection
* Triangulation
* Bundle adjustment

## Prerequisites
* Python 3.3+
* Install [opencv](http://opencv.org/)
* pip install -r requirements.txt

## 3D to 2D Projection
```sh
$ python3 camera.py
```

3D points of model house from Oxford University VGG datasets.
![](testsets/house_3d.png?raw=true)

Projected points

![](testsets/3d_to_2d_projection.png?raw=true)
## Datasets
* Oxford University, Visual Geometry Group: http://www.robots.ox.ac.uk/~vgg/data/data-mview.html
* EPFL computer vision lab: http://cvlabwww.epfl.ch/data/multiview/knownInternalsMVS.html

## References
* [Eight point algorithm](http://ece631web.groups.et.byu.net/Lectures/ECEn631%2013%20-%208%20Point%20Algorithm.pdf)
* [Multiple View Geometry in Computer Vision (Hartley & Zisserman)](http://www.robots.ox.ac.uk/~vgg/hzbook/)
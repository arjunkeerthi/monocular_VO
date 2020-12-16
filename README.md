# monocular_VO
This repo provides a Python implementation of a basic visual odometry algorithm. This program was part of a final project for one of my classes (see report [here](EECE6356-SemesterReport-ArjunKeerthi.pdf)).

Credit to Avi Singh, whose [C++ code](https://github.com/avisingh599/mono-vo.git) was referred to extensively for this program.



Before running, be sure to visit the [Kitti Odometry Benchmark data set](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and download the color odometry data set (I used color, but the grayscale set should work with minor changes to the code) and odometry ground truth poses. Then, in `visualodom.py`, near the beginning, set the value of `path` to the path to the folder containing the two downloaded datasets (I assumed both are in the same folder).

![](sequence9gfft.gif)
Demo running program on Sequence 09 in Kitti data set at 11x speed.

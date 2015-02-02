# BING-Objectness
Python implementation of BING Objectness method from "BING: Binarized Normed Gradients for Objectness Estimation at 300fps" ( http://mmcheng.net/bing/ , https://github.com/bittnt/Objectness ). 

I have found original C++ code brilliant but hard for prototyping, thus I have decided to write my easy to modify/plug python version.
Python, numpy, matplotlib, boost.python (>= 1.55 for using with Caffe), OpenCV with python wrapper, Scikit-learn, xmltodict are required.

For compiling the C++ code wrapped in python, type 'make'.

Add the build folder to the PYTHONPATH for allowing python interpreter to load c++-wrapped libraries.

Training set should be in the Pascal VOC 2007 form: http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/ .

For adapting input, take as model doc/bing_params.json with your dataset.

Edit the file "/path/to/repo/doc/bing_params.json" according to your system settings, if you want to use the pretrained coefficients obtained on Pascal VOC 2012 just modify "1st_stage_weights_fn", "2nd_stage_weights_fn" and "sizes_indeces_fn" parameters to the correct paths.

For performing training:
python source/train_bing.py /path/to/repo/doc/bing_params.json

For testing bing on a single image:
python source/bing.py /path/to/repo/doc/bing_params.json /path/to/repo/doc/fish-bike.jpg

I have tested the code on Linux Ubuntu 14.04.

The training is pretty memory demanding.

I have still to improve some optimization at the code to speed it up.

Authors:
Alessandro Ferrari - alessandroferrari87@gmail.com

Licensing:
gpl 3.0

Enjoy.
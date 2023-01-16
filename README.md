## Reproduction of the experiments of the incremental luxes vs detected ArUcos for the cameras

We introduce the necessary steps to reproduce the results provided in the paper:

1. Download and unzip the bags, model and used dataset for digit model:

```
$ mkdir data && cd data
```

```
$ curl -o luxesArucos.zip -L 'https://drive.google.com/uc?export=download&confirm=yes&id=1qeEf0w5B2XcS_o3YE7ed5Sw-EZSmaLV-'
```

```
$ unzip luxesArucos.zip && cd material_luxes_vs_arucos
```

2. Add the downloaded bags to the bag folder of *aruco_markers* package.

```
$ cp -r bags/ /path_to_parent_folder_of_the_repo/event-vs-frame/ros/src/aruco_markers
```

3. Add the downloaded model to the model folder of *text_from_images* package.

```
$ cp -r digit_classification.h5 /parent_folder_of_the_repo/event-vs-frame/ros/src/text_from_images/model
```

4. Add the packages *aruco_markers* and *text_from_images* to your catkin workspace and compile it.

```
$ cd /path_to_parent_folder_of_the_repo/event-vs-frame/ros/src/
```
```
$ cp -r aruco_markers /path_to_your_catkin_ws/src
```

```
$ cp -r text_from_images /path_to_your_catkin_ws/src
```

```
$ cd /path_to_your_catkin_ws && catkin_make
```

5. Change the *catkin_path* parameter in both packages *parameters.yaml* file, inside *config* folder indicating yout catkin workspace path.

6. Type the following command in terminal:

```
$ roslaunch aruco_markers dr_experiments.launch camera:=the_desired_camera show:=true
```

Where *the_desired_camera* option in *camera* parameter can be *blue*, *elp*, *zed*, *rs*, or *ecap*, and *show* is a boolean parameter indicating wheter or not to visualize the results online. 

7. The results will be output to the *output* folder inside the *text_from_images* package.

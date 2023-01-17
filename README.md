## 2nd practice ROS

We introduce the necessary steps to install and configure the repository:

1. Clone the repository:

```
git clone https://github.com/javierganan99/digit_pkgs.git
```

2. Add the packages *aruco_markers* and *text_from_images* to your catkin workspace and compile it.

```
cd digit_pkgs
```
```
cp -r aruco_markers/ /path_to_your_catkin_ws/src
```

```
cp -r text_from_images/ /path_to_your_catkin_ws/src
```

```
cd /path_to_your_catkin_ws && catkin_make
```

3. Download and unzip the bags, model and used dataset for digit model:

```
cd && mkdir data && cd data
```

```
curl -o luxesArucos.zip -L 'https://onedrive.live.com/download?cid=DBE28B7F30469A49&resid=DBE28B7F30469A49%21252228&authkey=AC9i8ziJHnd95Vo'
```

```
unzip luxesArucos.zip && cd material
```

4. Add the downloaded bags to the bag folder of *aruco_markers* package.

```
cp -r bags/ /path_to_your_catkin_ws/src/aruco_markers
```

5. Add the dataset to the dataset folder of *text_from_images* package.

```
cp -r dataset/ /path_to_your_catkin_ws/src/text_from_images
```

6. Change the *catkin_path* parameter in both packages *parameters.yaml* file, inside *config* folder indicating your catkin workspace path.

7. Install the following with pip (after installing pip if don´t installed yet):
  - To install pip:
  ```
  sudo apt update
  ```
  ```
  sudo apt install python3-pip
  ```
  - To install tensorflow:
  ```
  pip3 install tensorflow
  ```
  - To install scipy:
  ```
  pip3 install scipy
  ```


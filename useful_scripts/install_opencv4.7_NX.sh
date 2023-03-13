# part of the script to be modified
# 1. opencv version to install. The current version is 4.7.0
# 2. python version of anaconda virtual environment. The current version is 3.8
# 3. CUDA_ARCH_BIN. The Current value is "7.2" for Xavier NX


echo "OpenCV 4.7 Installation script for Jetson Xavier NX (Jetpack 5.1)"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <Install Folder>"
    exit
fi
folder="$1"

#You don't have to remove pre-installed OpenCV 4.5
#echo "** Remove Old OpenCV first"
#sudo apt-get purge *libopencv*

echo "** Install requirement"
sudo apt-get update
sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev 
sudo apt-get install -y libeigen3-dev libhdf5-dev libgflags-dev
sudo apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp
sudo apt-get update
conda install -y numpy

echo "** Download opencv-4.7.0"
cd $folder
wget https://github.com/opencv/opencv/archive/4.7.0.zip 
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.7.0.zip 

unzip 4.7.0.zip 
unzip opencv_contrib.zip 
cd opencv-4.7.0/

echo "** Building..."
mkdir release
cd release
cmake \
-D ENABLE_PRECOMPILED_HEADERS=0  \
-D CUDA_ARCH_BIN="7.2" \
-D CUDA_ARCH_PTX="" \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.7.0/modules \
-D WITH_OPENCL=0 \
-D WITH_CUDA=1 \
-D OPENCV_DNN_CUDA=1 \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_EIGEN=1 \
-D WITH_CUBLAS=1 \
-D WITH_FFMPEG=1 \
-D WITH_GSTREAMER=1 \
-D WITH_V4L=1 \
-D WITH_LIBV4L=1 \
-D BUILD_opencv_python2=0 \
-D BUILD_opencv_python3=1 \
-D BUILD_TESTS=0 \
-D BUILD_PERF_TESTS=0 \
-D BUILD_EXAMPLES=0 \
-D OPENCV_GENERATE_PKGCONFIG=1 \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
-D PYTHON3_LIBRARY=$CONDA_PREFIX/lib/python3.8 \
-D PYTHON3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.8 \
-D PYTHON3_EXECUTABLE=$CONDA_PREFIX/bin/python \
-D PYTHON3_PACKAGES_PATH=$CONDA_PREFIX/lib/python3.8/site-packages \
..

make -j6
sudo make install

echo "** Install opencv-4.7.0 successfully"
echo "** Bye :)"
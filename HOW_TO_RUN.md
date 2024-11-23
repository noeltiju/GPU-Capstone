## How to Run the Code 

* Compile the code using the following command:
* * nvcc main.cpp -o classify_tiff -ltiff -lcudnn -lcuda

* Run the compiled program, providing the path to the folder containing .tiff images and the output text file as arguments:
* * ./classify_tiff input_files output_file.txt

## Instaling the required Libraries

1. Update Your System

* First, update your system to ensure all packages are up to date:
* * sudo apt update
* * sudo apt upgrade

2. Install NVIDIA Driver

* You need to install the NVIDIA driver for your GPU. You can use the following commands:
* * sudo apt install nvidia-driver-525

* Replace 525 with the appropriate driver version for your GPU. You can check the available versions using:
* * ubuntu-drivers devices

* After installing the driver, reboot your system:
* * sudo reboot

3. Install CUDA Toolkit

* Go to the CUDA Toolkit download page and select your OS, architecture, distribution, and version to get the appropriate download link.

* Follow the instructions provided on the website. Here is an example for Ubuntu 20.04:

* Download the CUDA Toolkit .deb package
* * wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-1_amd64.deb

* Install the package
* * sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-1_amd64.deb

*  Add the GPG key
* * sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*.keyring /usr/share/keyrings/

* Update the package lists
* * sudo apt-get update

*  Install CUDA
* * sudo apt-get -y install cuda

* After installing CUDA, add the CUDA binaries to your PATH. Add these lines to your ~/.bashrc or ~/.zshrc file:
* * export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
* * export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

* Reload the shell configuration:
* * source ~/.bashrc

4. Install cuDNN
* Go to the cuDNN download page and download the cuDNN tar file for your CUDA version. You will need to log in or create an NVIDIA account.

* Extract the downloaded tar file and copy the files to the CUDA directory:


*  Example for cuDNN 8.7.0
* * tar -xzvf cudnn-12.2-linux-x64-v8.7.0.84.tgz

*  Copy the files to the CUDA directories
* * sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
* * sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
* * sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

5. Verify the Installation
* To verify that CUDA and cuDNN are installed correctly, you can check their versions:

* Check CUDA version
* * nvcc --version

* Check cuDNN version
* * cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

6. Install libtiff

* If you haven't already installed libtiff, you can do so with the following command:
* * sudo apt install libtiff-dev

* By following these steps, you should have CUDA and cuDNN installed and ready to use on your Ubuntu system. 

## Proof of execution

* Output File after running the code which provides the predicted class and the class probailities

![image.png in Proof folder](image.png)

* Terminal after running the code successfully

![image2.png in Proof folder](image2.png)
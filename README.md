# 🚀 Setting Up CUDA, cuDNN, and TensorRT on Linux

Welcome to the ultimate Markdown guide for installing CUDA, cuDNN, and TensorRT on Linux! 🚀 This tutorial is designed to get your machine ready for GPU-accelerated TensorFlow and other deep learning frameworks. I've included all the commands, paths, and explanations you need to get up and running like a pro.

This guide is heavily inspired by [this fantastic YouTube video](https://www.youtube.com/watch?v=1Tr1ifuSh6o&t=858s). Seriously, go check it out — it’s packed with insights and even has a GitHub repo for extra support. But if you’re more of a Markdown fan and want a no-fuss, easy-to-follow walkthrough, you’re in the right place. Let’s dive in! 💻🔥

--- 



## Phase 1: System Preparation

**Explanation**: This step updates the package manager and installs essential development tools.

**Commands**:
```bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential
```
**Note**: These commands keep your system up-to-date and ensure you have the necessary tools for compiling code.

---

## Phase 2: Installing Miniconda

**Explanation**: Miniconda is a lightweight Python distribution that lets you create isolated environments. This will help manage dependencies cleanly.

**Commands**:

1. Download Miniconda: 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
```
2. Run the installer: 
```bash
bash ./Miniconda3-latest-Linux-x86_64.sh
```
**Note**: Follow the prompts to complete the installation.

---

## Phase 3: Installing CUDA

**Explanation**: CUDA is the framework that allows TensorFlow to use your GPU for computations. It is very important to know your GPUs model so you can get the proper CUDA version, for example my GPU is a GeForce 960 Strix so the best CUDA version is 11.8, for more information check this site: https://developer.nvidia.com/cuda-gpus

**Commands**:

1. Download CUDA installer: 
https://developer.nvidia.com/cuda-toolkit-archive

2. Run the installer:

### (Important: If you have a Nvidia drivers already installed in your machine please uncheck the driver box) 
### Set Up Environment Variables for CUDA

1. Open `.bashrc`
```bash
nano ~/.bashrc
```
2. Add the following lines:

```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

3. Save and close, then update the environment with
```bash
 source ~/.bashrc
```


**Explanation**: These paths tell the system where to find CUDA binaries and libraries.

### Verify CUDA Installation

1. Check if CUDA is correctly installed:
```bash
echo $PATH
echo $LD_LIBRARY_PATH
```
2. Run 
```bash
nvcc --version
``` 
to confirm CUDA is accessible.

---

## Phase 4: Installing cuDNN

**Explanation**: cuDNN is a GPU-accelerated library for deep neural networks. It’s required for TensorFlow to make efficient use of CUDA.

**Commands**:

1. Download cuDNN: [Visit the cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

2. Extract the archive:
```bash
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```
3. Navigate to the extracted directory: 
```bash
cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive
```

### Move cuDNN Files

1. Copy the header and library files:
```bash
sudo cp include/cudnn*.h /usr/local/cuda-12.1/include
```
```bash
sudo cp lib/libcudnn* /usr/local/cuda-12.1/lib64
```
```bash
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*
```
2. Verify installation: 
```bash
ls -l /usr/local/cuda-12.1/lib64/libcudnn*
```
---

## Phase 5: Installing TensorRT

**Explanation**: TensorRT is an SDK for optimizing and running deep learning inference. TensorFlow uses it to improve performance on NVIDIA hardware.

**Commands**:

1. Download TensorRT: [Visit the TensorRT Download Page](https://developer.nvidia.com/tensorrt/download)
2. Extract TensorRT: 
```bash
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
```
3. Move TensorRT to `/usr/local`: 
```bash
sudo mv TensorRT-8.6.1 /usr/local/TensorRT-8.6.1
```
### Update Environment Variables for TensorRT

1. Open `.bashrc` again with nano 
```bash
~/.bashrc
```
2. Add these paths:
```bash
export PATH=/usr/local/cuda-11.8/bin:/usr/local/TensorRT-8.6.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/TensorRT-8.6.1/lib:$LD_LIBRARY_PATH
```
3. Save and reload `.bashrc` with 
```bash
source ~/.bashrc
```

### Finalize TensorRT Setup

1. Run `sudo ldconfig` to update library configurations.
2. Remove the old cuDNN library link (replace `x.x` with the correct version):
```bash
sudo rm /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn*.so.8
```


3. Create a new symbolic link for cuDNN:

```bash
sudo ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.x.x /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
```
---

## Phase 6: Creating a Conda Environment for TensorFlow

**Explanation**: Using a separate environment isolates TensorFlow’s dependencies from your main system.

1. Create a new environment with Python 3.9: `
```bash
conda create --name GPU python=3.9
```
2. Activate the environment: 
```bash
conda activate GPU
```

### Install TensorFlow with CUDA Support

1. Install TensorFlow with CUDA support: 
```bash
pip install tensorflow[and-cuda]
```
2. Verify TensorFlow detects the GPU:
```python
python -c "import tensorflow as tf; print(tf.test.gpu_device_name())"
```
---

## Phase 7: Installing TensorRT Python API

1. Navigate to the TensorRT Python directory: 
```bash
cd /usr/local/TensorRT-8.6.1/python
```
2. Install TensorRT Python packages:

```bash
pip install tensorrt-8.6.1-cp39-none-linux_x86_64.whl
pip install tensorrt_dispatch-8.6.1-cp39-none-linux_x86_64.whl
pip install tensorrt_lean-8.6.1-cp39-none-linux_x86_64.whl
```
---

---

## Phase 8: Verifying GPU and CUDA Compatibility

1. Run `nvidia-smi` to check GPU usage and confirm compatibility.

---

**You’re all set!** Each step of this setup is now complete, and your system should be ready for GPU-accelerated deep learning with TensorFlow.

# GPU环境搭建

如何选择和安装Nvidia显卡驱动？

如何安装CUDA和CUDNN？

## 概念

### CUDA

CUDA是由NVIDIA所推出的一种集成技术，是该公司对于GPGPU的正式名称。透过这个技术，用户可利用NVIDIA的GeForce 8以后的GPU和较新的Quadro GPU进行计算。亦是首次可以利用GPU作为C-编译器的开发环境。NVIDIA营销的时候，往往将编译器与架构混合推广，造成混乱。

#### CUDA Toolkit

包含以下组件：

- Compiler
- Tools
- Libraries
- CUDA Samples
- CUDA Driver

#### EULA

最终用户许可协议（End User License Agreement）

### CUDNN

专门为深度学习计算设计的软件库，里面提供了很多专门的计算函数，如卷积等。从上图也可以看到，还有很多其他的软件库和中间件，包括实现c++ STL的thrust、实现gpu版本blas的cublas、实现快速傅里叶变换的cuFFT、实现稀疏矩阵运算操作的cuSparse以及实现深度学习网络加速的cuDNN等等。

## 实践

### 基本安装步骤

#### 显卡驱动安装步骤

1. 确定自己电脑适合安装什么版本的显卡驱动

   查看显卡硬件型号。

   ```SAS
   $ ubuntu-drivers devices
   ```

2. 屏蔽开源驱动nouveau

   ```shell
   sudo vim /etc/modprobe.d/blacklist.conf
   ```

   添加以下内容后保存

   ```
   blacklist nouveau 
   ```

   更新使其生效

   ```shell
   $ sudo update-initramfs -u
   ```

   检查是否禁用成功

   ```shell
   $ lspci | grep nouveau
   ```

   

3. （可选）如果有旧驱动，需要先卸载

   ```
   sudo apt-get --purge remove nvidia-*
   sudo apt-get --purge remove xserver-xorg-video-nouveau
   ```

4. 重启电脑

5. 关闭X-server服务

   ```shell
   sudo service lightdm stop
   ```

   如果提示unit lightdm.service not loaded则先安装lightdm

   ```shell
   sudo apt install lightdm
   ```

   安装完毕后执行：

   ```shell
   sudo service lightdm stop
   ```

6. 下载并安装相应版本的显卡驱动

   （选择一：自动安装推荐版本）

   ```
   sudo ubuntu-drivers autoinstall
   ```

   （选择二：安装特定版本）

   ```
   sudo apt install nvidia-driver-450
   sudo apt install nvidia-driver-390
   sudo apt install xserver-xorg-core
   sudo apt install xserver-xorg-video-nouveau
   ```

   这里可能会遇到找不到源的情况，需要添加软件仓库：

   ```
   sudo add-apt-repository ppa:graphics-drivers/ppa
   ```

   （选择三：手动到[官网](https://www.nvidia.cn/Download/index.aspx?lang=cn)下载驱动并安装）

   暂略

7. 重启X-server服务

   ```shell
   $ sudo service lightdm start
   ```

8. 重启电脑

   ```shell
   $ sudo reboot
   ```

9. (可选) 配置xorg

   Nvidia提供了一个自动配置工具帮助创建xorg的配置文件(`xorg.conf`)

   可运行如下命令实现自动配置：

   ```shell
   $ nvidia-xconfig
   ```

   当Xorg的配置文件`xorg.conf`不存在时，这条命令会自动检测您的硬件，并创建文件`/etc/X11/xorg.conf`。假如配置文件已经存在的话，它会进行一些编辑，以方便在Xorg运行时能成功载入英伟达的专有驱动。

   ……

10. ……

#### CUDA安装基本步骤

如果存在旧版本，可以先卸载，以免和新版本产生冲突。（当然也可以考虑多版本共存）

1. 上cuda官网，查看哪个版本跟已安装好的nvidia显卡驱动适配，下载之

2. 安装cuda

   （使用deb）

   双击deb文件即可安装。

   （使用run文件）

   ```shell
   $ sudo sh cuda_xxxx.run
   ```

   默认安装在/usr/local/cuda-xxxx文件夹中。

   ```shell
   sudo sh cuda_10.2.89_440.33.01_linux.run --toolkit --silent --override
   ```

   

3. 配置环境变量

   ```shell
   $ vim ~/.bashrc#进入配置文件；
   # 添加以下两行：
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   
   # 使环境变量生效
   $ source ~/.bashrc
   ```

   

4. 安装多版本的cuda（可选）

   再到nvidia官网下载一个与已安装cuda不同版本的cuda，如cuda10.0的runfile，照常安装。

   不同的地方在于：一般这个时候已经安了nvidia驱动，所以当安装过程中命令行提示“Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 xxx”时，可以选no；问你是否要为当前安装版本创建软链接时，视需求而定，如果你此时需要使用这个版本的cuda就创建，否则选no，当然后面也可以改。

5. 在多个cuda版本之间进行切换

   要切换版本，只需要删除之前创建的软链接，然后再新创建一个即可。如：

   ```shell
   rm -rf /usr/local/cuda#删除之前创建的软链接
   sudo ln -s /usr/local/cuda-10.0/ /usr/local/cuda/
   nvcc --version #查看当前 cuda 版本
   ```

6. ……

#### CUDNN安装基本步骤

1. 按需求[下载cudnn安装文件](https://developer.nvidia.com/rdp/cudnn-archive)

2. 解压下载的文件，进行如下操作：

   ```shell
   $ tar -zxvf cudnn-10.2-linux-x64-v8.0.1.13.tgz
   ```

   ```shell
   $ sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
   $ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
   $ sudo chmod a+r /usr/local/cuda/include/cudnn.h
   $ sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
   ```

3. 设置软连接

   ```shell
   $ sudo ln -sf libcudnn.so.7.5.0 libcudnn.so.7
   $ sudo ln -sf libcudnn.so.7 libcudnn.so
   $ sudo ldconfig -v
   ```

4. ……

### 一些问题

#### 如何卸载Nvidia驱动？

先停止lightdm

```shell
$ sudo service lightdm stop
```

或

```shell
$ sudo /etc/init.d/lightdm stop
```

然后执行卸载命令：

```shell
sudo /usr/bin/nvidia-uninstall
```

或

```shell
sudo apt-get purge nvidia-*
sudo apt-get --purge remove xserver-xorg-video-nouveau
```

#### 使用sudo ubuntu-drivers autoinstall安装成功后执行nvidia-smi命令，报错“NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver”

```shell
$ sudo apt-get install dkms
$ sudo dkms build -m nvidia -v 450.80.2   
# 执行成功后：Module nvidia/455.45.01 already built for kernel 5.4.0-56-generic/4
$ sudo dkms install -m nvidia -v 450.80.2   
# 执行成功后：Module nvidia/455.45.01 already installed on kernel 5.4.0-56-generic/x86_64
```

注意，上面的450.80.2取决于/usr/src/目录下的nvidia-xxxx目录的版本号。如果/usr/src/下关于nvidia的文件夹名为nvidia-src-xxxx，先将其重命名为nvidia-xxx再执行以上语句。

执行完成后重启，通过如下语句查看显卡驱动情况：

```shell
$ watch -n 1 nvidia-smi
```

解决了。

#### nvcc --version显示command not found

nvcc是

### 信息查看

#### 查看版本

- 查看cuda版本

  ```shell
  $ nvcc --version
  ```

- 查看cudnn版本

  ```shell
  $ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
  $ cat cuda/include/cudnn_version.h |grep ^# 
  ```
  
- ……

## 问题（慢慢补充）

使用cuda9、cuda10、cuda11有什么区别？有什么坑吗？





### 参考资料

- [知乎:Ubuntu 18.04 安装 NVIDIA 显卡驱动](https://zhuanlan.zhihu.com/p/59618999)
- [CSDN:Ubuntu 18.04安装NVIDIA显卡驱动](https://blog.csdn.net/chentianting/article/details/85089403)

- [ubuntu环境下，系统无法与NVIDIA通信的解决方法](https://wangpei.ink/2019/01/19/NVIDIA-SMI-has-failed-because-it-couldn%27t-communicate-with-the-NVIDIA-driver的解决方法/)
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#abstract)
- [深度学习GPU环境搭建](https://xiaoyufenfei.github.io/2019/08/26/shen-du-xue-xi-gpu-huan-jing-da-jian-shang-pian/)
- [显卡，显卡驱动,nvcc, cuda driver,cudatoolkit,cudnn到底是什么？](https://www.cnblogs.com/marsggbo/p/11838823.html)
- [archlinux: NVIDIA](https://wiki.archlinux.org/index.php/NVIDIA_(简体中文))


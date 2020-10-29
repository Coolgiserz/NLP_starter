# GPU环境搭建

如何选择和安装Nvidia显卡驱动？

如何安装CUDA和CUDNN？

## 概念

### CUDA

CUDA是由NVIDIA所推出的一种集成技术，是该公司对于GPGPU的正式名称。透过这个技术，用户可利用NVIDIA的GeForce 8以后的GPU和较新的Quadro GPU进行计算。亦是首次可以利用GPU作为C-编译器的开发环境。NVIDIA营销的时候，往往将编译器与架构混合推广，造成混乱。

### CUDNN

GPU加速。



## 实践

### 基本安装步骤

#### 显卡驱动安装步骤

1. 确定自己电脑适合安装什么版本的显卡驱动

   查看显卡硬件型号。

   ```SAS
   $ ubuntu-drivers devices
   ```

2. 下载并安装相应版本的显卡驱动

   （选择一：自动安装推荐版本）

   ```
   sudo ubuntu-drivers autoinstall
   ```

   

   （选择二：安装特定版本）

   ```
   sudo apt install nvidia-450
   ```

   这里可能会遇到找不到源的情况，需要添加软件仓库：

   ```
   sudo add-apt-repository ppa:graphics-drivers/ppa
   ```

   

   （选择三：手动到官网下载驱动并安装）

   暂略

3. ……

#### CUDA安装基本步骤

如果存在旧版本，可以先卸载，以免和新版本产生冲突。（当然也可以考虑多版本共存）

1. 上cuda官网，查看哪个版本跟已安装好的nvidia显卡驱动适配，下载之
2. 安装cuda

### 一些问题

####使用sudo ubuntu-drivers autoinstall安装成功后执行nvidia-smi命令，报错“NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver”

```shell
$ sudo apt-get install dkms
$ sudo dkms build -m nvidia -v 450.80.2   
$ sudo dkms install -m nvidia -v 450.80.2   
```

注意，上面的450.80.2取决于/usr/src/目录下的nvidia-xxxx目录的版本号。如果/usr/src/下关于nvidia的文件夹名为nvidia-src-xxxx，先将其重命名为nvidia-xxx再执行以上语句。

执行完成后重启，通过如下语句查看显卡驱动情况：

```shell
$ watch -n 1 nvidia-smi
```

解决了。

### 信息查看

查看版本

## 问题（慢慢补充）

使用cuda9、cuda10、cuda11有什么区别？有什么坑吗？



### 参考资料

- Ubuntu 18.04 安装 NVIDIA 显卡驱动：https://zhuanlan.zhihu.com/p/59618999
- ubuntu环境下，系统无法与NVIDIA通信的解决方法：https://wangpei.ink/2019/01/19/NVIDIA-SMI-has-failed-because-it-couldn%27t-communicate-with-the-NVIDIA-driver的解决方法/
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#abstract)
- 深度学习GPU环境搭建：https://xiaoyufenfei.github.io/2019/08/26/shen-du-xue-xi-gpu-huan-jing-da-jian-shang-pian/


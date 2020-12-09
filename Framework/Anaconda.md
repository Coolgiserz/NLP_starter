# Anaconda

## 简介

数据科学平台。

## 入门

### 安装

1. 下载安装文件(.sh)

2. 安装Anaconda

   过程需要配置安装路径，可默认安装到用户家目录下；

   询问是否需要初始化Anaconda3，选择yes。如果选了no，则需要自己配置环境变量（第三步）

3. 配置环境变量

   ```shell
   $ vim ~/.bashrc
   
   export PATH=/home/USER_NAME/anaconda3/bin:$PATH
   ```

4. 更改权限

   刚安装完默认anaconda3目录所属用户和组都是root，如果想让非超级用户使用之，需要更改目录的所有权

   ```shell
   sudo chown -R USER_NAME anaconda3/
   ```

5. ……

### 基本使用

#### 管理源

##### 添加源

创建~/.condarc文件，把源添加进去，如：

```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
```

也可以通过命令添加源，如：

```shell
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

也可以在具体安装的时候添加-c参数指定源。

##### 换回原来的源

```shell
$ conda config --remove channels
```

#### 虚拟环境管理

##### 创建虚拟环境

##### 删除虚拟环境

##### 切换虚拟环境

##### 备份虚拟环境

备份conda

```shell
# 把环境中的包导出，方便在不同的平台和操作系统之间复现项目环境
$ conda env export > environment_py36.yml
```

或者

```shell
# 复制环境
$ conda create -n python35copy --clone python35
```

还可以通过conda-pack打包conda环境

```shell
# 1. 安装命令
conda install -c conda-forge conda-pack
# 2. 打包环境
## Pack environment my_env into my_env.tar.gz
conda pack -n my_env

## Pack environment my_env into out_name.tar.gz
conda pack -n my_env -o out_name.tar.gz

## Pack environment located at an explicit path into my_env.tar.gz
conda pack -p /explicit/path/to/my_env

# 3. 重现环境
# Unpack environment into directory `my_env`
mkdir -p my_env
tar -xzf my_env.tar.gz -C my_env
# Use Python without activating or fixing the prefixes. Most Python
# libraries will work fine, but things that require prefix cleanups
# will fail.
./my_env/bin/python
# Activate the environment. This adds `my_env/bin` to your path
source my_env/bin/activate
# Run Python from in the environment
(my_env) $ python
# Cleanup prefixes from in the active environment.
# Note that this command can also be run without activating the environment
# as long as some version of Python is already installed on the machine.
(my_env) $ conda-unpack
```



## 技巧

### 局域网内电脑访问jupyter

1. 生成配置文件

   ```
   jupyter notebook --generate-config
   ```

2. 修改配置

   ```
   c.ConnectionFileMixin.ip = '0.0.0.0'
   c.NotebookApp.ip = '0.0.0.0'
   ```

3. 开启防火墙的端口/关闭防火墙

4. 正常开启jupyter

## FAQ

##### 可能会遇到Pytorch安装速度过慢的问题

可通过更换源避免该问题。

## 参考

- Anaconda 镜像使用帮助：https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
- [Conda 环境迁移](https://zhuanlan.zhihu.com/p/87344422)
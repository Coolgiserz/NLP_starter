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
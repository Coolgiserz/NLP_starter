# Misc

或许有助于提升效率的工具：

## 0x01 IDE

#### Pycharm

#### VSCode

##### 1. 安装VSCode

到官网安装或者在命令行通过如下命令安装：

```shell
$ sudo apt update
$ sudo apt install software-properties-common apt-transport-https wget
$ wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
$ sudo add-apt-repository "deb [arch=amd64]https://packages.microsoft.com/repos/vscode stable main"
$ sudo apt install code

```

##### 2. 安装相应插件

如果进行Python开发

- python
- ……

## 0x02 远程工作

#### 远程桌面

- 向日葵
- Teamviewer
- ……

#### 远程终端

##### ssh

1. 在服务器（Ubuntu）上安装ssh服务器：

```shell
$ sudo apt-get install openssh-server
```

​	安装完成后，ssh服务会自动启动，可通过如下命令验证ssh的安装及运行状态

```shell
$ sudo systemctl status ssh
```

​	要保证客户端能够连接，需要允许防火墙开放ssh端口:

```shell
$ sudo ufw allow ssh
```

2. 在本机安装ssh客户端
3. 通过ssh连接远程服务器

```shell
$ ssh username@ip
```

4. (可选)

   如果是针对以前连接过的机器（ip地址不变），但机器的物理环境发生了很大变换（可能是重装过系统导致的），则原本的密钥可能不再合适，通过ssh连接时会出现如下报错。

   ```
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   @    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
   Someone could be eavesdropping on you right now (man-in-the-middle attack)!
   It is also possible that a host key has just been changed.
   The fingerprint for the ECDSA key sent by the remote host is
   SHA256:B3c4T0wju6KSV/RhlJgMS6P4qjmjCL1Y7HGkLOQ42uA.
   Please contact your system administrator.
   Add correct host key in /Users/zhuge/.ssh/known_hosts to get rid of this message.
   Offending ECDSA key in /Users/zhuge/.ssh/known_hosts:26
   ECDSA host key for 192.168.0.xxx has changed and you have requested strict checking.
   Host key verification failed.
   ```

   只需要清除缓存的密钥，重新生成ssh连接密钥即可

   ```shell
   $ ssh-keygen -R <SERVER_IP>
   ```

   ```
   # Host 192.168.0.xxx found: line 26
   /Users/xxx/.ssh/known_hosts updated.
   Original contents retained as /Users/xxx/.ssh/known_hosts.old
   ```

- ……

#### 远程文件传输

##### scp

Linux scp 命令用于 Linux 之间复制文件和目录。scp 是 secure copy 的缩写, scp 是 linux 系统下基于 ssh 登陆进行安全的远程文件拷贝命令。还有一个命令叫rcp，[rcp](https://www.runoob.com/linux/linux-comm-rcp.html) 是不加密的，scp 是 rcp 的加强版。

从本地复制到远程：

```shell
$ scp local_file remote_username@remote_ip:remote_folder 
或者 
$ scp local_file remote_username@remote_ip:remote_file 
或者 
$ scp local_file remote_ip:remote_folder 
或者 
$ scp local_file remote_ip:remote_file 
```

详情参见[文档](https://www.runoob.com/linux/linux-comm-scp.html)

## 0x03 VPN

#### 小火箭

支持Mac、Windows

#### v2ray+Qv2ray

1. 下载安装程序

   (1) 下载Qv2ray二进制文件AppImage

   ​	下载后为该文件赋以可执行权限，运行之。

   ​	或者直接通过如下命令安装qv2ray

      ```shell
   $ sudo snap install qv2ray
   # 安装成功后会提示类似信息：
   # 2.6.3.5841 from ymshenyu installed
      ```

   (2) 下载v2ray核心包

   ​	下载完后解压

   ```shell
   $ unzip v2ray-linux-64.zip -d vcore
   ```

   ```shell
   mkdir -p ~/.config/qv2ray/vcore/
   mv vcore/* ~/.config/qv2ray/vcore/
   ```

2. 注册账号并

   http://clyun.club/auth/register

3. ……

   

   

- ……

### 版本控制系统：Git

### 数据下载工具

####kaggle-api

1. 安装kaggle工具

```shell
$ pip install kaggle
```

2. 登陆kaggle网站，创建[API token](https://www.kaggle.com/coolgiser/account)

   会自动下载一个kaggle.json文件，并提示

   ```
   Ensure kaggle.json is in the location ~/.kaggle/kaggle.json to use the API.
   ```

   照提示做，并修改kaggle.json权限，保证仅自己可见（安全起见）

   ```shell
   chmod 600 /home/USERNAME/.kaggle/kaggle.json
   ```

3. 使用kaggle-api查看、下载数据集

   示例：

   ```shell
   (base) USERNAME@USERNAME-Desktop:~/Downloads$ kaggle competitions list -s health
   ref                                      deadline             category       reward  teamCount  userHasEntered
   ---------------------------------------  -------------------  ----------  ---------  ---------  --------------
   hhp                                      2013-04-04 07:00:00  Featured     $500,000       1350           False
   stanford-covid-vaccine                   2020-10-06 23:59:00  Research      $25,000       1636           False
   osic-pulmonary-fibrosis-progression      2020-10-06 23:59:00  Featured      $55,000       2097           False
   rsna-str-pulmonary-embolism-detection    2020-10-26 23:59:00  Featured      $30,000        784           False
   lish-moa                                 2020-11-30 23:59:00  Research      $30,000       4373           False
   hubmap-kidney-segmentation               2021-02-01 23:59:00  Research      $60,000        299           False
   datasciencebowl                          2015-03-16 23:59:00  Featured     $175,000       1049           False
   nfl-impact-detection                     2021-01-04 23:59:00  Featured      $75,000        186           False
   histopathologic-cancer-detection         2019-03-30 23:59:00  Playground  Knowledge       1157           False
   melbourne-university-seizure-prediction  2016-12-01 23:59:00  Research      $20,000        477           False
   covid19-global-forecasting-week-1        2020-03-25 23:59:00  Research    Knowledge        544           False
   aptos2019-blindness-detection            2019-09-07 23:59:00  Featured      $50,000       2931           False
   ultrasound-nerve-segmentation            2016-08-18 23:59:00  Featured     $100,000        922           False
   prostate-cancer-grade-assessment         2020-07-22 23:59:00  Featured      $25,000       1010           False
   trends-assessment-prediction             2020-06-29 23:59:00  Research      $25,000       1047           False
   covid19-global-forecasting-week-4        2020-04-15 23:59:00  Research    Knowledge        472           False
   covid19-global-forecasting-week-3        2020-04-08 23:59:00  Research    Knowledge        452            True
   covid19-global-forecasting-week-2        2020-04-06 03:56:00  Research    Knowledge        215           False
   instacart-market-basket-analysis         2017-08-14 23:59:00  Featured      $25,000       2622           False
   covid19-local-us-ca-forecasting-week-1   2020-03-25 23:59:00  Research    Knowledge        190           False
   ```

   ```shell
   (base) USERNAME@USERNAME-Desktop:~/Downloads$ kaggle competitions files favorita-grocery-sales-forecasting
   
   name                       size  creationDate
   ------------------------  -----  -------------------
   sample_submission.csv.7z  651KB  2018-06-20 06:10:54
   stores.csv.7z              648B  2018-06-20 06:10:54
   items.csv.7z               14KB  2018-06-20 06:10:54
   holidays_events.csv.7z      2KB  2018-06-20 06:10:54
   test.csv.7z                 5MB  2018-06-20 06:10:54
   transactions.csv.7z       214KB  2018-06-20 06:10:54
   train.csv.7z              452MB  2018-06-20 06:10:54
   oil.csv.7z                  4KB  2018-06-20 06:10:54
   ```

   

   详情参见[文档](https://github.com/Kaggle/kaggle-api)

## 参考

[How to Enable SSH on Ubuntu 18.04](https://linuxize.com/post/how-to-enable-ssh-on-ubuntu-18-04/)

[How to fix warning about ECDSA host key](https://superuser.com/questions/421004/how-to-fix-warning-about-ecdsa-host-key)


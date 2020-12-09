# Misc

或许有助于提升效率的工具：

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

#### VPN

- 小火箭
- ……

## 参考

[How to Enable SSH on Ubuntu 18.04](https://linuxize.com/post/how-to-enable-ssh-on-ubuntu-18-04/)

[How to fix warning about ECDSA host key](https://superuser.com/questions/421004/how-to-fix-warning-about-ecdsa-host-key)


## 设置使用 PyCharm 和 Anaconda 的 Python 开发环境

<h4 align="center">
    <p>
        <a href="https://github.com/Collaborative-AI/tutorial/blob/main/Python/README.md">English</a>
        <b>简体中文</b> |
    </p>
</h4>

## 目录
1. [介绍](#介绍)
2. [步骤1：验证现有的 Python 安装](#步骤1-验证现有的-python-安装)
    - [Windows](#windows)
    - [Mac](#mac)
    - [Linux](#linux)
3. [步骤2：安装 Anaconda](#步骤2-安装-anaconda)
    - [下载 Anaconda](#下载-anaconda)
    - [安装](#安装)
        - [Windows](#windows-1)
        - [Mac](#mac-1)
        - [Linux](#linux-1)
    - [验证 Anaconda 安装](#验证-anaconda-安装)
4. [步骤3：安装 PyCharm](#步骤3-安装-pycharm)
    - [下载 PyCharm](#下载-pycharm)
    - [安装](#安装-1)
        - [Windows](#windows-2)
        - [Mac](#mac-2)
        - [Linux](#linux-2)
5. [步骤4：配置 PyCharm 和 Anaconda](#步骤4-配置-pycharm-和-anaconda)
    - [打开 PyCharm](#打开-pycharm)
    - [设置 Anaconda 环境](#设置-anaconda-环境)
    - [配置终端为 Conda](#配置终端为-conda)
6. [步骤5：排除数据和输出文件夹](#步骤5-排除数据和输出文件夹)
    - [标记为排除](#标记为排除)
7. [其他提示](#其他提示)

## 介绍
本教程将指导您使用 PyCharm 和 Anaconda 设置一个稳健的 Python 开发环境。我们将涵盖不同操作系统的安装步骤，验证现有的 Python 安装，并配置 PyCharm 以提高编码效率。

## 步骤1：验证现有的 Python 安装
在安装新软件之前，检查是否已有 Python 安装以避免冲突。

### Windows
- 打开命令提示符并输入：`python --version` 或 `py --version`。
- 检查已安装的 Python 版本，并从控制面板中卸载不必要的版本。

### Mac
- 打开终端并输入：`python --version` 或 `python3 --version`。
- 卸载时，从 `/Library/Frameworks/Python.framework/Versions/` 中删除 Python 目录。

### Linux
- 打开终端并输入：`python --version` 或 `python3 --version`。
- 使用您的包管理器卸载不必要的 Python 版本（例如，`sudo apt-get remove python`）。

## 步骤2：安装 Anaconda
安装最新的 Anaconda 发行版，其中包括许多预安装的数据科学库和工具。

### 下载 Anaconda
- 前往 [Anaconda 下载页面](https://www.anaconda.com/products/distribution)，下载适用于您的操作系统的安装程序。

### 安装

#### Windows
- 运行下载的安装程序。
- 选择“仅我”，并在安装过程中勾选“将 Anaconda 添加到我的 PATH 环境变量”。
- 完成安装过程。

#### Mac
- 打开终端并运行下载的 `.sh` 脚本：`bash Anaconda3-2023.03-MacOSX-x86_64.sh`（版本号可能有所不同）。
- 按照提示完成安装。

#### Linux
- 打开终端并运行下载的 `.sh` 脚本：`bash Anaconda3-2023.03-Linux-x86_64.sh`（版本号可能有所不同）。
- 按照提示完成安装。

### 验证 Anaconda 安装
- 打开一个新的终端窗口。
- 输入 `python`，并验证输出中是否包含“packaged by Anaconda, Inc.”

## 步骤3：安装 PyCharm
PyCharm 是一个专为 Python 开发设计的强大 IDE。

### 下载 PyCharm
- 前往 [PyCharm 下载页面](https://www.jetbrains.com/pycharm/download/)，下载适用于您的操作系统的社区版。

### 安装

#### Windows
- 运行下载的安装程序并按照安装向导进行操作。

#### Mac
- 打开下载的 `.dmg` 文件并将 PyCharm 拖到应用程序文件夹。

#### Linux
- 解压下载的压缩包并从 `bin` 子目录运行 `./pycharm.sh`。

## 步骤4：配置 PyCharm 和 Anaconda

### 打开 PyCharm
创建一个新项目或打开一个现有项目。

### 设置 Anaconda 环境
- 进入 `File` > `Settings`（Mac 上为 `PyCharm` > `Preferences`）。
- 导航到 `Project: <project_name>` > `Python Interpreter`。
- 点击齿轮图标并选择 `Add`。
- 选择 `Conda Environment` 并指定 Anaconda 安装路径。

### 配置终端为 Conda
- 如果命令提示符前没有出现 `(base)` 前缀，请通过 `File` > `Settings` > `Tools` > `Terminal` 设置终端为 Conda，设置 shell 路径为 Conda 终端可执行文件，通常路径为：
  - **Windows**: `C:\Users\<YourUsername>\Anaconda3\Scripts\activate.bat`
  - **Mac/Linux**: `/Users/<YourUsername>/anaconda3/bin/activate` 或 `/home/<YourUsername>/anaconda3/bin/activate`

## 步骤5：排除数据和输出文件夹

### 标记为排除
- 在项目工具窗口中右键点击数据/输出文件夹。
- 选择 `Mark Directory as` > `Excluded`。

## 其他提示
- **使用终端而不是 PyCharm 运行器**：
  - 建议使用终端运行脚本，以避免内置运行器的问题。
  
- **代码导航**：
  - 使用 PyCharm 的优势之一是能够通过 `Ctrl + 左键点击` 导航到定义。
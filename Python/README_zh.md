# 使用 PyCharm 和 Anaconda 设置 Python 开发环境

<h4 align="center">
    <p>
        <a href="https://github.com/Collaborative-AI/tutorial/blob/main/Python/README.md">English</a>
        <b>简体中文</b> |
    </p>
</h4>


## 目录
1. [介绍](#介绍)
2. [步骤 1：验证现有的 Python 安装](#步骤-1-验证现有的-python-安装)
    - [Windows](#windows)
    - [Mac](#mac)
    - [Linux](#linux)
3. [步骤 2：安装 Anaconda](#步骤-2-安装-anaconda)
    - [下载 Anaconda](#下载-anaconda)
    - [安装](#安装)
        - [Windows](#windows-1)
        - [Mac](#mac-1)
        - [Linux](#linux-1)
    - [验证 Anaconda 安装](#验证-anaconda-安装)
4. [步骤 3：安装 PyCharm](#步骤-3-安装-pycharm)
    - [PyCharm 版本](#pycharm-版本)
    - [下载 PyCharm](#下载-pycharm)
    - [安装](#安装-1)
        - [Windows](#windows-2)
        - [Mac](#mac-2)
        - [Linux](#linux-2)
5. [步骤 4：配置 PyCharm 与 Anaconda](#步骤-4-配置-pycharm-与-anaconda)
    - [打开 PyCharm](#打开-pycharm)
    - [设置 Anaconda 环境](#设置-anaconda-环境)
    - [配置终端使用 Conda](#配置终端使用-conda)
6. [步骤 5：排除数据和输出文件夹](#步骤-5-排除数据和输出文件夹)
    - [标记为排除](#标记为排除)
7. [附加提示](#附加提示)

## 介绍
本教程将指导您使用 PyCharm 和 Anaconda 设置一个强大的 Python 开发环境。我们将涵盖不同操作系统的安装步骤，验证现有的 Python 安装，并配置 PyCharm 以实现高效编码。

## 步骤 1：验证现有的 Python 安装
在安装新软件之前，请检查是否有任何现有的 Python 安装以避免冲突。

### Windows
- 打开命令提示符并输入：`python --version` 或 `py --version`。
- 检查已安装的 Python 版本，并从控制面板中卸载不必要的版本。

### Mac
- 打开终端并输入：`python --version` 或 `python3 --version`。
- 要卸载，请从 `/Library/Frameworks/Python.framework/Versions/` 中删除 Python 目录。

### Linux
- 打开终端并输入：`python --version` 或 `python3 --version`。
- 使用您的包管理器卸载不必要的 Python 版本（例如，`sudo apt-get remove python`）。

## 步骤 2：安装 Anaconda
安装最新的 Anaconda 发行版，其中包括众多预安装的用于数据科学的库和工具。

### 下载 Anaconda
- 访问 [Anaconda 下载页面](https://www.anaconda.com/products/distribution) 并下载适用于您操作系统的安装程序。

### 安装

#### Windows
- 运行下载的安装程序。
- 选择“仅我自己”，并在安装过程中勾选“将 Anaconda 添加到我的 PATH 环境变量”。
- 完成安装过程。

#### Mac
- 打开终端并运行下载的 `.sh` 脚本：`bash Anaconda3-2023.03-MacOSX-x86_64.sh`（版本号可能有所不同）。
- 按提示完成安装。

#### Linux
- 打开终端并运行下载的 `.sh` 脚本：`bash Anaconda3-2023.03-Linux-x86_64.sh`（版本号可能有所不同）。
- 按提示完成安装。

### 验证 Anaconda 安装
- 打开一个新的终端窗口。
- 输入 `python` 并验证输出是否包含“packaged by Anaconda, Inc.”。

## 步骤 3：安装 PyCharm
PyCharm 是一个专为 Python 开发设计的强大 IDE。

### PyCharm 版本
PyCharm 有几个版本：

- **社区版**：免费且开源，适合纯 Python 开发。
- **专业版**：付费版本，具有用于 Web 开发、数据库管理等的附加功能。
- **教育版**：免费用于教育目的，包含专业版的功能。您可以在 [这里](https://www.jetbrains.com/community/education/#students) 了解更多并申请教育许可证。

### 下载 PyCharm
- 访问 [PyCharm 下载页面](https://www.jetbrains.com/pycharm/download/) 并下载适用于您操作系统的适当版本。

### 安装

#### Windows
- 运行下载的安装程序并按照安装向导操作。
- 选择“将 PyCharm 添加到上下文菜单（右键菜单）”选项以便于访问。
- 可选地，选择创建桌面快捷方式。

#### Mac
- 打开下载的 `.dmg` 文件并将 PyCharm 拖到应用程序文件夹中。

#### Linux
- 解压下载的 tarball 文件，并从 `bin` 子目录中运行 `./pycharm.sh`。

## 步骤 4：配置 PyCharm 与 Anaconda

### 打开 PyCharm
创建一个新项目或打开一个现有项目。

### 设置 Anaconda 环境
- 转到 `File` > `Settings`（在 Mac 上为 `PyCharm` > `Preferences`）。
- 导航到 `Project: <project_name>` > `Python Interpreter`。
- 点击齿轮图标并选择 `Add`。
- 选择 `Conda Environment` 并指定 Anaconda 安装路径。

### 配置终端使用 Conda
- 如果 `(base)` 前缀没有出现在命令提示符前，请通过转到 `File` > `Settings` > `Tools` > `Terminal` 并将 shell 路径设置为 Conda 终端可执行文件来设置默认终端为 Conda。

## 步骤 5：排除数据和输出文件夹

### 标记为排除
- 在项目工具窗口中右键点击数据/输出文件夹。
- 选择 `Mark Directory as` > `Excluded`。

## 附加提示
- **使用终端而不是 PyCharm 运行器**：
  - 建议使用终端运行脚本，以避免内置运行器的问题。
  
- **代码导航**：
  - 使用 PyCharm 的一个优势是可以通过 `Ctrl + 左键点击` 导航到定义。

- **教育许可证**：
  - 如果您是学生或教育工作者，可以通过 JetBrains 网站申请免费的 PyCharm 专业版教育许可证，申请地址 [这里](https://www.jetbrains.com/community/education/#students)。

通过遵循这些步骤，您将拥有一个强大且高效的 Python 开发环境，并且已设置好 PyCharm 和 Anaconda，随时准备进行任何数据科学或开发项目。
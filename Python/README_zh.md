### 使用 PyCharm 和 Anaconda 设置 Python 开发环境
<h4 align="center">
    <p>
        <a href="https://github.com/Collaborative-AI/tutorial/blob/main/Python/README.md">English</a>
        <b>简体中文</b> |
    </p>
</h4>

本教程将指导您使用 PyCharm 和 Anaconda 设置一个强大的 Python 开发环境。我们将介绍不同操作系统的安装步骤，验证现有的 Python 安装，并配置 PyCharm 以实现高效编码。

#### 步骤 1：验证现有的 Python 安装

在安装新软件之前，检查现有的 Python 安装以避免冲突。

1. **Windows**:
   - 打开命令提示符并输入：`python --version` 或 `py --version`。
   - 检查已安装的 Python 版本，并从控制面板中卸载不必要的版本。

2. **Mac**:
   - 打开终端并输入：`python --version` 或 `python3 --version`。
   - 要卸载，请删除 `/Library/Frameworks/Python.framework/Versions/` 中的 Python 目录。

3. **Linux**:
   - 打开终端并输入：`python --version` 或 `python3 --version`。
   - 使用您的包管理器卸载不必要的 Python 版本（例如 `sudo apt-get remove python`）。

#### 步骤 2：安装 Anaconda

安装最新的 Anaconda 发行版，其中包含众多预安装的库和数据科学工具。

1. **下载 Anaconda**:
   - 访问 [Anaconda 下载页面](https://www.anaconda.com/products/distribution) 并下载适用于您的操作系统的安装程序。

2. **安装**:
   - **Windows**:
     - 运行下载的安装程序。
     - 在安装过程中选择“Just Me”并勾选“Add Anaconda to my PATH environment variable”。
     - 完成安装过程。

   - **Mac**:
     - 打开终端并运行下载的 `.sh` 脚本：`bash Anaconda3-2023.03-MacOSX-x86_64.sh`（版本号可能会有所不同）。
     - 按提示完成安装。

   - **Linux**:
     - 打开终端并运行下载的 `.sh` 脚本：`bash Anaconda3-2023.03-Linux-x86_64.sh`（版本号可能会有所不同）。
     - 按提示完成安装。

3. **验证 Anaconda 安装**:
   - 打开一个新的终端窗口。
   - 输入 `python` 并验证输出中是否包含 "packaged by Anaconda, Inc."。

#### 步骤 3：安装 PyCharm

PyCharm 是一个专为 Python 开发设计的强大 IDE。

1. **下载 PyCharm**:
   - 访问 [PyCharm 下载页面](https://www.jetbrains.com/pycharm/download/) 并下载适用于您的操作系统的社区版。

2. **安装**:
   - **Windows**:
     - 运行下载的安装程序并按照安装向导进行操作。

   - **Mac**:
     - 打开下载的 `.dmg` 文件并将 PyCharm 拖动到应用程序文件夹中。

   - **Linux**:
     - 解压下载的压缩包，并从 `bin` 子目录中运行 `./pycharm.sh`。

#### 步骤 4：使用 Anaconda 配置 PyCharm

1. **打开 PyCharm** 并创建一个新项目或打开一个现有项目。

2. **设置 Anaconda 环境**:
   - 进入 `File` > `Settings`（Mac 上为 `PyCharm` > `Preferences`）。
   - 导航到 `Project: <project_name>` > `Python Interpreter`。
   - 点击齿轮图标并选择 `Add`。
   - 选择 `Conda Environment` 并指定 Anaconda 安装路径。

3. **配置终端以使用 Conda（如果需要）**:
   - 如果 `(base)` 前缀未出现，通过进入 `File` > `Settings` > `Tools` > `Terminal` 并将 Shell 路径设置为 Conda 终端可执行文件来设置默认终端。

#### 步骤 5：排除数据和输出文件夹

为了确保 PyCharm 顺利运行，排除数据和输出文件夹以避免索引：

1. **标记为排除**:
   - 在项目工具窗口中右键单击数据/输出文件夹。
   - 选择 `Mark Directory as` > `Excluded`。

#### 其他提示

- **使用终端而非 PyCharm 运行器**:
  - 建议使用终端运行脚本以避免内置运行器的问题。

- **代码导航**:
  - 使用 PyCharm 相对于 VSCode 的一个优势是可以通过 `Ctrl + 左键单击` 导航到定义。

---

这样调整后的教程可以更清晰地指导用户如何设置 Python 开发环境。
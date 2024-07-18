# Setting Up a Python Development Environment with PyCharm and Anaconda

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/Collaborative-AI/tutorial/blob/main/Python/README_zh.md">简体中文</a>
    </p>
</h4>


## Table of Contents
1. [Introduction](#introduction)
2. [Step 1: Verify Existing Python Installations](#step-1-verify-existing-python-installations)
    - [Windows](#windows)
    - [Mac](#mac)
    - [Linux](#linux)
3. [Step 2: Install Anaconda](#step-2-install-anaconda)
    - [Download Anaconda](#download-anaconda)
    - [Installation](#installation)
        - [Windows](#windows-1)
        - [Mac](#mac-1)
        - [Linux](#linux-1)
    - [Verify Anaconda Installation](#verify-anaconda-installation)
4. [Step 3: Install PyCharm](#step-3-install-pycharm)
    - [PyCharm Editions](#pycharm-editions)
    - [Download PyCharm](#download-pycharm)
    - [Installation](#installation-1)
        - [Windows](#windows-2)
        - [Mac](#mac-2)
        - [Linux](#linux-2)
5. [Step 4: Configure PyCharm with Anaconda](#step-4-configure-pycharm-with-anaconda)
    - [Open PyCharm](#open-pycharm)
    - [Set up Anaconda Environment](#set-up-anaconda-environment)
    - [Configure Terminal for Conda](#configure-terminal-for-conda)
6. [Step 5: Exclude Data and Output Folders](#step-5-exclude-data-and-output-folders)
    - [Mark as Excluded](#mark-as-excluded)
7. [Additional Tips](#additional-tips)

## Introduction
This tutorial will guide you through setting up a robust Python development environment using PyCharm and Anaconda. We'll cover installation steps for different operating systems, verify your existing Python installations, and configure PyCharm for efficient coding.

## Step 1: Verify Existing Python Installations
Before installing new software, check for any existing Python installations to avoid conflicts.

### Windows
- Open Command Prompt and type: `python --version` or `py --version`.
- Check installed Python versions and uninstall unnecessary ones from the Control Panel.

### Mac
- Open Terminal and type: `python --version` or `python3 --version`.
- To uninstall, remove the Python directories from `/Library/Frameworks/Python.framework/Versions/`.

### Linux
- Open Terminal and type: `python --version` or `python3 --version`.
- Use your package manager to uninstall unnecessary Python versions (e.g., `sudo apt-get remove python`).

## Step 2: Install Anaconda
Install the latest Anaconda distribution, which includes numerous pre-installed libraries and tools for data science.

### Download Anaconda
- Go to the [Anaconda download page](https://www.anaconda.com/products/distribution) and download the installer for your operating system.

### Installation

#### Windows
- Run the downloaded installer.
- Select "Just Me" and check "Add Anaconda to my PATH environment variable" during installation.
- Complete the installation process.

#### Mac
- Open Terminal and run the downloaded `.sh` script: `bash Anaconda3-2023.03-MacOSX-x86_64.sh` (version number may vary).
- Follow the prompts to complete the installation.

#### Linux
- Open Terminal and run the downloaded `.sh` script: `bash Anaconda3-2023.03-Linux-x86_64.sh` (version number may vary).
- Follow the prompts to complete the installation.

### Verify Anaconda Installation
- Open a new terminal window.
- Type `python` and verify the output includes "packaged by Anaconda, Inc."

## Step 3: Install PyCharm
PyCharm is a powerful IDE specifically designed for Python development.

### PyCharm Editions
PyCharm comes in several editions:

- **Community Edition**: Free and open-source, suitable for pure Python development.
- **Professional Edition**: Paid version with additional features for web development, database management, and more.
- **Educational Edition**: Free for educational purposes, includes features included in the Professional Edition. You can learn more and apply for an educational license [here](https://www.jetbrains.com/community/education/#students).

### Download PyCharm
- Go to the [PyCharm download page](https://www.jetbrains.com/pycharm/download/) and download the appropriate edition for your operating system.

### Installation

#### Windows
- Run the downloaded installer and follow the installation wizard.
- Select the option to "Add PyCharm to context menu (right-click menu)" for easier access.
- Optionally, choose to create a desktop shortcut.

#### Mac
- Open the downloaded `.dmg` file and drag PyCharm to the Applications folder.

#### Linux
- Extract the downloaded tarball and run `./pycharm.sh` from the `bin` subdirectory.

## Step 4: Configure PyCharm with Anaconda

### Open PyCharm
Create a new project or open an existing one.

### Set up Anaconda Environment
- Go to `File` > `Settings` (or `PyCharm` > `Preferences` on Mac).
- Navigate to `Project: <project_name>` > `Python Interpreter`.
- Click the gear icon and select `Add`.
- Choose `Conda Environment` and specify the path to your Anaconda installation.

### Configure Terminal for Conda
- If the `(base)` prefix does not appear in front of the command prompt, set the default terminal to Conda by going to `File` > `Settings` > `Tools` > `Terminal` and setting the shell path to the Conda terminal executable.

## Step 5: Exclude Data and Output Folders

### Mark as Excluded
- Right-click on the data/output folder in the Project tool window.
- Select `Mark Directory as` > `Excluded`.

## Additional Tips
- **Using Terminal Instead of PyCharm Runner**:
  - It's recommended to use the Terminal for running scripts to avoid issues with the built-in runner.
  
- **Code Navigation**:
  - One of the advantages of using PyCharm over VSCode is the ability to navigate to definitions with `Ctrl + Left Click`.

- **Educational Licenses**:
  - If you're a student or educator, you can apply for a free educational license for PyCharm Professional Edition through the JetBrains website [here](https://www.jetbrains.com/community/education/#students).

By following these steps, you will have a powerful and efficient Python development environment set up with PyCharm and Anaconda, ready for any data science or development projects.
# Git
<h4 align="center">
    <p>
        <a href="https://github.com/Collaborative-AI/tutorial/blob/main/Git/README.md">English</a>
        <b>简体中文</b> |
    </p>
</h4>

## 目录
1. [简介](#简介)
2. [安装 Git](#安装-git)
3. [配置用户名和密码](#配置用户名和密码)
4. [基本的 Git 命令](#基本的-git-命令)
    - [git clone](#git-clone)
    - [git pull](#git-pull)
    - [git branch](#git-branch)
    - [git checkout](#git-checkout)
    - [git status](#git-status)
    - [git add --all](#git-add---all)
    - [git reset](#git-reset)
    - [git commit -m "[一些信息在这里]"](#git-commit--m-一些信息在这里)
    - [git push](#git-push)
5. [将一个仓库合并到另一个仓库](#将一个仓库合并到另一个仓库)

## 简介
Git 是一个分布式版本控制系统，用于在软件开发过程中跟踪源代码的更改。GitHub 是一个基于 Web 的平台，它使用 Git 并提供协作、代码审查和项目管理等附加功能。本教程将引导你了解使用 Git 和 GitHub 的基础知识。

## 安装 Git
### Windows
1. 从 [git-scm.com](https://git-scm.com/downloads) 下载 Git 安装程序。
2. 运行安装程序并按照屏幕上的说明进行操作（默认设置即可）。
3. 验证安装，在命令提示符中输入：
   ```bash
   git --version
   ```

### MacOS
1. 如果尚未安装 Homebrew，请安装 Homebrew：
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. 使用 Homebrew 安装 Git：
   ```bash
   brew install git
   ```
3. 验证安装，在终端中输入：
   ```bash
   git --version
   ```

### Linux
1. 打开终端。
2. 使用包管理器安装 Git：
   - **Debian/Ubuntu:**
     ```bash
     sudo apt-get update
     sudo apt-get install git
     ```
   - **Fedora:**
     ```bash
     sudo dnf install git
     ```
3. 验证安装，输入：
   ```bash
   git --version
   ```

## 配置用户名和密码
安装 Git 后，你需要设置用户名和电子邮件地址。这很重要，因为每次 Git 提交都会使用这些信息。此外，当你将提交推送到 GitHub 时，这些详细信息将用于将提交归属于你的 GitHub 帐户。

1. 设置用户名：
   ```bash
   git config --global user.name "Your Name"
   ```

2. 设置电子邮件地址：
   ```bash
   git config --global user.email "your.email@example.com"
   ```

3. 配置 Git 存储你的凭据：
   ```bash
   git config --global credential.helper store
   ```

4. 下次执行需要身份验证的 Git 操作（例如 `git push`）时，Git 会提示你输入用户名和密码，并将它们存储在主目录的纯文本文件中 (`~/.git-credentials`)。

有关将 Git 连接到 GitHub 的更多信息，请参阅 [GitHub 文档](https://docs.github.com/en/get-started/quickstart/set-up-git)。

## 基本的 Git 命令

### git clone
将仓库克隆到新创建的目录中。这对于将远程仓库复制到本地计算机非常有用。

```bash
git clone <repository_URL>
```

### git pull
从另一个仓库或本地分支获取并集成。这用于使用远程仓库（例如 GitHub）中的最新更改更新本地仓库。

```bash
git pull
```

### git branch
列出、创建或删除分支。分支对于在彼此隔离的环境中开发功能非常有用。

列出所有分支：
```bash
git branch
```

创建新分支：
```bash
git branch <branch_name>
```

删除分支：
```bash
git branch -d <branch_name>
```

### git checkout
切换分支或恢复工作树文件。这对于在不同分支和提交之间移动非常有用。

切换到现有分支：
```bash
git checkout <branch_name>
```

创建并切换到新分支：
```bash
git checkout -b <branch_name>
```

### git status
显示工作树状态。它对于查看已暂存的更改、未暂存的更改以及未被 Git 跟踪的文件非常有用。

```bash
git status
```

### git add --all
将工作目录中的所有更改暂存到下一次提交中。这包括新文件、修改文件和删除文件。

```bash
git add --all
```

### git reset
取消暂存已添加到暂存区的文件，但不改变工作目录。

```bash
git reset
```

### git commit -m "[一些信息在这里]"
记录对仓库所做的更改，并附带一条消息。消息应描述所做的更改。

```bash
git commit -m "[一些信息在这里]"
```

### git push
将本地仓库内容上传到远程仓库（例如 GitHub）。这需要在 GitHub 上设置一个远程仓库并将你的本地仓库连接到它。如果你从 GitHub 克隆了一个仓库，它会自动连接。

```bash
git push
```

#### 推送到 GitHub 的步骤：
1. **在 GitHub 上创建一个仓库：**
   - 访问 [GitHub](https://github.com/) 并登录。
   - 点击右上角的 "+" 图标，选择“新建仓库”。
   - 输入仓库名称并点击“创建仓库”。

2. **将 GitHub 仓库添加为远程仓库：**
   - 复制 GitHub 仓库的 URL。
   - 在本地仓库中，添加远程仓库：

   ```bash
   git remote add origin <repository_URL>
   ```

3. **将你的更改推送到 GitHub：**

   ```bash
   git push -u origin main
   ```

通过这些步骤，你可以将本地仓库连接到 GitHub 并将更改推送到远程仓库。

## 将一个仓库合并到另一个仓库

要将一个仓库合并到另一个仓库并保留提交记录，请按照以下步骤操作：

1. 在目标仓库中添加源仓库作为远程仓库。
   ```bash
   git remote add source_repo <source_repo_URL>
   ```

2. 从源仓库获取分支及其提交记录。
   ```bash
   git fetch source_repo
   ```

3. 创建一个新分支以包含源仓库的历史记录。
   ```bash
   git checkout -b source_branch source_repo/main
   ```

4. 将源仓库分支合并到目标仓库的分支中。
   ```bash
   git checkout main
   git merge source_branch
   ```

5. 将更改推送到远程仓库。
   ```bash
   git push
   ```

通过这些步骤，你可以确保源仓库的提交历史记录保留在目标仓库中。
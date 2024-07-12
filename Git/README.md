# Git
<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/Collaborative-AI/tutorial/blob/main/Git/README_zh.md">简体中文</a>
    </p>
</h4>


## Table of Contents
1. [Introduction](#introduction)
2. [Installation of Git](#installation-of-git)
3. [Configuration of Username and Password](#configuration-of-username-and-password)
4. [Basic Git Commands](#basic-git-commands)
    - [git clone](#git-clone)
    - [git pull](#git-pull)
    - [git branch](#git-branch)
    - [git checkout](#git-checkout)
    - [git status](#git-status)
    - [git add --all](#git-add---all)
    - [git reset](#git-reset)
    - [git commit -m "[some message here]"](#git-commit--m-some-message-here)
    - [git push](#git-push)
5. [Merging One Repository into Another](#merging-one-repository-into-another)
6. [Best Practices](#best-practices)

## Introduction
Git is a distributed version control system used to track changes in source code during software development. GitHub is a web-based platform that uses Git and provides additional features such as collaboration, code review, and project management. This tutorial will guide you through the basics of using Git and GitHub.

## Installation of Git
### Windows
1. Download the Git installer from [git-scm.com](https://git-scm.com/downloads).
2. Run the installer and follow the on-screen instructions (default is fine).
3. To verify the installation, open Command Prompt and type:
   ```bash
   git --version
   ```

### MacOS
1. Install Homebrew if not already installed:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Git using Homebrew:
   ```bash
   brew install git
   ```
3. To verify the installation, open Terminal and type:
   ```bash
   git --version
   ```

### Linux
1. Open Terminal.
2. Install Git using the package manager:
   - **Debian/Ubuntu:**
     ```bash
     sudo apt-get update
     sudo apt-get install git
     ```
   - **Fedora:**
     ```bash
     sudo dnf install git
     ```
3. To verify the installation, type:
   ```bash
   git --version
   ```

## Configuration of Username and Password
After installing Git, you need to set up your username and email address. This is important because every Git commit uses this information. Additionally, when you push commits to GitHub, these details are used to attribute the commits to your GitHub account.

1. Set your username:
   ```bash
   git config --global user.name "Your Name"
   ```

2. Set your email address:
   ```bash
   git config --global user.email "your.email@example.com"
   ```

3. Configure Git to store your credentials:
   ```bash
   git config --global credential.helper store
   ```

4. The next time you perform a Git operation that requires authentication (e.g., `git push`), Git will prompt you for your username and password and store them in a plain text file in your home directory (`~/.git-credentials`).

For more information on connecting Git with GitHub, refer to the [GitHub documentation](https://docs.github.com/en/get-started/quickstart/set-up-git).

## Basic Git Commands

### git clone
Clones a repository into a newly created directory. This is useful for copying a remote repository to your local machine.

```bash
git clone <repository_URL>
```

### git pull
Fetches from and integrates with another repository or a local branch. This is used to update your local repository with the latest changes from the remote repository (e.g. GitHub).

```bash
git pull
```

### git branch
Lists, creates, or deletes branches. Branches are useful for developing features in isolation from each other.

To list all branches:
```bash
git branch
```

To create a new branch:
```bash
git branch <branch_name>
```

To delete a branch:
```bash
git branch -d <branch_name>
```

### git checkout
Switches branches or restores working tree files. This is useful for moving between different branches and commits.

To switch to an existing branch:
```bash
git checkout <branch_name>
```

To create and switch to a new branch:
```bash
git checkout -b <branch_name>
```

### git status
Shows the working tree status. It’s useful for seeing what changes have been staged, which haven’t, and which files aren’t being tracked by Git.

```bash
git status
```

### git add --all
Stages all changes in the working directory for the next commit. This includes new, modified, and deleted files.

```bash
git add --all
```

### git reset
Unstages files that have been added to the staging area, but leaves the working directory unchanged.

```bash
git reset
```

### git commit -m "[some message here]"
Records the changes made to the repository with a message. The message should describe what changes were made.

```bash
git commit -m "[some message here]"
```

### git push
Uploads local repository content to a remote repository (e.g. GitHub). This requires setting up a remote repository on GitHub and connecting your local repository to it. If you clone a repo from GitHub, it is automatically connected.

```bash
git push
```

#### Steps to Push to GitHub:
1. **Create a repository on GitHub:**
   - Go to [GitHub](https://github.com/) and log in.
   - Click on the "+" icon in the top-right corner and select "New repository".
   - Enter a repository name and click "Create repository".

2. **Add the GitHub repository as a remote:**
   - Copy the URL of the GitHub repository.
   - In your local repository, add the remote repository:

   ```bash
   git remote add origin <repository_URL>
   ```

3. **Push your changes to GitHub:**

   ```bash
   git push -u origin main
   ```

By following these steps, you can connect your local repository to GitHub and push your changes to the remote repository.

## Merging One Repository into Another

To merge one repository into another and preserve commits, follow these steps:

1. Add the source repository as a remote in the destination repository.
   ```bash
   git remote add source_repo <source_repo_URL>
   ```

2. Fetch the branches and their respective commits from the source repository.
   ```bash
   git fetch source_repo
   ```

3. Create a new branch to contain the history from the source repository.
   ```bash
   git checkout -b source_branch source_repo/main
   ```

4. Merge the source repository branch into the destination repository's branch.
   ```bash
   git checkout main
   git merge source_branch
   ```

5. Push the changes to the remote repository.
   ```bash
   git push
   ```

By following these steps, you ensure that the commit history from the source repository is preserved in the destination repository.

## Best Practices

### Use a .gitignore Template
To avoid pushing unnecessary files (such as OS-specific files, IDE config files, and build artifacts) to your repository, always use a `.gitignore` file. GitHub provides a collection of useful `.gitignore` templates for different programming languages and development environments. You can refer to [this example .gitignore file for Python](https://github.com/diaoenmao/RPipe/blob/main/.gitignore).

### Avoid Pushing Datasets and Output Files
Avoid pushing large datasets and output files to your GitHub repository. Use a `.gitignore` file to avoid pushing these files to your repository, ensuring that your repository remains clean and efficient.
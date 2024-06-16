# Git

## Table of Contents
1. [Introduction](#introduction)
2. [Installation of Git](#installation-of-git)
3. [Configuration of Username and Password](#configuration-of-username-and-password)
4. [Basic Git Commands](#basic-git-commands)
    - [git status](#git-status)
    - [git add --all](#git-add---all)
    - [git reset](#git-reset)
    - [git commit -m "[some message here]"](#git-commit--m-some-message-here)
    - [git push](#git-push)
5. [Merging One Repository into Another](#merging-one-repository-into-another)

## Introduction
Git is a distributed version control system used to track changes in source code during software development. GitHub is a web-based platform that uses Git and provides additional features such as collaboration, code review, and project management. This tutorial will guide you through the basics of using Git and GitHub.

## Installation of Git
### Windows
1. Download the Git installer from [git-scm.com](https://git-scm.com/downloads).
2. Run the installer and follow the on-screen instructions.
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
After installing Git, you need to set up your username and email address. This is important because every Git commit uses this information.

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Basic Git Commands

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
Uploads local repository content to a remote repository.

```bash
git push
```

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


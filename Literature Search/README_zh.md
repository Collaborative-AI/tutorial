## 文献调研
<h4 align="center">
    <p>
        <a href="https://github.com/Collaborative-AI/tutorial/blob/main/Literature%20Search/README.md">English</a> |
        <b>中文</b>
    </p>
</h4>

### 目录

1. [简介](#简介)
2. [使用 Google Scholar](#使用-google-scholar)
   1. [关键词搜索](#关键词搜索)
   2. [筛选论文](#筛选论文)
3. [了解出版场所](#了解出版场所)
   1. [场所类型](#场所类型)
4. [论文类型](#论文类型)
   1. [综述论文](#综述论文)
   2. [创新论文](#创新论文)
5. [下载论文](#下载论文)
6. [论文整理](#论文整理)
7. [复现结果](#复现结果)
8. [其他资源](#其他资源)

---

## 简介

进行彻底的文献搜索是学术和专业研究的基本步骤。它有助于了解现有的知识体系，识别空白，并在前人的工作基础上进行研究。本文教程将指导你如何找到相关论文，有效整理它们，并复现关键结果，以加深你对该领域的理解。

---

## 使用 Google Scholar

### 关键词搜索
- 从与你感兴趣的主题相关的广泛关键词开始。
- 如有必要，使用附加关键词来缩小搜索范围。
- 使用 Ctrl + F 在文章中搜索引用的论文。
- 使用“被引用次数”和“在引用文章中搜索”来查找后续论文。

你可以在[这里](https://scholar.google.com)访问 Google Scholar。

### 筛选论文
- **引用次数:** 一般来说，引用次数越多的论文越好。
- **出版场所:** 优先选择在知名场所发表的论文。你可以使用[Google Scholar metrics](https://scholar.google.com/citations?view_op=top_venues&hl=en)来确定这些场所的排名。
- **出版时间:** 最近的论文通常更具相关性。例如，深度学习研究在2013年后变得更加突出，使用 ResNet 的更先进模型在2016年后出现。最近的趋势包括2020年后的大模型和扩散模型。

---

## 了解出版场所

### 场所类型

#### 会议
- 有特定的截止日期。
- 论文由其他作者进行同行评审。
- 通知接受通常在大约3个月后。
- 被接受的论文在会议上展示，提供了网络交流的机会。

#### 期刊
- 没有特定的截止日期，但通常有特殊主题。期刊可能会定期发布关于特定新兴趋势的论文征集。
- 论文经过编辑初审，然后是同行评审。
- 审核过程可能需要6个月到一年以上，可能涉及多轮修订。
- 在 AI 领域，顶级会议优于顶级期刊，而在其他领域，情况通常相反。

---

## 论文类型

### 综述论文
- 总结和组织以往的研究。
- 这些是理解一个领域的理想起点。
- **评估标准:** 日期 > 引用 > 出版场所。

### 创新论文
- 提出新的发现，可以分为理论、方法和应用论文。有些论文可能涉及两到三个因素。
- 其中一些论文往往是一个领域的基石，引用次数高，发表在知名场所。
- **评估标准:** 引用 > 出版场所 > 日期。
- 对于最先进的进展，出版日期至关重要。

---

## 下载论文

- 如果原始来源的访问受限，请搜索论文标题并寻找替代来源，如 [arXiv](https://arxiv.org/)。

---


## 论文整理

- **创建电子表格:** 这是一个系统化地组织和管理文献收藏的工具。为了提高可读性并突出关键概念，建议在电子表格中**加粗重要关键词**。这些关键词可以包括关键概念、方法、数据集、评估指标或任何创新贡献。加粗关键词有助于一目了然地突出关键信息，便于在重读文献时快速查找具体细节。
  
- **建议列（按日期排序）：**

  - **标题:** 论文的标题，通常反映了核心研究问题或贡献。
  
  - **机构:** 建议列出机构名称而不是所有作者的名字，以便于快速参考。这有助于识别研究单位，并可以揭示特定研究团队的趋势。
  
  - **出版地点:** 论文发表的会议或期刊。这可以指示论文的可信度和影响力，因为有些会议或期刊更具权威性。
  
  - **日期:** 论文的发表日期。按日期排序有助于追踪研究主题的时间发展脉络。
  
  - **目标:** 对研究主要目标的概述，详细说明论文解决的问题。
  
  - **方法:** 对论文中使用的方法的简要描述，包括任何算法、模型或理论方法。通常可以从摘要中提取此信息。

  - **创新点:** 总结论文相较于之前工作的独特或创新之处。这可能是新方法、新应用或现有方法的重大改进。
  
  - **数据集:** 论文中使用的数据集，对于建立标准基准以与其他方法进行比较非常重要。

  - **评估指标:** 用来评估所提出方法性能的评估指标，如准确率和F1分数。

  - **结果:** 论文的主要贡献，通常突出所提出方法根据所选指标的表现。可以从摘要、引言和结论部分提取这些贡献。
  
  - **链接:** 访问论文的在线URL，无论是通过出版商网站、arXiv或其他资源库。
  
  - **代码:** 链接到与论文相关的任何代码库（如GitHub）。访问代码对于复现结果或在此基础上进行进一步研究非常有帮助。

---

## 复现结果

- 选择一篇基础论文来复现其结果，最好是有可用实现的。
- 尝试找到包含旧基准论文的基准代码库。
- 使用此过程来加深你对该领域的理解，并集思广益，产生新想法。

---

## 其他资源

- **会议截止日期:** 使用如 [AI Deadlines](https://aideadlin.es/) 的网站来跟踪重要日期。
- **GitHub 仓库:** 寻找“Awesome [主题名称]”仓库以获取精选的重要论文列表。还有一些特定领域的基准仓库。
- **Papers with Code:** 一个收集各种主题的排行榜、数据、代码和论文的网站。可以在[这里](https://paperswithcode.com/)访问。
- **论文阅读教程:** 有关如何有效阅读和分析学术论文的指导，请参考这个[论文阅读教程](https://github.com/Collaborative-AI/tutorial/blob/main/Paper%20Reading/README_zh.md)。


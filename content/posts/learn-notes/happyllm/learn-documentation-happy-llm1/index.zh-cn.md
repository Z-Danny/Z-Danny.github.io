---
weight: 1
title: "Happy_LLM_01 NLP前言"
date: '2025-08-26T17:17:53+08:00'
lastmod: '2025-08-26T17:17:53+08:00'
draft: false
#authorLink: "https://dillonzq.com"
description: "bilibili shortcode 提供了一个内嵌的用来播放 bilibili 视频的响应式播放器."
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"

tags: ["LLM"]
categories: ["learn-notes"]

hiddenFromHomePage: false

toc:
  enable: true
---


<!--more-->

# Task01：项目介绍 + 前言
（这是笔者自己的学习记录，仅供参考，原始学习链接见最下面，愿 LLM 越来越好❤
## 1. NLP 主要研究什么？
NLP（Natural Language Processing，自然语言处理）  
主要聚焦：计算机如何 **理解、处理、生成** 人类的语言。

---

## 2. LLM vs PLM：两种模型分别是什么？

| 简称 | 全称 | 中文 | 时代定位 |
| --- | --- | --- | --- |
| **LLM** | Large Language Model | 大语言模型 | 当下最火的模型，NLP的衍生成果 |
| **PLM** | Pretrain Language Model | 预训练语言模型 | NLP 过去的主流模型 |

---

## 3. LLM 在 PLM 基础上有什么改进？

| 维度 | PLM（如 BERT、GPT-1/2） | LLM（如 GPT-3/4、Qwen、ChatGLM 等） |
| --- | --- | --- |
| **训练数据规模** | 相对较小 | **海量数据** |
| **参数量** | 百万~十亿级 | **十亿~千亿级** |
| **微调方式** | 需要一定量的监督数据 | **指令微调 + RLHF（人类反馈强化学习）** |
| **能力特征** | 单一任务表现好 | **涌现能力（Emergent Ability）**<br>- 上下文学习（In-context Learning）<br>- 指令理解（Instruction Following）<br>- 高质量文本生成 |

一句话总结：  
> **模型更大（参数量大了） + 数据更多（预训练数据规模） + 训练策略更先进 ⇒ LLM 能力“chua”一下爆发！**

---

## 4. Datawhale 相关开源项目一览

| 项目名称 | 定位 | 在线地址 |
| --- | ---  | --- |
| **Self-LLM**<br>（开源大模型食用指南） |为开发者提供一站式开源 LLM 部署、推理、微调的使用教程  | https://github.com/datawhalechina/self-llm |
| **LLM-Universe**<br>（动手学大模型应用开发） |指导开发者从零开始搭建自己的 LLM 应用|https://github.com/datawhalechina/llm-universe|
| **Happy-LLM**<br>（从零开始的大语言模型原理与实践） | **深入 LLM 原理** + **动手复现 LLaMA2** | https://github.com/datawhalechina/happy-llm |

---

> 笔者一点点感受：
> LLM真的很奇妙，它让计算机用计算的方式能够生成人类语言，明明只是0101，却通过各种参数使得人类的语言符号被学习、理解、生成。虽然机器不像人类那样有脑子🧠，但是也感觉到很奇妙，似乎发现了更广阔神秘的天地。
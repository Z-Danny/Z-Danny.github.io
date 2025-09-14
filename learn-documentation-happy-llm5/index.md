# Happy_LLM_05 BERT encoder-only



<!--more-->



# Task05：第三章 预训练语言模型PLM
3.1 Encoder Only之BERT
（这是笔者自己的学习记录，仅供参考，原始[学习链接](https://datawhalechina.github.io/happy-llm/#/./chapter1/%E7%AC%AC%E4%B8%80%E7%AB%A0%20NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5?id=_13-nlp-%E4%BB%BB%E5%8A%A1)，愿 LLM 越来越好❤）

---

## 主流PLM的概况
PLM = pretrained language model，是在编解码的基础上引入了ELMo的预训练思路。

> **预训练+微调** 处理 NLP任务

主流代表如下：
| 模型   | 推出公司 | 架构类型        | 主要任务方向         | 预训练方式/特点                |
|--------|------------|-----------------|----------------------|--------------------------------|
| BERT   | Google 2018     | Encoder-only    | NLU（自然语言理解）   | 提出MLM（掩码mask），堆叠Encoder层 |
| GPT 系列 | OpenAI    | Decoder-only    | NLG（自然语言生成）   | 基于原有LM（语言模型），扩大参数量与训练语料   |
| T5     | Google     | Encoder-Decoder | 综合任务   | 保留完整Transformer架构，统一文本到文本 |


---
下面会介绍model的模型架构、选择的预训练任务、优势。
## BERT
- 来自Transformer的双向编码表示法
 = **B**idirectional **E**ncoder **R**epresentation from **T**ransformer
 - 是PLM-预训练语言**模型**，seq2seq模型，LLM出来之前在多个NLP任务中up to SOTA（state of the art这个领域的no.1）
 - 有很多变种，这里主要介绍 **BERT（Google）、RoBERTa（Facebook）、ALBERT（Google）**
 - 主要做NLU任务，理解强，可做评论分类
### BERT怎么问世的？
主要有2个原因：
- Transforme架构：2017年架构提出，牛逼。然后谷歌2018年就采用架构里的encoder优化，适配NLU任务
- 预训练+微调的范式：ELMo提出了很好的这种思路，有一个通用的pre模型，再结合具体任务微调，肯定更适配具体任务。

### 预训练+微调范式的优势？
1. 传统是每个任务要重新用新的标注（全监督）数据训练出模型
2. 而这种范式是`先花大钱` 训练【长时间、大规模数据和参数】一个通用的模型【强大的语言理解和生成能力】，再根据不同的下游任务`再花小钱` 用少量数据进行微调。（成本低、应用价值广）
3. 因为训练大规模的，所以数据肯定肯定用的（无监督）数据，所以一般训练都是LM任务，语言建模任务，用的文本语料（数亿规模token）。
### 常见的LM-语言建模任务
LM = language modeling，这是任务 不是语言模型哈。

#### *这类任务的target？*：

> 让模型学 语言的结构、语义、上下文关系。能预测词语或理解句子【理解会说】

#### *有哪些具体任务？*：
> 下面这些都是比较新的LM任务
> - MLM （masked遮蔽的）：随机遮蔽句子中的某些词，让模型预测被遮蔽的词——BERT
> - CLM （causal因果的）：根据前面的词预测下一个词——GPT
> - SOP（sentence order prediction）：判断两句话的顺序是否合理——ALBERT
> - NSP（next sentence prediction）：判断两句话是否是连续的——BERT

#### *传统LM pre-train的缺陷？*
只拟合从左到右单向的语义关系，忽略了双向的语义关系。
虽然Transformer中有用位置编码，但是和直接拟合双向还是有区别，如BiLSTM优于LSTM。


### BERT的pre-train任务
- BERT的架构和Transformer大差不差
- 创新点在于pre-train任务（语言建模任务）：
   - MLM = 完形填空
   - NSP
#### MLM
完形填空任务，token级别
task：在一个文本序列中随机遮蔽部分 token，然后将所有未被遮蔽的 token 输入模型，要求模型根据输入预测被遮蔽的 token。

##### *为什么MLM考虑到了双向语义和可用无监督数据？*

> - 掩码遮住token，模型就得从token的上文和下文一起预测masked token，从而拟合双向语义，实现更好的文本理解。
> - 对文本语料随机遮蔽token就行

##### MLM pre-train的缺陷？
MLM虽然比LM关注了双向的语义，但是和下游任务**不一致**，会影响微调的性能。
下游任务（微调和推理）：根据上文预测下文（LM的训练也是这样），不会有`<mask>`出现（MLM会出现），直接通过原文本得到对应的隐藏状态再根据下游任务进入分类器或其他组件。

##### 改进MLM pre-train的策略
只遮蔽训练语料中15%的token，这些里面又只有80%概率被真的mask
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0be8c70b0917470ea337f12385497e31.png)

#### NSP
句子级别
task：针对句级的 NLU 任务，句子级别拟合关系，判断两个句子之间的关系

##### *为什么NSP可用无监督数据？*
正样本直接随机抽句子文本，负样本随机打乱之后就可用了。

#### *BERT的预训练语料与训练设置？*

| 内容       | 详情                                                                 |
|------------|----------------------------------------------------------------------|
| 数据规模   | 3.3B Token（约 33 亿），约 13GB                                      |
| 文本来源   |  BooksCorpus：0.8B Token（约 8 亿，主要是小说）<br>  英文 Wikipedia：2.5B Token（约 25 亿） |
| 上下文长度 | 90% 使用 128，10% 使用 512                                           |
| Batch Size | 256 个样本                                                          |
| Step       | 1M （1百万）                                                                  |
| Epoch      | 40个    

NSP预训练时，（`<CLS>特殊token` 文本序列）代表句级的语义表征
MLM预训练时，用`<CLS>特殊token`对应的特征向量作为分类器的输入

#### *BERT的版本？*
| 版本   | Encoder 层数 | 隐藏层维度 | 参数量 | 训练资源 | 训练时长 |
|--------|--------------|------------|--------|----------|----------|
| Base   | 12           | 768        | 110M   | 16 块 TPU | 约 4 天   |
| Large  | 24           | 1024       | 340M   | 64 块 TPU | 约 4 天   |

### 下游任务微调
#### 怎么微调？
针对特定任务，用更少更精的数据，更小的batch size进行再训练，小幅调整更新参数。
#### 一些下游任务
- 文本分类：直接改模型结构中的prediction_heads最后的分类头
- 序列标注：集成 BERT 多层的隐含层向量再输出最后的标注结果
- 文本生成：取 Encoder 的输出直接解码得到最终生成结果
### BERT架构
就是transformer的encoder堆叠。
tokenizer、Embedding、Encoder 、prediction_heads 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/908f5983715e4452b4dd6e7951a6b8f1.png)
#### tokenizer-WordPiece
| 特点       | 说明                                                                 |
|------------|----------------------------------------------------------------------|
| 分词方法   | 使用 **WordPiece** 作为分词器                                         |
| 核心思想   | 基于统计的**子词切分**算法，将单词拆分为子词                             |
| 合并策略   | 按最大化语言模型的似然度来决定子词合并                               |
| 英文处理   | 将单词切分为子词单位，例如 *"playing"* → ["play", "##ing"]            |
| 中文处理   | 由于没有空格分词，通常将**单个汉字**作为最小分词单位（token）    例如 *"玩耍"* → ["玩", "耍"]       |

#### Intermediate 层 = FNN 
是 BERT 的特殊称呼,就是一个线性层加上激活函数

#### BERT的激活函数
GELU 函数，全名为高斯误差线性单元激活函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6eac527167fa4e469fbc1aad0d7e7199.png)

> GELU 的核心思路为将**随机正则**的思想引入激活函数，通过输入`自身的概率分布，来决定抛弃还是保留自身的神经元`。

#### BERT的位置编码和注意力计算
BERT 将相对位置编码融合在了注意力机制中，将相对位置编码同样视为可训练的权重参数

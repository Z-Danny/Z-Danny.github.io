# Happy_LLM_11 如何用transformers框架进行开源LLM的pretrain+SFT（deepspeed分布式）、高效微调（adapt、prefix、lora微调）的流程



<!--more-->

# Task11：第六章 大模型训练流程实践
为啥会有这一篇，希望大家明白前因后果，帮助更好理解现在流行的框架。
接task10，每训一个大模型每个组件都自己写代码，显然不合适。
原因如下：
- 麻烦
- 各个模型之间不一定通用、参数可能没法移植
- 自己写的分布式训练也不好搞

所以，机智的开源社区给出了LLM相关的
- 主流训练框架Transformers——进行模型的pretrain、sft
- 分布式框架deepseed
- 高效微调框架peft

---

## what Transformers框架？
### 框架介绍
Transformers框架 ≠ Transformer结构
它是hugging face开发的，
- 模块化组件，就可以支持拼拼凑凑出主流模型架构，如BERT、GPT等，`AutoModel类`
- 内置`Trainer类`，封装分布式训练（pytorch原生的DDP、DeepSeed、Megatron-LM等训练策略）的核心逻辑 + 简单`配置训练参数` ——数据/模型/流水线并行
ps：8卡A100集群就可以10+B参数训练
- 可实现训练过程`自动化管理`，（os：不然训那么久还要一直看着，太可怕了）配合SavingPolicy 和 LoggingCallback 等组件

### 咱今天来逛三园（拥抱脸）
逛的什么园，hugging face园（os：给大家来段贯口）
三园里面有什么？
- 数亿 pretrain 的`模型参数`
- 25w+ 不同类型 `数据集`
- 搭好的（pretrain模型+data+eval函数）框架
（os：这意味着什么你知道吗？那就是你要换哪个开源LLM、用什么data都可以，“随便放随便训”，也没那么随便其实hh）

### 目前比较多的LLM工作
要知道
- 训一个pretrain不容易（费时费钱费资源），所以`post-train`和`SFT`就很多，基模然后调一调满足下游业务才是真正的**落地**
- 就算是大款，LLM的 **para 也很大**，所以像`deepspeed等分布式训练框架`老必备技能了

---
## how Transformers 自定义 LLM（训参数）？
即，通过 Transformers 框架实现 LLM 的 Pretrain 及 SFT

> 这里讲的是从只有某个模型框架（参数是随机数）到模型参数pretrain填好，再到SFT监督微调的阶段
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e418a39829d9418b8e1cef08f1a8287d.png)

### 1.初始化一个 LLM + tokenizer
有两种，以**CausalLM**——**Qwen-2.5-1.5B** 为例（ps：因果LLM，就是用CLM任务训的LLM，前面预测下一个词）
用到的类：*AutoModelForCausalLM、AutoConfig*

- 直接加载一个`带para的LLM`（后面直接在自己的预料上训）【最多】**（ps：如果选择这种可以直接跳到下一节高效微调的步骤了）**
- 加载一个`盲盒零LLM`（权重para是随机的，然后用CLM任务去训出para）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9c86e1a371754482afb8e6c655b2d5e7.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f39e58bb27024499846b20aa39e8c0de.png)
tokenizer也是，一般就加载开源LLM自带的分词器就行
用到的类：*AutoTokenizer*
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7d89bf2ed66442cf83eaeccdf42914c2.png)
### 2.预训练数据处理
加载完分词器和架构后，自然就是处理下训练数据了。
这里用的是 *出门问问序列猴子开源数据集30+GB* （os：xdm为伟大的开源事业喝彩！）
【加载（下载解压）-- 分词 -- 拼接切分】

> ① 加载并看看 jsonl数据

用到的库：hf的*datasets*
用到的类：load_dataset
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a3f1bc71386340f888257b791afa2320.png)


- json格式 vs jsonl格式？
  - json：整个文件是一个json，用“，”分割每个
  - jsonl：**大型数据**，json line ，一行是一个json，用“\n”分割json对象

- ds是 `DatasetDict`对象，就是**超大数据集**，里面很多**子集**
```python
# 查看数据
ds["train"][0]

# 查看特征
column_names = list(ds["train"].features)
```


> ② 用tokenizer 对数据分词

`ds.map()`：对data中每个样本执行函数，返回新数据集。
ps：为什么不用for循环？—— 根本不是一个档次，map可以**①并行处理、②自动缓存、③进度条**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/527eb356b2624352bf8a0e23605dff6a.png)

```python
# 对数据集进行 tokenize
def tokenize_function(examples):
    # 使用预先加载的 tokenizer 进行分词
    output = tokenizer([item for item in examples["text"]])
    return output

# 批量处理
tokenized_datasets = ds.map(
    tokenize_function,
    batched=True,
    num_proc=10,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)
```

> ③ 拼接分块处理数据

操作：把多个文本段**拼接**在一起，处理成**统一长度**的文本块，再对每个文本块进行训练
【 切成每个2048 token 长度，再 map 批量处理】
原因： pretrain的CLM 任务，一次性学习多个样本的序列语义不显著提高模型性能❌，且训练数据量大❌、训练时间长❌，对训练效率要求比较高❌
用到的库和包：from itertools import `chain`

```python
# 预训练一般将文本拼接成固定长度的文本段
from itertools import chain

# 这里我们取块长为 2048
block_size = 2048

def group_texts(examples):
    # 将文本段拼接起来
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # 计算拼起来的整体长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 如果长度太长，进行分块
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # 按 block_size 进行切分
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # CLM 任务，labels 和 input 是相同的
    result["labels"] = result["input_ids"].copy()
    return result

# 批量处理
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=10,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
    batch_size = 40000,
)
train_dataset = lm_datasets["train"]

```

### 3.配置训练参数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/80b299fd23594bf19a9ba72ca3b90ed8.png)

### 4.用DeepSpeed分布式开训！
你直接用 **.jpynb** 运行，训不动，会断。
一般预训练过程：`多卡分布式 + bash 脚本（设定超参) + python脚本（训练控制）`

- logging 库来实现（日志）记录
- 固定间隔保存 checkpoint（防崩）
- （监测）训练进度、loss 下降趋势 ——swanlab 作为训练检测

具体代码就不展开了，平时也不咋会遇到，知道大概过程就行。[想看戳这里](https://datawhalechina.github.io/happy-llm/#/./chapter6/%E7%AC%AC%E5%85%AD%E7%AB%A0%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5?id=_61-%e6%a8%a1%e5%9e%8b%e9%a2%84%e8%ae%ad%e7%bb%83)
### 5.SFT开调（指令微调）
CLM任务 训练时的pre和sft 计算区别
- pretrain 会对全部 text 进行 loss 计算，要求模型对整个文本实现建模预测；
- 而 SFT 仅对`输出进行 loss 计算`，不计算指令部分的 loss

数据：准备微调数据集，这个数据量也不少哦。构造指令回答`数据对`

（具体也不说了）[想看戳这里](https://datawhalechina.github.io/happy-llm/#/./chapter6/%E7%AC%AC%E5%85%AD%E7%AB%A0%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5?id=_62-%e6%a8%a1%e5%9e%8b%e6%9c%89%e7%9b%91%e7%9d%a3%e5%be%ae%e8%b0%83)

----
## how 高效微调？
### Adapt Tuning
在模型架构中每层加入`adapter层`，**微调时**模型原参数`冻结`。
【LoRA就是其改进版】
存在的问题：
- 其实是增加了挺多参数量，也增加了计算量，推理会比原模型**更慢**

### Prefix Tuning 前缀微调
是在微调阶段，不同任务加前缀**prefix**=virtual token 加在输入token前面，微调训练时就只更新**这部分prefix参数**，原模型参数冻结。

【P-tuning就是其改进版】
存在的问题：
- 模型可用序列**长度缩短**了，因为被prefix占用了。
- **微调质量**越好，模型推理可用**context越少**

---
## 高效微调——LoRA微调（peft库）
### what peft库？
- peft也是拥抱脸小哥的，封装了LoRA、Adapt Tuning、P-tuning（就是主流的微调）等方法

- peft 库目前支持调用 LoRA 的层包括：**nn.Linear、nn.Embedding、nn.Conv2d** 三种。
### LoRA整体流程

> 核心： 选层换成lora层，复制冻结原参，更新旁路的两个A、B低秩矩阵数值

优点：
- 实际训练大幅降低显存占用，下游也能适配

缺点：
- 只调整低秩矩阵，其实模型很多的参数没变，所以整体的`知识注入没什么增加`。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/154d52741a6b46eeacc74d27fca7b0ad.png)
用peft进行LoRA微调：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e5fe68944ec4742ba9ad176e4367cc1.png)

---
[happy-llm参考资料](https://datawhalechina.github.io/happy-llm/#/./chapter6/%E7%AC%AC%E5%85%AD%E7%AB%A0%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5?id=_63-%e9%ab%98%e6%95%88%e5%be%ae%e8%b0%83)

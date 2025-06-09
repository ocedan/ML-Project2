# 推文情感分析：高级预处理、词嵌入与Transformer-based分类器的比较研究

**Machine Learning Project 2**

本项目是西湖大学机器学习课程的课程作业，旨在对大规模推文语料进行情感分析。我们从基础方法出发，逐步探索并比较了不同的文本预处理方法、词嵌入技术以及先进的分类模型架构，最终目标是找到实现高准确率的最佳组合。

## 1\. 项目简介 (Introduction)

本项目聚焦于预测推文的情感极性（积极或消极）。面对Twitter数据固有的非正式性、高噪音和特殊语言模式（如emoji、hashtag、缩写），我们设计并实现了一套包含高级预处理、多种词嵌入模型（自训练GloVe、预训练FastText、预训练GloVe Twitter）和不同分类器（逻辑回归、BERT）的完整机器学习管道。本研究量化了各项优化措施对模型性能的提升效果，并探讨了在实际开发过程中遇到的关键技术挑战及其解决方案。

## 2\. 项目特点与亮点 (Features & Highlights)

  * **全面的文本预处理：** 从基础的空格分词到基于`NLTK TweetTokenizer`的智能分词，以及小写化、低频词过滤和优化共现统计（滑动窗口、字典累积）。

  * **多维度词嵌入比较：**

      * **自训练GloVe：** 基于优化后的共现矩阵，在本地数据集上训练词向量。

      * **预训练FastText：** 利用其子词信息处理OOV词的强大能力。

      * **预训练GloVe Twitter：** 采用在Twitter语料上训练的领域特定词嵌入。

  * **模型架构升级：** 从传统的**逻辑回归**基线模型，升级到当前最先进的\*\*Transformer-based模型（BERT）\*\*进行文本分类微调。

  * **工程挑战与解决方案：** 记录并解决了处理大规模数据时的内存溢出、库下载受限及安装编译困难等实际工程问题。

  * **可复现性：** 项目代码结构清晰，提供详细的运行指南，旨在确保实验的可复现性。

## 3\. 数据集 (Dataset)

本项目使用Twitter数据集，包含：

  * **训练集：** 2.5百万条推文（1.25M积极，1.25M消极），已通过表情符号（:) / :(）标注。

  * **测试集：** 10,000条未标注推文，需要预测情感。

数据集文件位于 `twitter-datasets/` 目录下：

  * `twitter-datasets/train_pos_full.txt`

  * `twitter-datasets/train_neg_full.txt`

  * `twitter-datasets/test_data.txt`

## 4\. 环境搭建 (Getting Started)

### 4.1. 前提条件 (Prerequisites)

  * Python 3.8+ (推荐使用 [Anaconda](https://www.anaconda.com/products/distribution) 管理环境)

  * 稳定的互联网连接（部分模型下载可能需要较长时间）

  * 充足的磁盘空间（FastText模型可能占用数GB）

  * 建议使用 **GPU** (NVIDIA GPU with CUDA) 进行Transformer模型的训练，否则训练时间会非常长。

### 4.2. 库安装 (Installation)

在您的Python环境中，通过`pip`安装所有必要的库：

```bash
pip install numpy scipy scikit-learn transformers datasets evaluate nltk fasttext requests
```

### 4.3. NLTK数据下载 (NLTK Data Download)

`NLTK` 库的`TweetTokenizer`需要`punkt`资源。在首次运行使用NLTK的代码时，可能会自动提示下载，您也可以提前手动下载：

打开Python解释器或运行以下代码：


```
import nltk
nltk.download('punkt')
```

### 4.4. 预训练模型下载 (Pre-trained Models Download)

由于部分预训练模型文件较大且官方链接可能受限，请按照以下步骤处理：

  * **FastText (cc.en.300.bin):**

      * **官方推荐：** `fasttext.util.download_model('en', if_exists='ignore')`。此方法将在您的`load_pretrained_embeddings.py`脚本运行时自动触发下载。如果下载速度缓慢或失败，请转至手动下载。

      * **手动下载（推荐，若自动下载受阻）：**

        1.  通过浏览器访问 [FastText英文词向量页面](https://fasttext.cc/docs/en/english-vectors.html)。

        2.  找到 "Common Crawl (600B words)" 下的 `cc.en.300.bin` (约 3.7GB)，点击下载。

        3.  使用专业的下载管理器 (如 [IDM](https://www.internetdownloadmanager.com/), [FDM](https://www.freedownloadmanager.org/)) 进行下载，以支持多线程和断点续传。

        4.  下载完成后，将 `cc.en.300.bin` 文件放置到项目根目录下，与您的Python脚本同级。

  * **GloVe Twitter (glove.twitter.27B.100d.txt):**

      * **自动处理：** 运行 `load_pretrained_embeddings.py` 脚本时，它会尝试从 `http://nlp.stanford.edu/data/glove.twitter.27B.zip` 下载并解压该模型。

      * **手动下载：** 如果自动下载失败，您可以访问 [GloVe官方页面](https://nlp.stanford.edu/projects/glove/)，找到 "Twitter GloVe" 部分，下载 `glove.twitter.27B.zip` (100d 版本)。

      * 下载并解压后，确保 `glove.twitter.27B.100d.txt` 文件位于 `pretrained_embeddings/glove_twitter_extracted/` 目录下。

## 5\. 项目结构 (Project Structure)

```
.
├── twitter-datasets/
│   ├── train_pos_full.txt
│   ├── train_neg_full.txt
│   └── test_data.txt
├── preprocess_and_cooccurrence.py
├── load_pretrained_embeddings.py
├── text_classifier.py
├── bert_classifier.py
├── vocab.pkl                 # (生成文件) 自训练GloVe的词汇表
├── cooc.pkl                  # (生成文件) 自训练GloVe的共现矩阵
├── embeddings.npy            # (生成文件) 自训练GloVe的词向量
└── pretrained_embeddings/    # 预训练模型存放目录
    ├── glove_twitter_extracted/ # GloVe Twitter解压后的文件
    │   └── glove.twitter.27B.100d.txt
    ├── vocab_glove_twitter.pkl # (生成文件) 预训练GloVe Twitter的词汇表
    ├── embeddings_glove_twitter.npy # (生成文件) 预训练GloVe Twitter的词向量
    └── cc.en.300.bin         # FastText模型文件
```

## 6\. 如何运行 (How to Run)

本项目提供了多种模型和优化路径。请按以下步骤运行：

### 6.1. 预处理与自训练GloVe (Preprocessing & Self-trained GloVe)

如果您想使用自己训练的GloVe词嵌入进行测试，需要运行以下脚本：

```bash
# 1. 运行预处理脚本，生成词汇表(vocab.pkl)和共现矩阵(cooc.pkl)
#    该脚本包含了智能分词、小写化、低频词过滤和优化共现统计（滑动窗口、字典累积）
python preprocess_and_cooccurrence.py

# 2. 运行您的GloVe训练脚本（本项目未提供此脚本，假设您已有）
#    该脚本将读取cooc.pkl并生成embeddings.npy
#    例如：python your_glove_training_script.py
```

**注意：** `preprocess_and_cooccurrence.py` 中默认的 `MIN_WORD_COUNT` 为5，`window_size` 为10。

### 6.2. 加载和保存预训练词嵌入 (Loading Pre-trained Embeddings)

此脚本负责下载和处理预训练的FastText和GloVe Twitter模型，并将其转换为Python可加载的`.pkl`和`.npy`格式，供`text_classifier.py`使用。

在运行前，请编辑 `load_pretrained_embeddings.py` 文件顶部的配置：

  * `DOWNLOAD_FASTTEXT = True` （如果希望下载FastText模型）

  * `PROCESS_GLOVE_ZIP = True` （如果希望下载并解压GloVe Twitter的zip文件）

      * **注意：** 如果您已手动下载并解压了GloVe Twitter的`.txt`文件到正确目录，可以将 `PROCESS_GLOVE_ZIP` 设为 `False`。

<!-- end list -->

```bash
# 运行脚本进行下载、处理和保存
python load_pretrained_embeddings.py
```

### 6.3. 运行分类器 (Running the Classifier)

`text_classifier.py` 允许您在三种不同的词嵌入模型（自训练GloVe、预训练FastText、预训练GloVe Twitter）和逻辑回归分类器之间进行切换。

在运行前，请编辑 `text_classifier.py` 文件顶部的 `EMBEDDING_TYPE` 变量，选择您要测试的词嵌入模型：


```
# text_classifier.py (片段)
# --- Select which embedding model to use ---
# Options:
#   "self_trained_glove"
#   "fasttext"
#   "glove_twitter"
EMBEDDING_TYPE = "glove_twitter" # <-- 修改此行
```

然后运行：

```bash
python text_classifier.py
```

这将训练逻辑回归模型并生成 `submission.csv` 文件。

### 6.4. 运行基于BERT的分类器 (Running BERT Classifier)

此脚本用于使用BERT模型进行文本分类微调。

```bash
python bert_classifier.py
```

**注意：** BERT模型的训练需要 **GPU**。即使是较小的模型（如DistilBERT）也可能需要数十小时的微调时间。

## 7\. 结果与讨论 (Results & Discussion)

本项目通过实验量化了不同预处理和词嵌入方法对逻辑回归模型性能的影响，并初步评估了Transformer模型的潜力。

### 7.1. 逻辑回归分类器性能

| **Word Embedding Model** | **Validation Accuracy** | **F1-Score (Validation)** |
| :----------------------------------------- | :------------------ | :-------------------- |
| Baseline (Simple Preprocessing + Self-trained GloVe) | **0.548**  | **0.566**  |
| Pre-trained GloVe Twitter (100d)           | **0.764**  | **0.769**  |
| Pre-trained FastText (300d)                | N/A       | N/A        |

由于中国大陆官方源下载限制，FastText模型未能完全评估。

**关键发现：**

  * **预处理的显著影响：** 从基线（0.548准确率）到使用高级预处理（智能分词、小写化、低频词过滤、优化共现统计）并结合预训练GloVe Twitter模型（0.764准确率），性能实现了**大幅提升**。这表明，对Twitter数据进行细致的预处理对于模型性能至关重要。

### 7.2. Transformer模型（BERT）展望

| **Model** | **Training Accuracy** | **Validation Accuracy** | **F1-Score (Validation)** |
| :----------------------------------------- | :---------------- | :------------------ | :-------------------- |
| Logistic Regression (with Pre-trained GloVe Twitter) | N/A    | **0.764** | **0.769** |
| **BERT-base-uncased (Fine-tuned)** | N/A | **\>0.90** (预期)      | N/A |

逻辑回归在预训练GloVe Twitter下的具体训练准确率未直接从测试结果中获取。
由于微调训练时间过长（例如，DistilBERT在Kaggle上需要50小时），BERT模型的最终训练和验证结果尚未完全获取。

**预期和讨论：**

  * 尽管BERT模型的最终结果尚未完全获得，但基于其在类似NLP任务中的普遍表现，我们**预期准确率将接近或超过0.90**。

  * Transformer模型（如BERT）通过其上下文感知的词嵌入和强大的注意力机制，能够捕捉更深层次的语义和长距离依赖关系，这使其在处理复杂文本，特别是Twitter这种非正式语境中的细微情感时，具有超越传统线性模型的显著优势。

## 8\. 挑战与解决方案 (Challenges & Solutions)

在项目实施过程中，我们遇到了几个值得一提的挑战，并采取了相应的解决策略：

### 8.1. GloVe共现矩阵构建中的内存溢出 (MemoryError)

  * **困难：** 最初的共现统计方法是对推文内所有词对进行统计（N\*N复杂度），且将每个共现条目添加到大列表中，导致内存迅速耗尽。

  * **解决：**

    1.  **引入滑动窗口：** 将共现统计范围限制在词语周围的固定窗口内（例如`window_size=10`），大幅减少了生成的共现对数量。

    2.  **优化计数累积：** 改用Python字典直接累积唯一共现对的计数，避免了生成海量重复条目的中间列表，只在最后一步才将字典内容转换为列表。

### 8.2. FastText库下载与安装的持续挑战

  * **困难：**

    1.  **下载失败：** 官方链接在中国大陆访问受限（`403 Forbidden`）或下载速度极慢。

    2.  **安装编译失败：** `pip` install`  fasttext `报错`Could not build wheels`，提示C++编译环境不兼容。

    3.  **`isinstance`错误：** 即使不使用FastText，其底层库加载问题也可能导致`TypeError`。

  * **解决：**

    1.  **下载：** 最终采取手动通过浏览器配合专业的下载管理器进行下载。

    2.  **安装：** 针对编译问题，采用下载预编译的`.whl`文件（`fasttext`和`pybind11`）然后本地安装的方式，完全跳过C++编译。

    3.  **`isinstance`错误：** 在`text_classifier.py`中，将直接的`isinstance(obj, fasttext.FastText)`类型检查，替换为更具鲁棒性的鸭子类型检查（`hasattr(obj, 'get_word_vector')`）。

## 9\. 未来工作 (Future Work)

  * **完成BERT模型微调：** 在充足的计算资源（如更长时间的Colab Pro会话或本地GPU）下，完成BERT模型的完整微调，获取准确的性能指标。

  * **评估FastText性能：** 解决FastText模型的下载问题，并将其与逻辑回归模型结合，评估其在Twitter情感分类任务上的实际效果。

  * **超参数调优：** 对逻辑回归和BERT模型进行系统的超参数调优（如学习率、批次大小、正则化参数）。

  * **探索其他Transformer变体：** 尝试RoBERTa、ELECTRA或DistilBERT等其他Transformer模型，并比较其性能与计算效率。

  * **集成学习：** 结合多个模型（如逻辑回归和BERT）的预测结果，以进一步提高鲁棒性和准确性。

  * **错误分析：** 对模型预测错误的推文进行深入分析，找出错误模式，指导未来的改进方向。

## 10\. 贡献 (Contributing)

欢迎任何形式的贡献、建议和改进。如果您有任何问题或想法，请随时提交Issue或Pull Request。

## 11\. 许可证 (License)

本项目遵循MIT许可证。



```

```

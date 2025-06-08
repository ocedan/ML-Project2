import numpy as np
import os
import torch
from datasets import Dataset 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score 
import evaluate 

# --- 配置参数 ---
TRAIN_POS_FILE = "twitter-datasets/train_pos_full.txt"
TRAIN_NEG_FILE = "twitter-datasets/train_neg_full.txt"
TEST_FILE = "twitter-datasets/test_data.txt"

MODEL_NAME = "bert-base-uncased"

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

SUBMISSION_FILENAME = "submission_bert.csv"
MODEL_OUTPUT_DIR = "./results_bert"

def load_and_prepare_data(tokenizer_name):
    """
    加载原始文本数据，并使用Transformer的tokenizer进行预处理。
    """
    print("步骤 1: 加载原始文本数据...")
    texts = []
    labels = []

    try:
        with open(TRAIN_POS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
                labels.append(1) # 正面情感标签为1
    except FileNotFoundError:
        print(f"错误: 训练文件未找到: {TRAIN_POS_FILE}")
        return None, None

    try:
        with open(TRAIN_NEG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
                labels.append(0) # 负面情感标签为0
    except FileNotFoundError:
        print(f"错误: 训练文件未找到: {TRAIN_NEG_FILE}")
        return None, None
    
    print(f"训练数据加载完成。样本总数: {len(texts)}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}")

    print(f"步骤 2: 加载 {tokenizer_name} 的分词器...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    print("步骤 3: 构建 Hugging Face Dataset 对象并进行分词...")
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["text"])
    tokenized_train_dataset.set_format("torch")
    tokenized_val_dataset.set_format("torch")
    
    return tokenized_train_dataset, tokenized_val_dataset, tokenizer

# 使用 evaluate 库加载指标
# accuracy_metric = evaluate.load("accuracy") 
# f1_metric = evaluate.load("f1") # 默认是 micro F1，可以指定 average='macro' 或 'weighted'

def compute_metrics(eval_pred):
    """
    定义评估指标：准确率和F1分数。
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 通过 evaluate.load() 来获取指标
    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=labels)['accuracy']
    f1 = evaluate.load("f1").compute(predictions=predictions, references=labels)['f1']
    # 如果需要指定F1的平均方式，例如 macro F1:
    # f1_macro = evaluate.load("f1").compute(predictions=predictions, references=labels, average="macro")['f1']
    
    return {"accuracy": accuracy, "f1": f1}


def load_test_data_and_predict(tokenizer, model, filepath):
    """
    加载测试数据，进行分词，并用训练好的模型进行预测。
    """
    print(f"\n步骤 5: 加载测试数据 {filepath} 并进行预测...")
    test_texts = []
    test_ids = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line_stripped = line.strip()
                parts = line_stripped.split(',', 1) 
                if len(parts) == 2:
                    current_tweet_id = parts[0]
                    text_content = parts[1]
                else:
                    current_tweet_id = str(line_idx + 1)
                    text_content = line_stripped 
                
                test_ids.append(current_tweet_id)
                test_texts.append(text_content)
    except FileNotFoundError:
        print(f"错误: 测试文件未找到: {filepath}")
        return [], []
    except UnicodeDecodeError:
        print(f"错误: 读取 {filepath} 时发生 UnicodeDecodeError。请确保文件编码为 UTF-8。")
        return [], []

    test_dataset = Dataset.from_dict({"text": test_texts})
    tokenized_test_dataset = test_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length"),
        batched=True
    )
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])
    tokenized_test_dataset.set_format("torch")

    trainer = Trainer(model=model)
    predictions = trainer.predict(tokenized_test_dataset)

    logits = predictions.predictions
    predicted_labels = np.argmax(logits, axis=-1)
    
    final_predictions = np.where(predicted_labels == 1, 1, -1)
    
    return test_ids, final_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"检测到设备: {device}")
    if device.type == "cpu":
        print("警告: 强烈建议使用 GPU 进行 Transformer 模型训练，否则速度会非常慢。")

    tokenized_train_dataset, tokenized_val_dataset, tokenizer = load_and_prepare_data(MODEL_NAME)
    if tokenized_train_dataset is None:
        return

    print(f"\n步骤 4: 加载预训练模型 {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        dataloader_num_workers=0, # Windows上设为0
        # fp16=True if torch.cuda.is_available() else False, # 开启混合精度训练 (如果支持)
    )

    print("\n步骤 5: 创建 Trainer 并开始模型微调...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("模型微调完成。")

    print("\n步骤 6: 在验证集上评估最终模型...")
    eval_results = trainer.evaluate()
    print(f"验证集评估结果: {eval_results}")
    
    test_ids, final_predictions = load_test_data_and_predict(tokenizer, model, TEST_FILE)

    if test_ids and final_predictions.size > 0:
        with open(SUBMISSION_FILENAME, "w", encoding='utf-8') as f:
            f.write("Id,Prediction\n")
            for i in range(len(test_ids)):
                f.write(f"{test_ids[i]},{final_predictions[i]}\n")
        print(f"\n提交文件已保存到 {SUBMISSION_FILENAME}。")
    else:
        print("未生成提交文件，测试数据加载或预测失败。")

if __name__ == "__main__":
    main()


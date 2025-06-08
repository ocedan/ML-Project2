import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC # Optionally use LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import os
from nltk.tokenize import TweetTokenizer # NLTK TweetTokenizer for tokenization
# import fasttext # NOT importing fasttext globally for type-checking now
# import fasttext.util # NOT importing fasttext globally for type-checking now

# --- Configuration for training data files (using _full versions for consistency) ---
TRAIN_POS_FILE = "twitter-datasets/train_pos_full.txt"
TRAIN_NEG_FILE = "twitter-datasets/train_neg_full.txt"

# --- Path to the test data file ---
TEST_FILE = "twitter-datasets/test_data.txt"

# --- Directory for storing pre-trained embeddings ---
EMBEDDINGS_DIR = "pretrained_embeddings"

# --- Paths for pre-trained GloVe Twitter model (if downloaded and processed by load_pretrained_embeddings.py) ---
GLOVE_TWITTER_VOCAB_FILE = os.path.join(EMBEDDINGS_DIR, "vocab_glove_twitter.pkl")
GLOVE_TWITTER_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings_glove_twitter.npy")

# --- Path for pre-trained FastText model (downloaded by fasttext.util.download_model to current working directory) ---
FASTTEXT_LANG = 'en'
FASTTEXT_MODEL_DIM = 300 
FASTTEXT_BIN_MODEL_PATH = os.path.join(os.getcwd(), f"cc.{FASTTEXT_LANG}.{FASTTEXT_MODEL_DIM}.bin")

# --- Paths for self-trained GloVe model ---
SELF_TRAINED_GLOVE_VOCAB_FILE = "vocab.pkl"
SELF_TRAINED_GLOVE_EMBEDDINGS_FILE = "embeddings.npy"

# --- Select which embedding model to use ---
# Options:
#   "self_trained_glove" (Uses your self-trained GloVe)
#   "fasttext" (Uses pre-trained FastText, loaded directly from its .bin file)
#   "glove_twitter" (Uses pre-trained GloVe Twitter, loaded from saved .pkl/.npy)
EMBEDDING_TYPE = "self_trained_glove" # <--- MODIFY THIS LINE to choose different models

# --- Global variable to hold FastText model type (if successfully loaded) ---
# Initialize to None, will be set in main() if EMBEDDING_TYPE is "fasttext"
_FastTextModelType = None 

def load_data(filepath, embeddings_source, is_test_data=False):
    """
    Loads text data and constructs features by averaging word vectors,
    using NLTK TweetTokenizer for tokenization.
    This function handles two types of embeddings_source:
    1. fasttext.FastText model object (checked by hasattr for key methods)
    2. (vocab_dict, embeddings_array) tuple (for GloVe models, looks up via dictionary)
    
    Args:
        filepath (str): Path to the text data file.
        embeddings_source: The source of word embeddings, can be a fasttext.FastText model object 
                           or a (vocab_dict, embeddings_array) tuple.
        is_test_data (bool): True if it's a test data file.
        
    Returns:
        tuple: Contains feature matrix (np.array), corresponding raw texts (list),
               and a list of tweet IDs (list).
    """
    features = []
    raw_texts = []
    tweet_ids = []
    
    tokenizer = TweetTokenizer() 
    
    # Determine embedding dimension based on embeddings_source type
    embedding_dim = 0
    # Use duck typing: check if the object has methods characteristic of a FastText model
    # This avoids directly referencing fasttext.FastText type if it's broken
    if hasattr(embeddings_source, 'get_dimension') and hasattr(embeddings_source, 'get_word_vector'):
        embedding_dim = embeddings_source.get_dimension()
    else: # Assume it's a (vocab_dict, embeddings_array) tuple
        _, embeddings_array = embeddings_source
        embedding_dim = embeddings_array.shape[1]

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line_stripped = line.strip()
                
                current_tweet_id = str(line_idx + 1)
                text_content = line_stripped
                
                if is_test_data:
                    # Assuming test file format is "ID,TEXT" (comma-separated)
                    parts = line_stripped.split(',', 1) 
                    if len(parts) == 2:
                        current_tweet_id = parts[0]
                        text_content = parts[1]
                
                tweet_ids.append(current_tweet_id)
                raw_texts.append(text_content)
                
                # Tokenize using TweetTokenizer and lowercase
                tokens = [token.lower() for token in tokenizer.tokenize(text_content)]
                
                tweet_vectors = []
                # Use duck typing here as well for getting word vectors
                if hasattr(embeddings_source, 'get_word_vector'): # Characteristic of FastText model
                    for t in tokens:
                        tweet_vectors.append(embeddings_source.get_word_vector(t))
                else: # For GloVe (self-trained or pre-trained GloVe Twitter), use vocab lookup
                    vocab_dict, embeddings_array = embeddings_source # Unpack
                    for t in tokens:
                        idx = vocab_dict.get(t, -1)
                        if idx != -1: # Only include words present in the vocabulary
                            tweet_vectors.append(embeddings_array[idx])
                
                if tweet_vectors:
                    # Average all word vectors to form the tweet's feature
                    tweet_feature = np.mean(tweet_vectors, axis=0)
                else:
                    # If no valid words/tokens in the tweet, use a zero vector as feature
                    tweet_feature = np.zeros(embedding_dim)
                
                features.append(tweet_feature)
    except FileNotFoundError:
        print(f"错误: 数据文件未找到: {filepath}")
        return np.array([]), [], []
    except UnicodeDecodeError:
        print(f"错误: 读取 {filepath} 时发生 UnicodeDecodeError。请确保文件编码为 UTF-8。")
        return np.array([]), [], []
    except Exception as e:
        print(f"错误: 从 {filepath} 加载数据时发生异常: {e}")
        return np.array([]), [], []
        
    return np.array(features), raw_texts, tweet_ids

def main():
    # 1. Load vocabulary and embeddings based on EMBEDDING_TYPE
    print(f"加载词汇表和词嵌入 (使用: {EMBEDDING_TYPE})...")
    
    embeddings_source = None # This will hold either the fasttext model object or (vocab, embeddings_array) tuple
    embedding_dim = 0 

    try:
        if EMBEDDING_TYPE == "self_trained_glove":
            current_vocab_file = SELF_TRAINED_GLOVE_VOCAB_FILE
            current_embeddings_file = SELF_TRAINED_GLOVE_EMBEDDINGS_FILE
            
            if not os.path.exists(current_vocab_file) or not os.path.exists(current_embeddings_file):
                print(f"错误: {current_vocab_file} 或 {current_embeddings_file} 未找到。请确保已生成。")
                print("建议: 运行 'preprocess_and_cooccurrence.py' 来生成自训练模型的词汇表和共现矩阵，然后运行你的 GloVe 训练脚本。")
                return
            with open(current_vocab_file, "rb") as f:
                vocab = pickle.load(f)
            embeddings_array = np.load(current_embeddings_file)
            embeddings_source = (vocab, embeddings_array)
            embedding_dim = embeddings_array.shape[1]
            print(f"词汇表已加载 (大小: {len(vocab)})。词嵌入已加载 (形状: {embeddings_array.shape})。")

        elif EMBEDDING_TYPE == "fasttext":
            # Only import fasttext and load model if this type is chosen
            try:
                import fasttext # Local import to control when it's accessed
                import fasttext.util # Local import
                global _FastTextModelType # Reference the global variable
                _FastTextModelType = fasttext.FastText # Safely attempt to get the type here
            except (ImportError, AttributeError, Exception) as e:
                print(f"致命错误: 无法导入 fasttext 库或访问 fasttext.FastText 类型。请确保 fasttext 库已正确安装且其底层 C++ 模块正常工作。错误: {e}")
                return # Exit if fasttext is critical and broken

            if not os.path.exists(FASTTEXT_BIN_MODEL_PATH):
                print(f"错误: FastText模型 {FASTTEXT_BIN_MODEL_PATH} 未找到。请确保已下载。")
                print("建议: 如果要使用 FastText，请运行 'load_pretrained_embeddings.py' 并将其中 'DOWNLOAD_FASTTEXT' 设置为 True 来下载模型。")
                return 
            
            ft_model = fasttext.load_model(FASTTEXT_BIN_MODEL_PATH)
            embeddings_source = ft_model
            embedding_dim = ft_model.get_dimension()
            print(f"FastText 模型已加载。维度: {embedding_dim}")

        elif EMBEDDING_TYPE == "glove_twitter":
            current_vocab_file = GLOVE_TWITTER_VOCAB_FILE
            current_embeddings_file = GLOVE_TWITTER_EMBEDDINGS_FILE
            
            if not os.path.exists(current_vocab_file) or not os.path.exists(current_embeddings_file):
                print(f"错误: {current_vocab_file} 或 {current_embeddings_file} 未找到。请确保已下载并保存。")
                print("建议: 运行 'load_pretrained_embeddings.py' 来下载并保存该模型。", flush=True)
                return
            with open(current_vocab_file, "rb") as f:
                vocab = pickle.load(f)
            embeddings_array = np.load(current_embeddings_file)
            embeddings_source = (vocab, embeddings_array)
            embedding_dim = embeddings_array.shape[1]
            print(f"词汇表已加载 (大小: {len(vocab)})。词嵌入已加载 (形状: {embeddings_array.shape})。")
        else:
            print("错误: 无效的 EMBEDDING_TYPE。请选择 'self_trained_glove', 'fasttext', 或 'glove_twitter'。")
            return

    except Exception as e:
        print(f"错误: 加载词汇表/词嵌入时发生异常: {e}")
        return

    # 2. Construct features for training texts
    print("构建训练文本特征...")
    X_pos, _, _ = load_data(TRAIN_POS_FILE, embeddings_source)
    X_neg, _, _ = load_data(TRAIN_NEG_FILE, embeddings_source)

    if X_pos.size == 0 or X_neg.size == 0:
        print("错误: 未能加载训练数据。退出。")
        return

    X_train = np.vstack((X_pos, X_neg))
    y_train = np.array([1] * len(X_pos) + [0] * len(X_neg))
    
    print(f"训练数据形状: {X_train.shape}，训练标签形状: {y_train.shape}")

    # 3. Train a linear classifier
    print("训练线性分类器 (逻辑回归)...")
    model = LogisticRegression(max_iter=1000, random_state=42) 
    model.fit(X_train, y_train)
    print("分类器训练完成。")

    # 4. Predict labels and generate submission.csv
    print("在训练数据上进行预测 (仅供演示)...")
    y_pred_train = model.predict(X_train)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"训练准确率: {train_accuracy:.4f}")
    print("\n训练分类报告:")
    print(classification_report(y_train, y_pred_train, target_names=['负面', '正面']))

    print("\n加载测试数据并生成提交文件...")
    X_test, _, test_ids = load_data(TEST_FILE, embeddings_source, is_test_data=True) 
    
    if X_test.size > 0:
        y_pred_test = model.predict(X_test)
        
        # Convert predicted labels from 0/1 to -1/1
        y_pred_submission = np.where(y_pred_test == 1, 1, -1)
        
        # Debugging Prediction Output
        print("\n--- 调试预测输出 ---")
        print(f"示例测试 ID (前5个): {test_ids[:5]}")
        print(f"测试 ID 的类型: {type(test_ids[0])}")
        print(f"示例提交预测 (前5个): {y_pred_submission[:5]}")
        print(f"提交预测的类型: {type(y_pred_submission[0])}")
        print("--------------------")

        # Save predictions to submission.csv file
        submission_filename = "submission.csv"
        with open(submission_filename, "w", encoding='utf-8') as f:
            f.write("Id,Prediction\n")
            for i, pred in enumerate(y_pred_submission):
                f.write(f"{test_ids[i]},{pred}\n")
        print(f"预测结果已保存到 {submission_filename}，格式符合要求。")
    else:
        print("未找到或加载测试数据，跳过测试预测和提交文件生成。")

if __name__ == "__main__":
    # Ensure NLTK resources are downloaded if not already
    try:
        TweetTokenizer() # Attempt to initialize to check if resources are ready
    except LookupError:
        print("NLTK 'punkt' 资源未找到。正在下载...")
        import nltk
        nltk.download('punkt')
        print("下载完成。请重新运行脚本。")
        exit() 
    
    main()


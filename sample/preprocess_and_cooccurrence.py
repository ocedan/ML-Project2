# preprocess_and_cooccurrence.py (Updated for Detailed Error Logging and Memory Optimization)

import numpy as np
import pickle
import os
from collections import Counter
from scipy.sparse import coo_matrix
from nltk.tokenize import TweetTokenizer 
# If you don't have nltk installed: pip install nltk
# If you get resource errors for TweetTokenizer: 
# import nltk
# nltk.download('punkt')


# --- 配置参数 ---
TRAIN_POS_FULL_FILE = "twitter-datasets/train_pos_full.txt"
TRAIN_NEG_FULL_FILE = "twitter-datasets/train_neg_full.txt"

VOCAB_PKL_FILE = "vocab.pkl"
COOC_PKL_FILE = "cooc.pkl"

MIN_WORD_COUNT = 5 

def build_vocabulary_and_filter(min_count=MIN_WORD_COUNT):
    """
    构建词汇表，并进行低频词过滤。
    使用 NLTK TweetTokenizer 进行分词，并进行小写化。
    """
    print("Step 1: Building raw vocabulary and counting frequencies (using TweetTokenizer)...")
    word_counts = Counter()
    tokenizer = TweetTokenizer() 
    
    for fn in [TRAIN_POS_FULL_FILE, TRAIN_NEG_FULL_FILE]:
        if not os.path.exists(fn):
            print(f"Error: Training file not found: {fn}")
            print("Please ensure 'twitter-datasets' folder and its content are in the correct directory.")
            return None
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = [token.lower() for token in tokenizer.tokenize(line.strip())]
                    word_counts.update(tokens)
        except UnicodeDecodeError:
            print(f"Error: UnicodeDecodeError when reading {fn}. Please ensure file encoding is UTF-8.")
            return None
        except Exception as e:
            print(f"Error reading {fn}: {e}")
            return None

    print(f"Raw vocabulary size: {len(word_counts)}")
    
    print(f"Step 2: Filtering low-frequency words (min_count={min_count})...")
    filtered_vocab_list = [word for word, count in word_counts.most_common() if count >= min_count]
    
    vocab = {word: idx for idx, word in enumerate(filtered_vocab_list)}
    
    print(f"Filtered vocabulary size: {len(vocab)}")
    
    print(f"Step 3: Saving vocabulary to {VOCAB_PKL_FILE}...")
    try:
        with open(VOCAB_PKL_FILE, "wb") as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
        print("Vocabulary saved successfully.")
    except Exception as e:
        print(f"Error saving vocabulary: {e}")
        return None
        
    return vocab

def build_cooccurrence_matrix(vocab, window_size=10): # Added window_size parameter
    """
    构建词语共现矩阵。
    使用 NLTK TweetTokenizer 进行分词，并进行小写化。
    现在也使用 _full 数据集。
    引入滑动窗口限制共现范围，并直接统计唯一共现对的次数以节省内存。
    """
    print(f"\nStep 4: Building co-occurrence matrix (using TweetTokenizer, _full data, and window_size={window_size})...")
    if vocab is None:
        print("Vocabulary is not available. Cannot build co-occurrence matrix.")
        return None

    # 使用字典存储 (row_idx, col_idx): count，避免创建超大列表
    cooccurrence_counts = {} 

    line_counter = 0
    tokenizer = TweetTokenizer() 

    for fn in [TRAIN_POS_FULL_FILE, TRAIN_NEG_FULL_FILE]: 
        if not os.path.exists(fn):
            print(f"Error: Training file not found for co-occurrence: {fn}")
            return None
        try:
            with open(fn, encoding='utf-8') as f:
                for line in f:
                    line_counter += 1
                    tokens = [token.lower() for token in tokenizer.tokenize(line.strip())]
                    tokens_indices = [vocab.get(t, -1) for t in tokens]
                    valid_tokens_indices = [t for t in tokens_indices if t >= 0]
                    
                    # 实现滑动窗口共现统计
                    for i, target_word_idx in enumerate(valid_tokens_indices):
                        # 定义共现窗口，前后各 window_size 个词
                        start_idx = max(0, i - window_size)
                        end_idx = min(len(valid_tokens_indices), i + window_size + 1) # +1 是因为 range 是 exclusive
                        
                        for j in range(start_idx, end_idx):
                            context_word_idx = valid_tokens_indices[j]
                            
                            # 避免词与自身共现（标准 GloVe 做法）
                            if target_word_idx == context_word_idx:
                                continue 
                            
                            # 直接在字典中累加计数
                            pair = (target_word_idx, context_word_idx)
                            cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1

                    if line_counter % 100000 == 0: 
                        print(f"Processed {line_counter} lines for co-occurrence. Current unique pairs: {len(cooccurrence_counts)}")
        except UnicodeDecodeError as ude: 
            print(f"Error: UnicodeDecodeError when reading {fn} for co-occurrence: {ude}")
            print(f"Error occurred at line number: {line_counter}") 
            return None
        except Exception as e: 
            print(f"Error reading {fn} for co-occurrence: {type(e).__name__}: {e}") 
            print(f"Error occurred at line number: {line_counter}") 
            return None

    if not cooccurrence_counts: 
        print("No co-occurrence data collected. Check input files or vocabulary.")
        return None

    # 从字典转换回 data, row, col 列表，此时它们只包含唯一的共现对及其总数
    data = list(cooccurrence_counts.values())
    row = [pair[0] for pair in cooccurrence_counts.keys()]
    col = [pair[1] for pair in cooccurrence_counts.keys()]

    print(f"Collected {len(data)} unique co-occurrence pairs. Building coo_matrix...")
    vocab_size = len(vocab)
    # coo_matrix 期望 data, (row_indices, col_indices)
    cooc = coo_matrix((data, (row, col)), shape=(vocab_size, vocab_size))
    
    # sum_duplicates 在这种直接累加的模式下通常不再需要，但保留以防万一或未来修改。
    print("Summing duplicates (if any remain from dictionary accumulation, usually not needed)...")
    cooc.sum_duplicates() 
    
    print(f"Final co-occurrence matrix has {cooc.nnz} non-zero entries.")

    print(f"Saving co-occurrence matrix to {COOC_PKL_FILE}...")
    try:
        with open(COOC_PKL_FILE, "wb") as f:
            pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
        print("Co-occurrence matrix saved successfully.")
    except Exception as e:
        print(f"Error saving co-occurrence matrix: {type(e).__name__}: {e}") 
        return None

    return cooc

def main():
    # 1. 构建词汇表并过滤
    vocab = build_vocabulary_and_filter()
    if vocab is None:
        print("Vocabulary building failed. Exiting.")
        return

    # 2. 构建共现矩阵
    cooccurrence_matrix = build_cooccurrence_matrix(vocab) # 默认使用 window_size=10
    if cooccurrence_matrix is None:
        print("Co-occurrence matrix building failed. Exiting.")
        return

    print("\nPreprocessing and co-occurrence matrix generation complete!")

if __name__ == "__main__":
    # Ensure NLTK resources are downloaded if not already
    try:
        TweetTokenizer() # Attempt to initialize to check if resources are ready
    except LookupError:
        print("NLTK 'punkt' resource not found. Downloading...")
        import nltk
        nltk.download('punkt')
        print("Download complete. Please re-run the script.")
        exit() # Exit to ensure resources are available on next run
    
    main()


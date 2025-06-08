import numpy as np
import pickle
import os
import requests
import zipfile
import io
import gzip
import logging
import fasttext.util # Import fasttext.util
import fasttext # Import fasttext

# Set up logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Directories for storing downloaded models
EMBEDDINGS_DIR = "pretrained_embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Control whether to download/load FastText. Set to False to skip.
DOWNLOAD_FASTTEXT = False 

# NEW: Control whether to download/extract GloVe Twitter ZIP. Set to False to skip.
# This assumes the .txt file (e.g., glove.twitter.27B.100d.txt) is ALREADY extracted
# into the 'glove_twitter_extracted' subdirectory.
PROCESS_GLOVE_ZIP = False # <--- 在这里修改以控制是否下载/解压GloVe ZIP

# FastText Configuration (only used if DOWNLOAD_FASTTEXT is True)
FASTTEXT_LANG = 'en'
FASTTEXT_MODEL_DIM = 300 
FASTTEXT_BIN_FILENAME = f"cc.{FASTTEXT_LANG}.{FASTTEXT_MODEL_DIM}.bin" 
FASTTEXT_BIN_MODEL_PATH = os.path.join(os.getcwd(), FASTTEXT_BIN_FILENAME) # Default download path

# GloVe Twitter Configuration
GLOVE_TWITTER_MODEL_URL = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
GLOVE_TWITTER_MODEL_FILENAME = "glove.twitter.27B.zip"
GLOVE_TWITTER_VEC_FILENAME = "glove.twitter.27B.100d.txt" 
GLOVE_TWITTER_EMBEDDING_DIM = 100 

# --- Helper Functions (for GloVe download/extraction/load) ---
def download_file(url, destination_path):
    """Downloads a file from a given URL to a destination path."""
    if os.path.exists(destination_path):
        logging.info(f"文件已存在: {destination_path}. 跳过下载。")
        return True
    
    logging.info(f"正在下载 {url} 到 {destination_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192 # 8KB
        downloaded_size = 0

        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Basic progress indicator
                    if total_size > 0:
                        progress = downloaded_size * 100 / total_size
                        # Update every 8MB or every 1% if it's a smaller file
                        if total_size > 0 and (downloaded_size % (block_size * 1000) == 0 or progress % 1 == 0): 
                            logging.info(f"已下载: {downloaded_size / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB ({progress:.2f}%)")
        logging.info(f"下载完成: {destination_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"下载 {url} 失败: {e}")
        return False

def extract_zip(zip_path, extract_to_dir):
    """Extracts a zip file."""
    if not os.path.exists(zip_path):
        logging.error(f"Zip 文件未找到: {zip_path}")
        return False
    
    logging.info(f"正在解压 {zip_path} 到 {extract_to_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        logging.info(f"解压完成: {zip_path}")
        return True
    except zipfile.BadZipFile as e:
        logging.error(f"损坏的 zip 文件: {zip_path}. 错误: {e}")
        return False
    except Exception as e:
        logging.error(f"解压 {zip_path} 失败: {e}")
        return False

def load_glove_embeddings(filepath, embedding_dim):
    """
    Loads GloVe embeddings from a .txt file.
    GloVe .txt files usually have format: "word float1 float2 ... floatN"
    """
    logging.info(f"正在从 {filepath} 加载 GloVe 词嵌入...")
    vocab = {}
    embeddings = []
    
    if not os.path.exists(filepath):
        logging.error(f"GloVe .txt 文件未找到: {filepath}")
        return None, None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split(' ')
                # Expecting word + embedding_dim values
                if len(parts) != embedding_dim + 1: 
                    logging.warning(f"跳过 {filepath} 中的格式错误行 {idx+1}: {line.strip()}")
                    continue
                
                word = parts[0]
                vector = np.array([float(val) for val in parts[1:]])
                
                vocab[word] = len(vocab) # Assign sequential index
                embeddings.append(vector)
                
        embeddings_array = np.array(embeddings, dtype=np.float32)
        logging.info(f"从 GloVe 加载了 {len(vocab)} 个词，维度为 {embedding_dim}。")
        logging.info(f"词嵌入数组形状: {embeddings_array.shape}")
        return vocab, embeddings_array
    except Exception as e:
        logging.error(f"加载 GloVe 词嵌入失败: {e}")
        return None, None


def main():
    print("--- 开始预训练词嵌入的下载和加载 ---")

    # --- FastText (可选下载/加载) ---
    if DOWNLOAD_FASTTEXT:
        logging.info(f"FastText 下载已启用。尝试下载模型: {FASTTEXT_BIN_FILENAME}")
        try:
            # Download and load FastText model
            # if_exists='ignore' will skip download if file already exists
            ft_model = fasttext.util.download_model(FASTTEXT_LANG, if_exists='ignore') 
            
            logging.info(f"FastText 模型下载并加载成功。")
            
            # Extract vocab and embeddings from the FastText model object
            fasttext_words = ft_model.get_words()
            fasttext_vocab = {word: idx for idx, word in enumerate(fasttext_words)}
            
            fasttext_embeddings_list = []
            for word in fasttext_words:
                fasttext_embeddings_list.append(ft_model.get_word_vector(word))
            fasttext_embeddings = np.array(fasttext_embeddings_list, dtype=np.float32)

            logging.info(f"FastText 模型已加载: 词汇量={len(fasttext_vocab)}, 词嵌入形状={fasttext_embeddings.shape}")
            
            # --- Save FastText for classifier use ---
            try:
                vocab_fasttext_path = os.path.join(EMBEDDINGS_DIR, "vocab_fasttext.pkl")
                embeddings_fasttext_path = os.path.join(EMBEDDINGS_DIR, "embeddings_fasttext.npy")
                
                with open(vocab_fasttext_path, "wb") as f:
                    pickle.dump(fasttext_vocab, f, pickle.HIGHEST_PROTOCOL)
                np.save(embeddings_fasttext_path, fasttext_embeddings)
                logging.info(f"FastText 词汇表和词嵌入已保存供分类器使用 ({vocab_fasttext_path}, {embeddings_fasttext_path})。")
            except Exception as e:
                logging.error(f"保存 FastText 供分类器使用失败: {e}")

        except Exception as e:
            logging.error(f"FastText 模型无法加载/下载: {e}. 请确保 'fasttext' 已安装并检查网络连接。")
            # Ensure they are None on failure so we don't try to save
            fasttext_vocab, fasttext_embeddings = None, None 
    else:
        logging.info("FastText 下载已禁用。跳过 FastText 模型处理。")

    print("\n" + "="*50 + "\n") # Separator

    # --- GloVe Twitter ---
    glove_zip_path = os.path.join(EMBEDDINGS_DIR, GLOVE_TWITTER_MODEL_FILENAME)
    glove_extract_dir = os.path.join(EMBEDDINGS_DIR, "glove_twitter_extracted")
    glove_vec_path = os.path.join(glove_extract_dir, GLOVE_TWITTER_VEC_FILENAME)
    os.makedirs(glove_extract_dir, exist_ok=True) # Ensure extraction directory exists

    if PROCESS_GLOVE_ZIP: # NEW: Only attempt download/extract if this flag is True
        logging.info(f"GloVe Twitter ZIP 处理已启用。尝试下载/解压模型: {GLOVE_TWITTER_MODEL_FILENAME}")
        if download_file(GLOVE_TWITTER_MODEL_URL, glove_zip_path):
            extract_zip(glove_zip_path, glove_extract_dir)
    else:
        logging.info("GloVe Twitter ZIP 处理已禁用。假定 .txt 文件已存在于解压目录。")
    
    glove_vocab, glove_embeddings = None, None
    if os.path.exists(glove_vec_path):
        glove_vocab, glove_embeddings = load_glove_embeddings(glove_vec_path, GLOVE_TWITTER_EMBEDDING_DIM)
    
    if glove_vocab and glove_embeddings is not None:
        logging.info(f"GloVe Twitter 模型已加载: 词汇量={len(glove_vocab)}, 词嵌入形状={glove_embeddings.shape}")
        # --- 保存GloVe Twitter为text_classifier可用的格式 ---
        try:
            vocab_glove_twitter_path = os.path.join(EMBEDDINGS_DIR, "vocab_glove_twitter.pkl")
            embeddings_glove_twitter_path = os.path.join(EMBEDDINGS_DIR, "embeddings_glove_twitter.npy")

            with open(vocab_glove_twitter_path, "wb") as f:
                pickle.dump(glove_vocab, f, pickle.HIGHEST_PROTOCOL)
            np.save(embeddings_glove_twitter_path, glove_embeddings)
            logging.info(f"GloVe Twitter 词汇表和词嵌入已保存供分类器使用 ({vocab_glove_twitter_path}, {embeddings_glove_twitter_path})。")
        except Exception as e:
            logging.error(f"保存 GloVe Twitter 供分类器使用失败: {e}")
    else:
        logging.info("\nGloVe Twitter 模型无法加载。请检查日志或文件是否存在。")
    
    print("\n--- 预训练词嵌入处理流程结束 ---")

if __name__ == "__main__":
    main()


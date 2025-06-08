import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC # Optionally use LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import os
from nltk.tokenize import TweetTokenizer # Import TweetTokenizer

# Ensure training data files exist (now using _full versions for consistency)
TRAIN_POS_FILE = "twitter-datasets/train_pos_full.txt" # Changed to _full
TRAIN_NEG_FILE = "twitter-datasets/train_neg_full.txt" # Changed to _full

# Ensure generated vocabulary and embeddings files exist
VOCAB_FILE = "vocab.pkl"
EMBEDDINGS_FILE = "embeddings.npy"

# Test data file path
TEST_FILE = "twitter-datasets/test_data.txt"

def load_data(filepath, vocab, embeddings, is_test_data=False):
    """
    Loads text data and constructs features by averaging word vectors.
    Uses NLTK TweetTokenizer for tokenization.
    
    Args:
        filepath (str): Path to the text data file.
        vocab (dict): Vocabulary mapping words to indices.
        embeddings (np.array): Word embedding matrix, each row is a word vector.
        is_test_data (bool): True if it's a test data file (may contain IDs).
        
    Returns:
        tuple: Contains feature matrix (np.array), corresponding raw texts (list),
               and a list of tweet IDs (list).
               tweet_ids will always be populated: if is_test_data is True and ID is parsed, it's used; 
               otherwise, line number is used as ID.
    """
    features = []
    raw_texts = []
    tweet_ids = []
    
    tokenizer = TweetTokenizer() # Initialize tokenizer inside the function
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line_stripped = line.strip()
                
                current_tweet_id = str(line_idx + 1) # Default ID: line number starting from 1
                text_content = line_stripped # Default content: entire line
                
                if is_test_data:
                    # Assuming test file format is "ID,TEXT" (comma-separated)
                    parts = line_stripped.split(',', 1) # Split only on the first comma
                    if len(parts) == 2:
                        current_tweet_id = parts[0]
                        text_content = parts[1]
                    # If format is not "ID,TEXT", it falls back to line number as ID and entire line as content
                
                tweet_ids.append(current_tweet_id)
                raw_texts.append(text_content)
                
                # Tokenize using TweetTokenizer and lowercase
                tokens = [token.lower() for token in tokenizer.tokenize(text_content)]
                
                # Get vocabulary indices for tokens
                tokens_indices = [vocab.get(t, -1) for t in tokens]
                # Filter out tokens not in vocabulary (index -1)
                valid_tokens = [t for t in tokens_indices if t != -1]
                
                if valid_tokens:
                    # Get word vectors for valid tokens
                    word_vectors = embeddings[valid_tokens]
                    # Average all word vectors to form the tweet's feature
                    tweet_feature = np.mean(word_vectors, axis=0)
                else:
                    # If no valid words in the tweet, use a zero vector as feature
                    tweet_feature = np.zeros(embeddings.shape[1])
                
                features.append(tweet_feature)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return np.array([]), [], []
    except UnicodeDecodeError:
        print(f"Error: UnicodeDecodeError when reading {filepath}. Please ensure file encoding is UTF-8.")
        return np.array([]), [], []
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return np.array([]), [], []
        
    return np.array(features), raw_texts, tweet_ids

def main():
    # 1. Load vocabulary and embeddings
    print("Loading vocabulary and embeddings...")
    try:
        if not os.path.exists(VOCAB_FILE) or not os.path.exists(EMBEDDINGS_FILE):
             print(f"Error: {VOCAB_FILE} or {EMBEDDINGS_FILE} not found. Please ensure they are generated.")
             print("Recommended steps: 1. Run preprocess_and_cooccurrence.py; 2. Run your GloVe training script.")
             return

        with open(VOCAB_FILE, "rb") as f:
            vocab = pickle.load(f)
        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"Vocabulary loaded (size: {len(vocab)}). Embeddings loaded (shape: {embeddings.shape}).")
    except Exception as e:
        print(f"Error loading vocab/embeddings: {e}")
        return

    # 2. Construct features for training texts
    print("Constructing features for training texts...")
    X_pos, _, _ = load_data(TRAIN_POS_FILE, vocab, embeddings)
    X_neg, _, _ = load_data(TRAIN_NEG_FILE, vocab, embeddings)

    if X_pos.size == 0 or X_neg.size == 0:
        print("Error: Failed to load training data. Exiting.")
        return

    X_train = np.vstack((X_pos, X_neg))
    y_train = np.array([1] * len(X_pos) + [0] * len(X_neg))
    
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")

    # 3. Train a linear classifier
    print("Training a linear classifier (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Classifier training complete.")

    # 4. Predict labels and generate submission.csv
    print("Making predictions on training data (for demonstration)...")
    y_pred_train = model.predict(X_train)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_pred_train, target_names=['Negative', 'Positive']))

    print("\nLoading test data and generating submission file...")
    # Since test_data.txt is "ID,TEXT" comma-separated, is_test_data=True is needed
    X_test, _, test_ids = load_data(TEST_FILE, vocab, embeddings, is_test_data=True) 
    
    if X_test.size > 0:
        y_pred_test = model.predict(X_test)
        
        # Convert predicted labels from 0/1 to -1/1
        y_pred_submission = np.where(y_pred_test == 1, 1, -1)
        
        # Debugging Prediction Output (kept for confirmation)
        print("\n--- Debugging Prediction Output ---")
        print(f"Sample test_ids (first 5): {test_ids[:5]}")
        print(f"Type of test_ids[0]: {type(test_ids[0])}")
        print(f"Sample y_pred_submission (first 5): {y_pred_submission[:5]}")
        print(f"Type of y_pred_submission[0]: {type(y_pred_submission[0])}")
        print("-----------------------------------")

        # Save predictions to submission.csv file
        submission_filename = "submission.csv"
        with open(submission_filename, "w", encoding='utf-8') as f:
            f.write("Id,Prediction\n")
            for i, pred in enumerate(y_pred_submission):
                f.write(f"{test_ids[i]},{pred}\n")
        print(f"Predictions saved to {submission_filename} in the required format.")
    else:
        print("No test data found or loaded, skipping test prediction and submission file generation.")

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


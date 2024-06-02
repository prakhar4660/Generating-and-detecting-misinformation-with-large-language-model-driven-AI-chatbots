import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("gpu")
else:
    device = torch.device("cpu")
    print("cpu")

# Load the model
model_path = 'BERT_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)  # Move the model to the device

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def read_and_clean_tweets_from_csv(filenames):
    all_tweets = []

    for filename in filenames:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filename, encoding='ISO-8859-1')

        # Check if 'Tweet' column exists in the dataframe
        if 'Tweet' not in df.columns:
            print(f"'Tweet' column not found in {filename}. Skipping...")
            continue

        # Drop rows where 'Tweet' column is empty or NaN
        df = df[df['Tweet'].str.len() >= 20]

        # Append the cleaned tweets to the all_tweets list
        all_tweets.extend(df['Tweet'].tolist())

    return all_tweets

# Example CSV file paths
filenames = [
    #'generated_tweets.csv'
    # 'generated_tweets2.csv',
    #'generated_tweets3.csv'
    'generated_tweets4.csv'
]

tweets = read_and_clean_tweets_from_csv(filenames)

# Print the number of valid tweets gathered
print(f"Number of valid tweets: {len(tweets)}")

# If you want to see the tweets, you can print them too
# for tweet in tweets:
#     print(tweet)

test_texts = tweets
test_labels = [0] * len(test_texts)

test_loader = test_texts  # DataLoader for your test data

def predict(model, dataloader, tokenizer):
    all_predictions = []
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Predicting", unit="batch"):
            # Tokenize inputs
            inputs = tokenizer(inputs, return_tensors='pt', truncation=True, padding=True, max_length=512)
            
            # Send inputs to the device
            for key, value in inputs.items():
                inputs[key] = value.to(device)
            
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
    return np.array(all_predictions)

y_pred = predict(model, test_loader, tokenizer)

# Ground truth labels
y_true = test_labels 

def accuracy(y_true, y_pred):
    # Check if both lists are of the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Both lists should have the same length.")
    
    correct_predictions = sum(t == p for t, p in zip(y_true, y_pred))
    return correct_predictions / len(y_true)


acc = accuracy(y_true, y_pred)
print(f"Accuracy: {acc * 100:.4f}%")

with open("result3.txt", "w") as f:
    f.write("Calculated Result:\n")
    f.write("-------------------\n")
    f.write(f"Accuracy: {acc * 100:.4f}%\n")
    f.write(f"Number of tweets: {len(tweets)}")

df = pd.DataFrame({'tweet': tweets, 'labels': y_pred})
df.to_csv('pred.csv', index=True)

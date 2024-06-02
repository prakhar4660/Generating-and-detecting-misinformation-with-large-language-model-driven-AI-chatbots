from transformers import BertTokenizerFast
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
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

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

df = pd.read_csv('test.csv', encoding='ISO-8859-1')
df['label'] = df['label'].map(
    {'true': 1, 'false': 0, 'mixture': 2, 'unproven': 3})
df['label'] = df['label'].replace('snopes', pd.NA)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)
test_texts = list(df['claim'])
test_labels = list(df['label'])

def predict(model, dataloader, tokenizer):
    all_predictions = []
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Predicting", unit="batch"):
            # Tokenize inputs
            inputs = tokenizer(inputs, return_tensors='pt',
                               truncation=True, padding=True, max_length=512)

            # Send inputs to the device
            for key, value in inputs.items():
                inputs[key] = value.to(device)

            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
    return np.array(all_predictions)

# Prepare DataLoader for your test data
test_loader = test_texts 

y_pred = predict(model, test_loader, tokenizer)

# Ground truth labels
y_true = test_labels

# Calculate metrics for all classes
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1, 2, 3])

# Save metrics to .txt file
with open("metrics_output.txt", "w") as f:
    f.write("Calculated Metrics:\n")
    f.write("-------------------\n")
    for i, label in enumerate(['False', 'True', 'Mixture', 'Unproven']):
        f.write(f"Precision ({label}): {precision[i]:.4f}\n")
        f.write(f"Recall ({label}): {recall[i]:.4f}\n")
        f.write(f"F1-score ({label}): {f1[i]:.4f}\n")

# Plotting
labels = ['False', 'True', 'Mixture', 'Unproven']
x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, precision, width, label='Precision')
rects2 = ax.bar(x + width/2, recall, width, label='Recall')
rects3 = ax.bar(x + 1.5*width, f1, width, label='F1-score')

ax.set_ylabel('Scores')
ax.set_title('Scores by class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig('model_metrics_plot.png')
plt.show()

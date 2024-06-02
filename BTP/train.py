from sklearn.metrics import confusion_matrix
from transformers import Trainer, TrainingArguments, BertForSequenceClassification
import torch
import numpy as np
from transformers import BertTokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

df = pd.read_csv('train.csv', encoding='ISO-8859-1')
# df = df.head(100)
df['label'] = df['label'].map(
    {'true': 1, 'false': 0, 'mixture': 2, 'unproven': 3})
df['label'] = df['label'].replace('snopes', pd.NA)
# df['label'] = df['label'].fillna(-1)
df = df.dropna(subset=['label'])
# Filter rows where the 'claim' column is a string
# df = df[df['claim'].apply(lambda x: isinstance(x, str))]
df['label'] = df['label'].astype(int)
train_texts = list(df['claim'])
train_labels = list(df['label'])
# train_labels=list(pd.get_dummies(train_labels,drop_first=True)['real'])

df = pd.read_csv('dev.csv', encoding='ISO-8859-1')
# df = df.head(30)
df['label'] = df['label'].map(
    {'true': 1, 'false': 0, 'mixture': 2, 'unproven': 3})
df['label'] = df['label'].replace('snopes', pd.NA)
# df['label'] = df['label'].fillna(-1)
df = df.dropna(subset=['label'])
# Filter rows where the 'claim' column is a string
# df = df[df['claim'].apply(lambda x: isinstance(x, str))]
df['label'] = df['label'].astype(int)
val_texts = list(df['claim'])
val_labels = list(df['label'])
# val_labels=list(pd.get_dummies(val_labels,drop_first=True)['real'])

df = pd.read_csv('test.csv', encoding='ISO-8859-1')
# df = df.head(30)
df['label'] = df['label'].map(
    {'true': 1, 'false': 0, 'mixture': 2, 'unproven': 3})
df['label'] = df['label'].replace('snopes', pd.NA)
# df['label'] = df['label'].fillna(-1)
df = df.dropna(subset=['label'])
# Filter rows where the 'claim' column is a string
# df = df[df['claim'].apply(lambda x: isinstance(x, str))]
df['label'] = df['label'].astype(int)
test_texts = list(df['claim'])
test_labels = list(df['label'])
# test_labels=list(pd.get_dummies(test_labels,drop_first=True)['real'])
# df.shape

# train_labels

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# !pip install transformers[torch]

# Load model directly

# tokenizer = AutoTokenizer.from_pretrained("sarkerlab/SocBERT-base")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# from transformers import DistilBertTokenizerFast
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# print(train_texts[:5])  # Print the first 5 elements of train_texts
# print(type(train_texts[0]))  # Print the data type of the first element


test_encodings = tokenizer(test_texts, truncation=True,
                           padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True,
                          padding=True, max_length=512)
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=512)

# print(len(val_texts))
# print(val_texts)
# print(type(val_encodings))
# print(val_encodings)


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

# train_dataset[5]


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    overwrite_output_dir=True,
    num_train_epochs=5,             # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    gradient_accumulation_steps=4,
    save_total_limit=1,
    load_best_model_at_end=True,
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=500
)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
# model = BertForSequenceClassification.from_pretrained("sarkerlab/SocBERT-base")

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("gpu")
else:
    device = torch.device("cpu")
    print("cpu")

# Move model and data to the device
model.to(device)

trainer = Trainer(
    # the instantiated 🤗 Transformers model to be trained
    model=model,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

print("done")
trainer.train()

trainer.evaluate(test_dataset)


trainer.predict(test_dataset)

trainer.predict(test_dataset)[1].shape

output = trainer.predict(test_dataset)[1]


# cm
cm = confusion_matrix(test_labels, output)
# Save confusion matrix to a file
np.savetxt('confusion_matrix.txt', cm, fmt='%d')

trainer.save_model('BERT_model')

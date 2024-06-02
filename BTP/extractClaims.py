import pandas as pd

def extract_max_words(text, max_words=150):
    words = text.split()[:max_words]
    return ' '.join(words)

# Load the CSV file into a DataFrame
df = pd.read_csv('train.csv', encoding='ISO-8859-1') 

# Specify the column you want to process
column_name = 'claim' 

# Filter rows where 'reliability' column has value 0
df = df[df['label'] == 'false']

# Apply the extraction function to the specified column
# df[column_name] = df[column_name].apply(extract_max_words)

# Keep only required columns
df = df[['claim_id', column_name, 'label']]

# Limit to a maximum of 20 lines
df = df.head(100) 
# df = df.iloc[40:60]

# Save the modified DataFrame back to a CSV file
df.to_csv('falseClaims.csv', index=False)
print("done")
s = "hvbid"
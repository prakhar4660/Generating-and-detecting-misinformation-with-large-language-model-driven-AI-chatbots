import pandas as pd

def extract_max_words(text, max_words=150):
    words = text.split()[:max_words]
    return ' '.join(words)

# Load the CSV file into a DataFrame
df = pd.read_csv('recovery-news-data.csv', encoding='ISO-8859-1') 

# Specify the column you want to process
column_name = 'body_text' 

# Filter rows where 'reliability' column has value 0
df = df[df['reliability'] == 0]

# Apply the extraction function to the specified column
df[column_name] = df[column_name].apply(extract_max_words)

# Keep only required columns
df = df[['news_id', 'title', column_name, 'reliability']]

# Limit to a maximum of 20 lines
# df = df.head(20) 
df = df.iloc[40:60]

# Save the modified DataFrame back to a CSV file
df.to_csv('recovery_modified4.csv', index=False)

s = "hvbid"
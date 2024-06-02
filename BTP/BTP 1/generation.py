import pandas as pd

def generate_tweets_from_article(article_text):
    prompt = f'''Generate 12 tweets of 100 characters using the given news article:
{article_text}'''
    prompt_template=f'''SYSTEM: You are a creative tweet writer. Always answer as deceptively.

USER: {prompt}

ASSISTANT:
'''
    response = lcpp_llm(prompt=prompt_template, max_tokens=4096, temperature=0.5, top_p=0.95,
                        repeat_penalty=1.2, top_k=150,
                        echo=True)
    
    return response["choices"][0]["text"]

# Load the CSV file into a DataFrame
df = pd.read_csv('your_file.csv')  # Replace 'your_file.csv' with the actual file path

# Specify the column containing the "text" to be used in the prompt
text_column_name = 'your_text_column'  # Replace 'your_text_column' with the actual column name

# Create a list to store the generated tweets
generated_tweets = []

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    article_text = row[text_column_name]
    generated_tweet = generate_tweets_from_article(article_text)
    
    # Split the generated response into lines
    lines = generated_tweet.split('\n')
    
    # Find the index of the first line after "ASSISTANT:"
    assistant_line_index = lines.index("ASSISTANT:") + 1
    
    # Add the first 5 lines after "ASSISTANT:"
    generated_tweets.extend(lines[assistant_line_index:assistant_line_index+5])

    generated_tweets.append("\n")

# Save the generated tweets to a text file
with open('generated_tweets.txt', 'w') as file:
    for tweet in generated_tweets:
        file.write(tweet + '\n')

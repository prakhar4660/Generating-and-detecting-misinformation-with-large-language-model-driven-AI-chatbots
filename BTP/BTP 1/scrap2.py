import requests
import pandas as pd
import time
import sys

# Backup the original stdout
original_stdout = sys.stdout 

df = pd.read_csv('coaid_train.csv', encoding='ISO-8859-1')
# df = df['id']
df['id'] = df['id'].astype(str)
df = df.sample(100)
# print(len(df))
ids = list(df['id'])
url = "https://cdn.syndication.twimg.com/tweet-result"
cnt = 0

with open('scrape.txt', 'w') as f:
    sys.stdout = f  # Redirect stdout to the file
    for x in ids:
        # print(type(x))
        # x = "1546621144358391808"
        querystring = {"id": x,"lang":"en", "token" : "vsbva"}
        payload = ""
        headers = {
            
        }
        response = requests.request("GET", url, data=payload, headers=headers, params=querystring)
        print(response.status_code)
        if(response.status_code == 200):
            cnt += 1
            result = response.json()
            print(result["text"])
        # break
        time.sleep(3)

    print(f'''Number of hits: {cnt} / 100''')

sys.stdout = original_stdout  # Restore the original stdout
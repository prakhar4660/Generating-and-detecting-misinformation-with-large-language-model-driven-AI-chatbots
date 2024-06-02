import snscrape.modules.twitter as sntwitter
import pandas as pd


# Created a list to append all tweet attributes(data)
attributes_container = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:john').get_items()):
    if i>5:
        break
    attributes_container.append([tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
    
# Creating a dataframe from the tweets list above 
tweets_df = pd.DataFrame(attributes_container, columns=["Date Created", "Number of Likes", "Source of Tweet", "Tweets"])

# query = "(from:elonmusk) until:2020-01-01 since:2010-01-01"
# query = "covid-19"
# tweets = []
# limit = 5


# for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
#     print(vars(tweet))
#     break
#     if len(tweets) == limit:
#         break
#     else:
#         tweets.append([tweet.date, tweet.username, tweet.content])
        
# df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
# print(df)

# to save to csv
# df.to_csv('tweets.csv')
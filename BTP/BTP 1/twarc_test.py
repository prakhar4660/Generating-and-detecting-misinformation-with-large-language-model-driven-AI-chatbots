from twarc import Twarc

t = Twarc(consumer_key='fGMTpA29tXejhRrZW0rFFxQRE',
                consumer_secret='QxCYZKPoPauPjZPDKFIQCEn2Z4cN81iJMcEXW9vWyd1cOFzujl',
                access_token='952825243474657281-x2S2mSR8WZOBig6ChRDuBcPEilpU5IV',
                access_token_secret='7KFplOtJYraBZFCfDT9jCuc6KiI9yS6MQbYmK7QCdx6me')

# usernames = ['mufaddal_vohra']#['narendramodi', 'OnePlus_IN', 'imVkohli']

# users = []

# for user in t.user_lookup(usernames):
#     users.append(user)

tweet_ids = ['1657057272395358208', '165691237941287731', '1656941001951559680']

tweets = []

for tweet in t.hydrate(tweet_ids):
    tweets.append(tweet)
# Importing important libraries
import pandas as pd
import numpy as np
import re
import pickle
import tweepy
import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
import tkinter as tk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tkinter import filedialog, messagebox, ttk
from tkinter import *

# Twitter Credentials
consumer_key = '78F6GWmlPoJX4CtW8E5A4dQYf'
consumer_secret = 'Uj3ZCkYe47HOLG0OOyXUly3wyvwhFAG8GLuQqZHVqse6VipfwJ'
access_token = '1310463220281495552-D8AOegt4AcMXgiC738DgD6STjQbaVi'
access_token_secret = 'PrcCzW7N4LmEcLIKnxv7VcX8ytwwCsAIdal2EWXROhKAh'


#Authenticate with credentials
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)



# key_word = 'Tesla'
limit = 50

def clean_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    clean_list = []
    for token, tag in pos_tag(tokens):
        token = token.lower()
        token = re.sub(r'@[a-z0-9_]\S+', '', token)
        token = re.sub(r'#[a-z0-9_]\S+', '', token)
        token = re.sub(r'&[a-z0-9_]\S+', '', token)
        token = re.sub(r'[?!.+,;$£%&"]+', '', token)
        token = re.sub(r'rt[\s]+', '', token)
        token = re.sub(r'\d+', '', token)
        token = re.sub(r'\$', '', token)
        token = re.sub(r'rt+', '', token)
        token = re.sub(r'https?:?\/\/\S+', '', token)
        if tag.startswith('NN'):
            position = 'n'
        elif tag.startswith('VB'):
            position = 'v'
        elif tag.startswith('RB'):
            position = 'r'
        elif tag.startswith('JJ'):
            position = 'a'
        else:
            position = 'n'

        clean_list.append(lemmatizer.lemmatize(token, pos = position))
        clean_list = [i for i in clean_list if i not in stop_words and len(i) > 0 and i != ':']

    return clean_list


# This method return vader Sentiment
def vader_compound_score(tweet):
    vader = SentimentIntensityAnalyzer()
    if vader.polarity_scores(tweet)['compound'] >= 0.05:
        return 'Positive'
    elif vader.polarity_scores(tweet)['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# This method returns textBlob Sentiment
def textblob_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# This method returns random forest Sentiment
def rf_sentiment(tweet):
    classifier = pickle.load(open('rf_classifier', 'rb'))
    vectorizer = pickle.load(open('rf_vectorizer', 'rb'))

    return classifier.predict(vectorizer.transform(list(tweet)))


# This method uses tweepy for search for tweets and store the tweets into a dataframe and returns the dataframe
def tweet_search(key_word):
    i = 0
    tweets_df = pd.DataFrame(columns=['id','Datetime', 'Tweet', 'Original tweet'])
    for tweet in tweepy.Cursor(api.search, q = key_word, count = 100, lang = 'en', tweet_mode = 'extended').items():
        print('Tweets downloaded:', i, '/', limit, end = '\r')
        if tweet.user.followers_count > 1000:
            tweet_tokenizer = TweetTokenizer()
            tweets_df = tweets_df.append({'id': tweet.id,
                                        'Datetime': tweet.created_at,
                                          'Tweet': tweet_tokenizer.tokenize(tweet.full_text),
                                         'Original tweet': tweet.full_text}, ignore_index = True)
            i += 1
        if i >= limit:
            break
        else:
            pass

    tweets_df['Datetime'] = pd.to_datetime(tweets_df['Datetime'], format = '%Y.%m.%d %H:%M:%S')
    tweets_df.set_index('Datetime', inplace = True)
    tweets_df['CleanTweet'] = tweets_df['Tweet'].apply(clean_tokens)
    tweets_df['rf_sent'] = rf_sentiment(tweets_df['CleanTweet'])
    tweets_df['CleanTweet'] = [', '.join(map(str, token)) for token in tweets_df['CleanTweet']]
    tweets_df.drop_duplicates(subset = ['CleanTweet'], inplace = True)
    tweets_df['Vader_sent'] = tweets_df['CleanTweet'].apply(vader_compound_score)
    tweets_df['TextBlob_sent'] = tweets_df['CleanTweet'].apply(textblob_sentiment)
 

    return tweets_df





# ---------------------------- G U I Starts from here --------------------------------------
root = Tk()

root.geometry("1000x650") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0) # makes the root window fixed in size.


# Frame for TreeView
frame1 = tk.LabelFrame(root, text="Data")
frame1.place(height=400, width=1000)
# Frame for open file dialog
keyword_frame = tk.LabelFrame(root, text="Enter Hashtag")
keyword_frame.place(height=100, width=800, rely=0.65, relx=0)

# Buttons
button1 = tk.Button(keyword_frame, text="Enter", command=lambda: show_tweets())
button1.place(rely=0.60, relx=0.50)

entry_keyword = StringVar()
e1 = tk.Entry(keyword_frame, textvariable = entry_keyword )
e1.place(rely = 0.65, relx= 0.25)




## Treeview Widget
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget

def selectItem(a):
    curItem = tv1.focus()
    tweet_id = tv1.item(curItem)['values'][0]
    print(tweet_id)
    status = api.get_status(tweet_id)
    # retweeted = status.retweets
    """print(f"Author: {status.author.screen_name}")
    print(f"Location: {status.author.location}")
    print(f"Followers count: {status.author.followers_count}")
    print(f"Friends count: {status.author.friends_count}")
    print(f"Account created at: {status.author.created_at}")
    print(f"Author: {status.author.screen_name}")
    print(f"Number of likes on tweet: {status.author.favorites_count}")
    print(f"Author: {status.author.screen_name}")"""
    #retweets_list = api.retweets(tweet_id)
    #for retweet in retweets_list:
    #    print(retweet.user)
    #print(type(status[0]))
    #print(status)
    #print(f"Retweet status: {status.retweets}")



tv1.bind('<ButtonRelease-1>', selectItem)


def clear_data():
    tv1.delete(*tv1.get_children())
    return None


def show_tweets():
    clear_data()
    # Storing tweets in tweets_df
    tweets_df = tweet_search(entry_keyword.get())

    #Dropping the columns we dont need to show
    newTweets_df = tweets_df.drop(['Tweet','Original tweet'], axis=1)
    
    tv1["column"] = list(newTweets_df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name
    df_rows = newTweets_df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert





root.mainloop()
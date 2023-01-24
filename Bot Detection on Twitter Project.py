#Collecting Tweets

import pandas as pd
import tweepy

def printtweetdata(n, ith_tweet):
    print()
    print(f"Tweet {n}:")
    print(f"Username:{ith_tweet[0]}")
    print(f"ID:{ith_tweet[1]}")
    print(f"Display Name:{ith_tweet[2]}")
    print(f"Date Account Created:{ith_tweet[3]}")
    print(f"Following Count:{ith_tweet[4]}")
    print(f"Follower Count:{ith_tweet[5]}")
    print(f"Total Tweets:{ith_tweet[6]}")
    print(f"Retweet Count:{ith_tweet[7]}")
    print(f"Bio:{ith_tweet[8]}")
    print(f"Geo Enabled:{ith_tweet[9]}")
    print(f"Verified:{ith_tweet[10]}")
    print(f"Favourites Count:{ith_tweet[11]}")
    print(f"Tweet Text:{ith_tweet[12]}")
    print(f"Hashtags Used:{ith_tweet[13]}")
    
def scrape(words, date_since, numtweet):
    db = pd.DataFrame(columns=['username','id','displayname', 'datecreated','following','followers','totaltweets',
                               'retweetcount', 'bio', 'geoenabled','verified', 'favourites','text','hashtags'])
    
    tweets = tweepy.Cursor(api.search, words, lang="en", since_id=date_since, tweet_mode='extended').items(numtweet)
    
    list_tweets = [tweet for tweet in tweets]
    
    i = 1
    
    for tweet in list_tweets:
        username = tweet.user.screen_name
        id = tweet.user.id_str
        displayname = tweet.user.name 
        datecreated = tweet.user.created_at
        following = tweet.user.friends_count
        followers = tweet.user.followers_count
        totaltweets = tweet.user.statuses_count
        retweetcount = tweet.retweet_count
        bio = tweet.user.description
        geoenabled = tweet.user.geo_enabled
        verified = tweet.user.verified
        favourites = tweet.user.favourites_count
        hashtags = tweet.entities['hashtags']
        
        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])
        
        ith_tweet = [username, id, displayname, datecreated, following, followers, totaltweets, retweetcount, 
                     bio, geoenabled, verified, favourites, text, hashtext]
        db.loc[len(db)] = ith_tweet
        
        printtweetdata(i, ith_tweet)
        i = i+1
        
    filename = 'Data Tweets3.csv'
    db.to_csv(filename)
    
if __name__ == '__main__':
    
    consumer_key = "XXXXXXXXXXX"
    consumer_secret = "XXXXXXXXXXX"
    access_key = "XXXXXXXXXXX"
    access_secret = "XXXXXXXXXXX"
    #access tokens have to be kept private in accordance to Twitter Developer rules
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    print("Enter Twitter HashTag to search for")
    words = input()
    print("Enter Date since The Tweets are required in yyyy-mm--dd")
    date_since = input()
    
    numtweet = 2000
    scrape(words, date_since, numtweet)
    print('Scraping has completed!')
 
 col_list = ["username", "id", "displayname", "datecreated", "following", "followers", "totaltweets", "retweetcount", 
            "bio", "geoenabled", "verified", "favourites", "text", "hashtags"]
df = pd.read_csv("Data Tweets3.csv", usecols=col_list)
df = df.drop_duplicates(subset=['username'], keep='first')

import botometer

rapidapi_key = "XXXXXXXXXXX"

twitter_app_auth = {
    'consumer_key': 'XXXXXXXXXXX',
    'consumer_secret': 'XXXXXXXXXXX',
    'access_token': 'XXXXXXXXXXX',
    'access_token_secret': 'XXXXXXXXXXX',
  }

bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)

import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

for screen_name, result in bom.check_accounts_in(df["username"]):
    
    api_data = result
    
    keys_to_remove = ['display_scores']
    for key in keys_to_remove:
        api_data.pop(key, None)
    
    api_data2 = flatten(api_data)
    api_data3 = {key: api_data2[key] for key in api_data2 if key != 'cap_universal'}
    api_data4 = {key: api_data3[key] for key in api_data3 if key != 'user_majority_lang'}
    api_data5 = {key: api_data4[key] for key in api_data4 if key != 'user_user_data_id_str'}
    api_data6 = {key: api_data5[key] for key in api_data5 if key != 'raw_scores_english_astroturf'}
    api_data7 = {key: api_data6[key] for key in api_data6 if key != 'raw_scores_english_fake_follower'}
    api_data8 = {key: api_data7[key] for key in api_data7 if key != 'raw_scores_english_financial'}
    api_data9 = {key: api_data8[key] for key in api_data8 if key != 'raw_scores_english_other'}
    api_data10 = {key: api_data9[key] for key in api_data9 if key != 'raw_scores_english_self_declared'}
    api_data11 = {key: api_data10[key] for key in api_data10 if key != 'raw_scores_universal_astroturf'}
    api_data12 = {key: api_data11[key] for key in api_data11 if key != 'raw_scores_universal_fake_follower'}
    api_data13 = {key: api_data12[key] for key in api_data12 if key != 'raw_scores_universal_financial'}
    api_data14 = {key: api_data13[key] for key in api_data13 if key != 'raw_scores_universal_other'}
    api_data15 = {key: api_data14[key] for key in api_data14 if key != 'raw_scores_universal_overall'}
    api_data16 = {key: api_data15[key] for key in api_data15 if key != 'raw_scores_universal_self_declared'}
    api_data17 = {key: api_data16[key] for key in api_data16 if key != 'raw_scores_universal_spammer'}
    api_data18 = {key: api_data17[key] for key in api_data17 if key != 'raw_scores_english_spammer'}

    df2 = pd.DataFrame([api_data18])
    
    df3 = df2.rename(columns={"cap_english": "Cap_Score", "raw_scores_english_overall": "Raw_Scores", "user_user_data_screen_name": "username"})
        
    print(df3)

df4 = pd.read_csv('Data Tweets3.csv')
df4 = df4.drop(['username.1'], axis=1)
print(df4)

df4['Cap_Score'] = pd.to_numeric(df4['Cap_Score'])
df4['Raw_Scores'] = pd.to_numeric(df4['Raw_Scores'])

percent = df4['Cap_Score'] * 100

df4['PercentAbove'] = percent
df4['PercentAbove'] = (round(df4['PercentAbove'], 2))
print(df4)

false = 100 - df4['PercentAbove']

df4['FalsePositive'] = false
print(df4)

result2 = df4['displayname'].str.contains("bot", na=False, case=False)
result3 = df4['username'].str.contains("bot", na=False, case=False)
result4 = df4['bio'].str.contains(" bot ", na=False, case=False)

results = pd.concat([result2, result3, result4], axis=1)
print(results)

results["Combined"] = results[['displayname', 'username', 'bio']].any(axis='columns')

print(results)

results = results.drop(['displayname', 'username', 'bio'], axis=1)
results = results.rename(columns={"Combined": "SelfDeclared"})
print(results)

df4 = pd.concat([df4, results], axis=1)
print(df4)

def f(row):
    if row['PercentAbove'] >= 85:
        val = 1
    elif row['SelfDeclared'] == True:
        val = 1
    else:
        val = 0
    return val

df4['BotLabel'] = df4.apply(f, axis=1)

print(df4)

df5 = pd.read_csv('Finished_Dataset.csv')
df5 = df5.drop(['Unnamed: 0'], axis=1)
df6 = pd.concat([df5, df4], axis=0)
print(df6)

df6.to_csv('/Users/Danielle/Desktop/Final Dataset.csv', encoding='utf-8-sig')

#Visualisations

import pandas as pd
df = pd.read_csv('Finished Dataset.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop_duplicates(subset=['username'], keep='first')
df.to_csv('/Users/Danielle/Desktop/Finished Dataset.csv', encoding='utf-8-sig')

from matplotlib import pyplot as plt
df['BotLabel'].value_counts().plot.bar()
plt.title('No. of Human and Bot Accounts' )
plt.xlabel('Account Type')
plt.ylabel('No. of Accounts')

min(df.datecreated)
max(df.datecreated)

df['datecreated'] =  pd.to_datetime(df['datecreated'], format='%d/%m/%Y %H:%M')

import numpy as np

year = df['datecreated'].dt.year

Bots = df[df.BotLabel != 0]
year = Bots['datecreated'].dt.year

bins = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]

plt.hist(year, bins = bins,edgecolor ='black')
plt.xticks(np.arange(min(year), max(year)+1, 1.0))
plt.xticks(rotation = 45)
plt.title('Year Bot Account Created')
plt.xlabel('Year')
plt.ylabel('No. of Accounts')

Humans = df[df.BotLabel != 1]

year = Humans['datecreated'].dt.year

bins = [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]

plt.hist(year, bins = bins,edgecolor ='black')
plt.xticks(np.arange(min(year), max(year)+1, 1.0))
plt.xticks(rotation = 45)
plt.title('Year Human Account Created')
plt.xlabel('Year')
plt.ylabel('No. of Accounts')

Bots['SelfDeclared'].value_counts().plot.bar()
plt.title('No. of Self Declared Bots vs Non Declared' )
plt.xlabel('Declared')
plt.ylabel('No. of Accounts')

min(df.totaltweets)
max(df.totaltweets) plt.xticks(np.arange(min(year), max(year)+1, 1.0))

bins = [0,50000,100000,150000,200000,250000,300000,350000,400000,450000,500000,550000]

plt.hist(Humans.totaltweets, bins = bins,edgecolor ='black')
plt.xticks(bins)
plt.xticks(rotation = 45)
plt.title('Human Total Number of Tweets')
plt.xlabel('No. of Tweets')
plt.ylabel('No. of Accounts')

bins = [0,50000,100000,150000,200000,250000,300000,350000,400000,450000,500000,550000]

plt.hist(Bots.totaltweets, bins = bins,edgecolor ='black')
plt.xticks(bins)
plt.xticks(rotation = 45)
plt.title('Bot Total Number of Tweets')
plt.xlabel('No. of Tweets')
plt.ylabel('No. of Accounts')

mylabels = ["False", "True"]

plt.pie(Humans['geoenabled'].value_counts(), labels = mylabels, autopct="%1.1f%%")
plt.title("Geo Enabled Human Accounts")
plt.show() 

mylabels = ["False", "True"]

plt.pie(Bots['geoenabled'].value_counts(), labels = mylabels, autopct="%1.1f%%")
plt.title("Geo Enabled Bot Accounts")
plt.show() 

max(Humans.retweetcount)
max(Bots.retweetcount)

from statistics import mean
mean(df.retweetcount)
mean(Humans.retweetcount)
mean(Bots.retweetcount)
min(Humans.retweetcount)
min(Bots.retweetcount)

df["verified"] = df["verified"].astype(int)
df["SelfDeclared"] = df["SelfDeclared"].astype(int)
df["geoenabled"] = df["geoenabled"].astype(int)

result = df['bio'].str.contains("")
result = result.fillna(False)
result = result.astype(int)
df = df.drop(['bio'], axis=1)
df = pd.concat([df, result], axis=1)
df["hashtags"] = df["hashtags"].apply(lambda n: len(n.split()))

new_cols = [col for col in df.columns if col != 'BotLabel'] + ['BotLabel']
df = df[new_cols]
print(df)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 9))
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

from scipy.stats import pearsonr
import numpy as np
rho = df.corr()
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
rho.round(2).astype(str) + p

#Decision Tree

import pandas as pd
df = pd.read_csv('Final Dataset.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['bio'], axis=1)
df = df.drop(['id'], axis=1)
df = df.drop(['username'], axis=1)
df = df.drop(['displayname'], axis=1)
df = df.drop(['text'], axis=1)
df = df.drop(['PercentAbove'], axis=1)
df = df.drop(['FalsePositive'], axis=1)


df["hashtags"] = df["hashtags"].apply(lambda n: len(n.split()))
df['datecreated'] =  pd.to_datetime(df['datecreated'], format='%d/%m/%Y %H:%M')
df['datecreated'] = pd.DatetimeIndex(df['datecreated']).year

X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

from sklearn.model_selection import train_test_split
X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train_Bot)
X_train_Bot = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test_Bot)
X_test_Bot = pd.DataFrame(x_test_scaled)

from sklearn.tree import DecisionTreeClassifier

tree_classifier = DecisionTreeClassifier(max_depth=2, random_state=0)
tree_classifier.fit(X_train_Bot, y_train_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(tree_classifier.score(X_test_Bot, y_test_Bot)))

pred_Bot = tree_classifier.predict(X_test_Bot)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, pred_Bot)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, pred_Bot))

depths = []
scores = []

for k in range(1, 10, 1):
    depths.append(k)
    tree_classifier = DecisionTreeClassifier(max_depth=k, random_state=0)
    tree_classifier.fit(X_train_Bot, y_train_Bot)
    score_new = tree_classifier.score(X_test_Bot,y_test_Bot)
    scores.append(score_new)
    
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 6))
plt.title('Accuracy for Depth Size of Decison Tree')
plt.plot(depths, scores)
plt.xlabel('Depth Size')
plt.ylabel('Accuracy')
plt.show()

data = [[0.95, 0.05, 94.8], [0.90, 0.10, 97.4], [0.85, 0.15, 97.7],[0.80, 0.20, 97.2],[0.75, 0.25, 97.1],[0.70, 0.30, 98.2],
       [0.65, 0.35, 98.3], [0.60, 0.40, 98.5],[0.55, 0.45,98.6],[0.50, 0.50, 98.3],[0.45, 0.55, 98.4],[0.40, 0.60, 98.5],
       [0.35, 0.75, 98.5], [0.30, 0.70, 98.9], [0.25, 0.75, 98.6], [0.20, 0.80, 98.6], [0.15, 0.85, 97.9],[0.10, 0.85, 96.3],
       [0.05, 0.95, 96.4]]
accuracy = pd.DataFrame(data, columns=['Training', 'Test','Accuracy'])
print(accuracy)

import numpy as np

plt.figure(figsize=(10, 6))
plt.title('Accuracy for Training Set Sizes')
plt.plot(accuracy.Training, accuracy.Accuracy)
plt.xticks(np.arange(min(accuracy.Training), max(accuracy.Training)+0.05, 0.05))
plt.xticks(rotation = 45)
plt.yticks(np.arange(55, 105, 5))
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.show()

from sklearn import tree

tree.plot_tree(tree_classifier);
fn=['datecreated','following','followers','totaltweets','retweetcount','geoenabled','verified','favourites','hashtags',
    'Cap_Score','Raw_Scores','SelfDeclared']
cn=['0', '1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(tree_classifier,
               feature_names = fn, 
               class_names=cn,
               filled = True);

df = df.drop(['Cap_Score'], axis=1)
df = df.drop(['SelfDeclared'], axis=1)

X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

from sklearn.model_selection import train_test_split
X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train_Bot)
X_train_Bot = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test_Bot)
X_test_Bot = pd.DataFrame(x_test_scaled)

from sklearn.tree import DecisionTreeClassifier

tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=0)
tree_classifier.fit(X_train_Bot, y_train_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(tree_classifier.score(X_test_Bot, y_test_Bot)))

pred_Bot = tree_classifier.predict(X_test_Bot)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, pred_Bot)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, pred_Bot))

depths = []
scores = []

for k in range(1, 10, 1):
    depths.append(k)
    tree_classifier = DecisionTreeClassifier(max_depth=k, random_state=0)
    tree_classifier.fit(X_train_Bot, y_train_Bot)
    score_new = tree_classifier.score(X_test_Bot,y_test_Bot)
    scores.append(score_new)

df5 = df.drop("Raw_Scores",axis=1)

X = df5.drop("BotLabel",axis=1)   
y = df5["BotLabel"] 

X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train_Bot)
X_train_Bot = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test_Bot)
X_test_Bot = pd.DataFrame(x_test_scaled)

from sklearn.tree import DecisionTreeClassifier

tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=0)
tree_classifier.fit(X_train_Bot, y_train_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(tree_classifier.score(X_test_Bot, y_test_Bot)))

pred_Bot = tree_classifier.predict(X_test_Bot)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, pred_Bot)
print("\nConfusion matrix:\n{}".format(confusion))


df = pd.read_csv('Final Dataset.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['bio'], axis=1)
df = df.drop(['id'], axis=1)
df = df.drop(['username'], axis=1)
df = df.drop(['displayname'], axis=1)
df = df.drop(['text'], axis=1)
df = df.drop(['PercentAbove'], axis=1)
df = df.drop(['FalsePositive'], axis=1)
df = df.drop(['Cap_Score'], axis=1)
df = df.drop(['Raw_Scores'], axis=1)
df = df.drop(['SelfDeclared'], axis=1)
df = df.drop(['PercentAbove'], axis=1)
df = df.drop(['FalsePositive'], axis=1)
df = df.drop(['verified'], axis=1)
df = df.drop(['followers'], axis=1)

X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

BotSample = RandomUnderSampler(random_state=42, replacement=True)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train_Bot)
X_train_Bot = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test_Bot)
X_test_Bot = pd.DataFrame(x_test_scaled)

from sklearn.tree import DecisionTreeClassifier

tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=0)
tree_classifier.fit(X_train_Bot, y_train_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(tree_classifier.score(X_test_Bot, y_test_Bot)))

pred_Bot = tree_classifier.predict(X_test_Bot)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, pred_Bot)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, pred_Bot))

depths = []
scores = []

for k in range(1, 10, 1):
    depths.append(k)
    tree_classifier = DecisionTreeClassifier(max_depth=k, random_state=0)
    tree_classifier.fit(X_train_Bot, y_train_Bot)
    score_new = tree_classifier.score(X_test_Bot,y_test_Bot)
    scores.append(score_new)
    
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 6))
plt.title('Accuracy for Depth Size of Decison Tree')
plt.plot(depths, scores)
plt.xlabel('Depth Size')
plt.ylabel('Accuracy')
plt.show()


X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

BotSample = RandomOverSampler(random_state=42)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train_Bot)
X_train_Bot = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test_Bot)
X_test_Bot = pd.DataFrame(x_test_scaled)

from sklearn.tree import DecisionTreeClassifier

tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=0)
tree_classifier.fit(X_train_Bot, y_train_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(tree_classifier.score(X_test_Bot, y_test_Bot)))

pred_Bot = tree_classifier.predict(X_test_Bot)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, pred_Bot)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, pred_Bot))

depths = []
scores = []

for k in range(1, 10, 1):
    depths.append(k)
    tree_classifier = DecisionTreeClassifier(max_depth=k, random_state=0)
    tree_classifier.fit(X_train_Bot, y_train_Bot)
    score_new = tree_classifier.score(X_test_Bot,y_test_Bot)
    scores.append(score_new)
    
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 6))
plt.title('Accuracy for Depth Size of Decison Tree')
plt.plot(depths, scores)
plt.xlabel('Depth Size')
plt.ylabel('Accuracy')
plt.show()



X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

BotSample = RandomOverSampler(random_state=42)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train_Bot)
X_train_Bot = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test_Bot)
X_test_Bot = pd.DataFrame(x_test_scaled)

from sklearn.tree import DecisionTreeClassifier

tree_classifier = DecisionTreeClassifier(max_depth=9, random_state=0)
tree_classifier.fit(X_train_Bot, y_train_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(tree_classifier.score(X_test_Bot, y_test_Bot)))

pred_Bot = tree_classifier.predict(X_test_Bot)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, pred_Bot)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, pred_Bot))

#Random Forest

import pandas as pd
df = pd.read_csv('Final Dataset.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['bio'], axis=1)
df = df.drop(['id'], axis=1)
df = df.drop(['username'], axis=1)
df = df.drop(['displayname'], axis=1)
df = df.drop(['text'], axis=1)
df = df.drop(['PercentAbove'], axis=1)
df = df.drop(['FalsePositive'], axis=1)
df = df.drop(['Cap_Score'], axis=1)
df = df.drop(['SelfDeclared'], axis=1)
df = df.drop(['Raw_Scores'], axis=1)
df["hashtags"] = df["hashtags"].apply(lambda n: len(n.split()))
df['datecreated'] =  pd.to_datetime(df['datecreated'], format='%d/%m/%Y %H:%M')
df['datecreated'] = pd.DatetimeIndex(df['datecreated']).year
df = df.drop(['followers'], axis=1)
df = df.drop(['verified'], axis=1)

X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

from sklearn.model_selection import train_test_split
X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train_Bot)
X_train_Bot = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test_Bot)
X_test_Bot = pd.DataFrame(x_test_scaled)

from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
forest_classifier.fit(X_train_Bot, y_train_Bot)

y_pred = forest_classifier.predict(X_test_Bot)
print("Accuracy of decision tree on test set: {:.3f}".format(forest_classifier.score(X_test_Bot, y_test_Bot)))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, y_pred)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, y_pred))

depths = []
scores = []

for k in range(1, 10, 1):
    depths.append(k)
    forest_classifier = RandomForestClassifier(max_depth=k, random_state=0)
    forest_classifier.fit(X_train_Bot, y_train_Bot)
    score_new = forest_classifier.score(X_test_Bot,y_test_Bot)
    scores.append(score_new)


X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler 

BotSample = RandomUnderSampler(random_state=42, replacement=True)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
forest_classifier.fit(X_train_Bot, y_train_Bot)

y_pred = forest_classifier.predict(X_test_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(forest_classifier.score(X_test_Bot, y_test_Bot)))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, y_pred)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, y_pred))

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

param_grid = { 
    'n_estimators': [10,20,30,40,50,60,70,80,90,100],
    'max_depth' : [2,3,4,5,6,7,8,9,10],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],
    'min_samples_split': [2,3,4,5,6,7,8,9,10],
    'max_features': ['auto', 'sqrt']}

CV_rfc = GridSearchCV(estimator=forest_classifier, param_grid=param_grid)
CV_rfc.fit(X_train_Bot, y_train_Bot)

print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_train_pred = forest_classifier.predict(X_train_Bot)
y_test_pred = forest_classifier.predict(X_test_Bot)


X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler 

BotSample = RandomUnderSampler(random_state=42, replacement=True)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators=70, max_depth=7, random_state=0)
forest_classifier.fit(X_train_Bot, y_train_Bot)

y_pred = forest_classifier.predict(X_test_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(forest_classifier.score(X_test_Bot, y_test_Bot)))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, y_pred)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, y_pred))

classes = ['Human','Bot']

def plot_confusionmatrix(y_train_pred,y_train_Bot,dom):
    print(f'{dom} Confusion matrix')
    cf = confusion_matrix(y_train_pred,y_train_Bot)
    sns.heatmap(cf,annot=True,yticklabels=classes
               ,xticklabels=classes,cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()

print(f'Train score {accuracy_score(y_train_pred,y_train_Bot)}')
print(f'Test score {accuracy_score(y_test_pred,y_test_Bot)}')
plot_confusionmatrix(y_train_pred,y_train_Bot,dom='Train')
plot_confusionmatrix(y_test_pred,y_test_Bot,dom='Test')

X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

from sklearn.model_selection import train_test_split

import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler 

BotSample = RandomOverSampler(random_state=42)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
forest_classifier.fit(X_train_Bot, y_train_Bot)

y_pred = forest_classifier.predict(X_test_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(forest_classifier.score(X_test_Bot, y_test_Bot)))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, y_pred)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, y_pred))

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

param_grid = { 
    'n_estimators': [10,20,30,40,50,60,70,80,90,100],
    'max_depth' : [2,3,4,5,6,7,8,9,10],
  }

CV_rfc = GridSearchCV(estimator=forest_classifier, param_grid=param_grid)
CV_rfc.fit(X_train_Bot, y_train_Bot)

print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

X = df.drop("BotLabel",axis=1)   
y = df["BotLabel"] 

from sklearn.model_selection import train_test_split

import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler 

BotSample = RandomOverSampler(random_state=42)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train_Bot, X_test_Bot, y_train_Bot, y_test_Bot = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier

forest_classifier = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=0)
forest_classifier.fit(X_train_Bot, y_train_Bot)

y_pred = forest_classifier.predict(X_test_Bot)

print("Accuracy of decision tree on test set: {:.3f}".format(forest_classifier.score(X_test_Bot, y_test_Bot)))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test_Bot, y_pred)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test_Bot, y_pred))

#Bag of Words

import pandas as pd
df = pd.read_csv('Final Dataset.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.drop(['id'], axis=1)
df = df.drop(['username'], axis=1)
df = df.drop(['displayname'], axis=1)
df = df.drop(['PercentAbove'], axis=1)
df = df.drop(['FalsePositive'], axis=1)
df = df.drop(['Cap_Score'], axis=1)
df = df.drop(['SelfDeclared'], axis=1)
df = df.drop(['Raw_Scores'], axis=1)
df = df.drop(['hashtags'], axis=1)
df = df.drop(['datecreated'], axis=1)
df = df.drop(['followers'], axis=1)
df = df.drop(['verified'], axis=1)
df = df.drop(['following'], axis=1)
df = df.drop(['totaltweets'], axis=1)
df = df.drop(['retweetcount'], axis=1)
df = df.drop(['geoenabled'], axis=1)
df = df.drop(['favourites'], axis=1)
df = df.drop(['bio'], axis=1)
print(df)

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
 
vectorizer = CountVectorizer()

arr = df.text.to_numpy()
flat_array = arr.flatten()
flat_array = [str (item) for item in flat_array]

docs = flat_array
bag = vectorizer.fit_transform(docs)

X = bag.toarray()
y = df.BotLabel.to_numpy()
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')

lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)

print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_test, y_predict))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_predict)
print("\nConfusion matrix:\n{}".format(confusion))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

X = bag.toarray()
y = df.BotLabel.to_numpy()

import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler 

BotSample = RandomUnderSampler(random_state=42, replacement=True)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')

lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)
  
print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_test, y_predict))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_predict)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

X = bag.toarray()
y = df.BotLabel.to_numpy()

from sklearn.model_selection import train_test_split

import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler 

BotSample = RandomOverSampler(random_state=42)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')

lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)
  
print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_test, y_predict))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_predict)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

import seaborn as sns

from sklearn.metrics import accuracy_score,confusion_matrix
from matplotlib import pyplot as plt

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

classes = ['Human','Bot']

def plot_confusionmatrix(y_train_pred,y_train,dom):
    print(f'{dom} Confusion matrix')
    cf = confusion_matrix(y_train_pred,y_train)
    sns.heatmap(cf,annot=True,yticklabels=classes
               ,xticklabels=classes,cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')
plot_confusionmatrix(y_train_pred,y_train,dom='Train')
plot_confusionmatrix(y_test_pred,y_test,dom='Test')

from sklearn.model_selection import GridSearchCV

solvers = ['sag', 'lbfgs']
penalty = ['l2']
c_values = [1,20,40,60,80,100]
max_iter = [100,200,300,400,500]
multi_class = ['ovr']

grid = dict(solver=solvers,penalty=penalty,C=c_values, max_iter = max_iter, multi_class = multi_class)
grid_search = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

X = bag.toarray()
y = df.BotLabel.to_numpy()

from sklearn.model_selection import train_test_split

import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler 

BotSample = RandomOverSampler(random_state=42)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(x_test_scaled)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr = LogisticRegression(C=200, random_state=0, solver='lbfgs', multi_class='ovr', max_iter = 1000)

lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)
  
print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_test, y_predict))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_predict)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.title('ROC Curve for Over Sampled Bag of Words Model')
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#ROC Curves

from sklearn import metrics
from matplotlib import pyplot as plt

y_pred_proba = forest_classifier.predict_proba(X_test_Bot)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test_Bot,  y_pred_proba)
auc = metrics.roc_auc_score(y_test_Bot, y_pred_proba)

plt.title('ROC Curve for Over Sampled Random Forest Model')
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

from sklearn import metrics
from matplotlib import pyplot as plt

y_pred_proba = tree_classifier.predict_proba(X_test_Bot)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test_Bot,  y_pred_proba)
auc = metrics.roc_auc_score(y_test_Bot, y_pred_proba)

#create ROC curve
plt.title('ROC Curve for Over Sampled Decision Tree Model')
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#Tweet Sentiments

df2= df[df.BotLabel == 1]
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df2['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df2['text']]
df2['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df2['text']]
df2['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df2['text']]
df2['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df2['text']]
print(df2)

def f(row):
    if row['compound'] >= 0.05:
        val = "positive"
    elif row['compound'] <= -0.05:
        val = "negative"
    else:
        val = "neutral"
    return val

df2['Sentiment'] = df2.apply(f, axis=1)

print(df2)

df2 = df2.drop(['compound'], axis=1)

vectorizer = CountVectorizer()

arr = df2.text.to_numpy()
flat_array = arr.flatten()
flat_array = [str (item) for item in flat_array]
docs = flat_array
bag = vectorizer.fit_transform(docs)

X = bag.toarray()
y = df2.Sentiment.to_numpy()

import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler 

BotSample = RandomOverSampler(random_state=42)
x_sample, y_sample = BotSample.fit_resample(X, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_sample))

X = x_sample   
y = y_sample 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(x_test_scaled)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr = LogisticRegression(C=200, random_state=0, solver='lbfgs', multi_class='ovr', max_iter = 1000)

lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)
  
print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_test, y_predict))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_predict)
print("\nConfusion matrix:\n{}".format(confusion))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


df3= df[df.BotLabel == 0]
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df3['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df3['text']]
df3['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df3['text']]
df3['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df3['text']]
df3['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df3['text']]
print(df3)

def f(row):
    if row['compound'] >= 0.05:
        val = "positive"
    elif row['compound'] <= -0.05:
        val = "negative"
    else:
        val = "neutral"
    return val

df3['Sentiment'] = df3.apply(f, axis=1)

df3 = df3.drop(['compound'], axis=1)

df2[df2['Sentiment'].str.contains("negative")]
101/313*100

df3[df3['Sentiment'].str.contains("negative")]
1701/4280 * 100

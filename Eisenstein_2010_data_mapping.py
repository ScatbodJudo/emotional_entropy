import pandas
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math
import datetime

#global variables
def printProgress(nTokens):
    currentToken+=1
    #start_time = datetime.now()
    ratioComplete = currentToken/nTokens
    if ratioComplete%10 == 0: 
        print("Percent Complete: "+ str(ratioComplete*100))

#calculate number of unique users
def getUniqueUsers(dataFrame):
    users = {
        "user_id":[]
    }
    for user in dataFrame['user_id']:
        if user not in users['user_id']:
            users['user_id'].append(user)
    print(len(users['user_id']))



def calcEntropy(tokens, tokens_pdf):
    h = []
    for token in tokens:
        pi = tokens_pdf[token]
        h.append(pi*math.log(1/pi,2))
        return sum(h)
    
    
def calcEmoH(dataFrame):
    unique_tokens = []
    tokens_pdf = {
    }
    n_tokens = 0
    current_token = 0
    for token_list in dataFrame.tokens:
        n = len(token_list)
        n_tokens+=n

    print("Calculating Probability Density...")
    for array in dataFrame.tokens:
        for token in array:
            current_token+=1
            if (current_token/n_tokens)%5==0: print("Progress: "+ str(current_token/n_tokens*100))
            if token not in unique_tokens: 
                unique_tokens.append(token)
                tokens_pdf[token] = 1
            else: tokens_pdf[token] += 1
                
    for item in tokens_pdf.keys():
        tokens_pdf[item] /= n_tokens

    print("Number of unique tokens: "+ str(len(unique_tokens)))
    print("Total number of tokens: "+ str(n_tokens))

    df_tokens_pdf = pandas.DataFrame.from_dict(tokens_pdf)
    df_tokens_pdf.to_csv("Tokens_Probability_Density.csv")

    print("Calculating H Entropy...")
    dataFrame["H_Entropy"] = dataFrame['tokens'].apply(lambda x: calcEntropy(x, tokens_pdf))
    print("Calculating emotional entropy...")
    dataFrame['emo_h'] = dataFrame.H_Entropy*dataFrame.compound_sent

    #save data to csv
    print("Exporting data...")
    dataFrame.to_csv('Eisenstein_2010_Sentiment.csv')

def analyzeData(filePath):
    #import tab delimited 2010 data and enter into Pandas dataframe
    #"Eisenstein_2010_tweets.txt"
    print("importing data from "+filePath)
    file_path = filePath
    df_tweets = pandas.read_csv(file_path, sep='\t', encoding = 'ISO-8859-1', names = ['user_id','datetime','location','lat','lon','text'])
    print("Loaded dataframe with shape: "+ str(df_tweets.shape))

    #tokenize tweet text using nltk .tokenize function
    print("Tokenizing tweets...")
    df_tweets['tokens'] = df_tweets['text'].fillna("").apply(lambda x: nltk.word_tokenize(x))

    #get sentiment polarity score using VADER
    print("Analyzing sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    df_tweets['sentiment'] = df_tweets['text'].fillna("").apply(lambda x: analyzer.polarity_scores(x))

    #assign compound score to independent column
    df_tweets['compound_sent'] = [i['compound'] for i in df_tweets.sentiment]

    calcEmoH(df_tweets)

analyzeData("Eisenstein_2010_tweets.txt") 


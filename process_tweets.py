import pandas as pd
import numpy as np
import os
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import preprocessor as p

def read_lexicon(filepath):
	lex = []
	f = open(filepath)
	lex = f.read().splitlines()
	return lex

def analyze_tweets(tweet_col, lexicon):
	scores_dict = {}
	analyzer = SentimentIntensityAnalyzer()
	for keyword in lexicon:
		scores_dict[keyword] = calc_score(tweet_col, keyword, analyzer)
	return scores_dict
		

def calc_score(tweet_col, word, analyzer):
	cursum = 0
	count = 0
	for tweet in tweet_col:
		if word.lower() in tweet.lower():
			scores = analyzer.polarity_scores(tweet)
			#print(scores)
			score = scores['compound']
			cursum += score
			count += 1
	if count == 0:
		return 0
	else:
		return cursum  / len(tweet_col)


def clean_tweet(tweet):
	return p.clean(tweet)	


def main():
	output_list = []
	lexicon = read_lexicon('lexicon.txt')
	folderpath  = os.path.join(os.getcwd(), sys.argv[1])
	for filename in os.listdir(folderpath):
		excel_path = './' + sys.argv[1] + '/' + filename
		sen_df = pd.read_excel(excel_path)
		senator_name = sen_df['Name'].iloc[0]
		senator_handle = sen_df['Screen Name'].iloc[0]
		tweet_col = sen_df['Text'].apply(clean_tweet)
		scores_dict = analyze_tweets(tweet_col, lexicon)
		scores_dict['Name'] = senator_name
		scores_dict['Handle'] = senator_handle
		output_list.append(scores_dict)
	for d in output_list:
		print(d)
	output_df = pd.DataFrame(output_list)
	output_df.to_csv('scores.csv')

if __name__ == '__main__':
    main()

	
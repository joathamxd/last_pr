import re
from pandas import read_csv, Series
from emoji import emoji_count
from text_unidecode import unidecode
# from wordcloud import WordCloud


class Process:
    TWEET_COL = "tweet_text"

    def __init__(self, data, directory="../"):
        self.data = data
        self.directory = directory

    def run(self, ):
        print("Starting process")
        self.data = self.process_before_clean()
        neg, pos = self.get_pos_neg_words()

        self.data[["count_personal_positive",
                   "count_personal_negative"]] = self.data["tweet_clean"].apply(
            lambda x: Series(self.count_personal_words(x, neg, pos)))

    def process_before_clean(self):
        print("Process previous clean text")
        self.data["tweet_mensaje"] = self.data[self.TWEET_COL].str.len()
        self.data["n_emojis"] = self.data[self.TWEET_COL].map(emoji_count)
        self.data["n_lower"] = self.data[self.TWEET_COL].map(lambda x: sum(map(str.islower, x)))
        self.data["n_upper"] = self.data[self.TWEET_COL].map(lambda x: sum(map(str.isupper, x)))
        self.data["n_digit"] = self.data[self.TWEET_COL].map(lambda x: sum(map(str.isdigit, x)))
        self.data["n_whitespaces"] = self.data[self.TWEET_COL].map(lambda x: len(re.findall('\s', x)))
        self.data["n_words"] = self.data[self.TWEET_COL].str.split(" ").str.len()
        self.data["has_tags"] = self.data[self.TWEET_COL].str.lower().str.contains("@").astype(int)
        self.data["has_hashtag"] = self.data[self.TWEET_COL].str.lower().str.contains("#").astype(int)
        self.data["has_urls"] = self.data[self.TWEET_COL].str.lower().str.contains("http").astype(int)
        self.data["n_exclamation"] = self.data[self.TWEET_COL].str.contains("!").astype(int)
        self.data["n_question"] = self.data[self.TWEET_COL].str.contains("!").astype(int)
        self.data["n_hashtag"] = self.data[self.TWEET_COL].map(lambda x: len(re.findall(r"#(\w+)", x)))
        self.data["n_tags"] = self.data[self.TWEET_COL].map(lambda x: len(re.findall(r"@(\w+)", x)))
        self.data["n_urls"] = self.data[self.TWEET_COL].map(lambda x: len(re.findall(r"http+", x)))
        self.data["tweet_clean"] = self.data[self.TWEET_COL].map(lambda x: self.clean_tweet(x))

        return self.data

    @staticmethod
    def clean_tweet(tweet):
        text = unidecode(tweet).lower()
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
        return text.lower()

    def get_pos_neg_words(self):
        print("Evaluating negative words")
        word_column = 'word'
        neg = read_csv(self.directory + "data/neg.csv", header=None)
        neg.columns = [word_column, ]
        pos = read_csv(self.directory + "data/pos.csv", header=None)
        pos.columns = [word_column, ]

        return set(neg[word_column]), set(pos[word_column])

    def get_personal_stop_words(self):
        with open(self.directory + 'data/stopwords.csv') as f:
            lines = f.read().splitlines()
        return set(lines)

    def generate_freq_words(self, sentiment):
        comment_words = ''
        freq_words = {}
        stopwords = self.get_personal_stop_words()
        data_sentiment = self.data.loc[self.data["sentiment"] == sentiment]
        for val in data_sentiment["tweet_clean"]:
            tokens = val.split(" ")
            comment_words += " ".join(tokens) + " "

        # wc = WordCloud(
        #     stopwords=stopwords).generate(comment_words)
        #
        # for k, v in wc.words_.items():
        #     if v > 0.1:
        #         freq_words[k] = v
        return freq_words

    @staticmethod
    def count_personal_words(tweet, n, p):
        #print("Counting negative and positive words")
        positive = 0
        negative = 0
        tweet_list = tweet.split(' ')
        for w in n:
            if w in tweet_list:
                negative += 1
        for w in p:
            if w in tweet_list:
                positive += 1
        return [positive, negative]

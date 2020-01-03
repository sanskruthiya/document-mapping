from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import word2vec
from statistics import mean
import sys
import csv
import string
import re
import MeCab
import pandas as pd
import numpy as np

#NLTK Tokenizer
def nltk_tokenizer(text):
    words = RegexpTokenizer(r'\w+').tokenize(text)
    node = pos_tag(words)
    stop_nltk = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    token_list = []
    for i in node:
        if not i in stop_nltk:
            t = lemmatizer.lemmatize(i[0])    
            token_list.append(t)
    return list(token_list)

#MeCab Tokenizer
def mecab_tokenizer(text):
    token_list = []
    tagger = MeCab.Tagger('/usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    tagger.parse('') 
    node = tagger.parseToNode(text)
    while node:
        pos = node.feature.split(",")
        if not node.surface in stop_words:
            if pos[6] != '*':
                token_list.append(pos[6])
            else:
                token_list.append(node.surface)
        node = node.next
    return list(token_list)

#Loading input dataset
df = pd.read_csv('input.csv', encoding="utf-8")
#Combine two columns into a new column
#df['text'] = df['title'].str.cat(df['abstract'], sep=' ')
df['text'] = df['description']

#Selecting Japanese(MeCab) or English(NLTK) based tokenizer
lang = input('Select language: en or jp >> ')
if lang == 'en':
    text_tokenizer = nltk_tokenizer
elif lang == 'jp':
    text_tokenizer = mecab_tokenizer
else:
    print('Program cancelled due to invalid input. Please enter en or jp.')
    sys.exit

#Remove https-links
df['text_clean'] = df.text.map(lambda x: re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', "", x))
#Remove numerals
df['text_clean'] = df.text.map(lambda x: re.sub(r'\d+', '', x))
#Converting all letters into lower case
df['text_clean'] = df.text_clean.map(lambda x: x.lower())
#Creating DataFrame for Token-list
df['text_tokens'] = df.text_clean.map(lambda x: text_tokenizer(x))

#Feature words extraction by tf-idf (option: min_df, max_df)
text_list = list(df['text_clean'])
vectorizer = TfidfVectorizer(tokenizer=text_tokenizer, max_df=0.95, ngram_range=(1,1))
tfidf_X = vectorizer.fit_transform(text_list).toarray()

#Print the number of data and words
print('The number of documents and words ->' + tfidf_X.shape)

#Sorting Tf-Idf result
index = tfidf_X.argsort(axis=1)[:,::-1]
feature_names = np.array(vectorizer.get_feature_names())
feature_words = feature_names[index]

#Feature words by documents 
fword_list = []
num_fword = 10 #set the number of displayed feature words
for i in range(len(index)):
    if len(feature_words[i]) > num_fword-1: #If the number of fwords less than set, input N/A.
        fword = feature_words[i][0]
        for c in range(num_fword):
            if not c == 0:
                fword = fword + '|' + feature_words[i][c]
        fword_list.append(fword)
    else:
        fword_list.append("N/A")

#Creating Dataframe for tf-idf values
df_tfidf = pd.DataFrame(data=tfidf_X, columns=vectorizer.get_feature_names())
#Switching raw and column: documents and words
df_tfidf_2 = df_tfidf.T
#Setting False if tfidf value = 0: the word is not included in the document
df_bool = (df_tfidf_2 > 0)
#Counting the number of True: document frequecy by each word
df_dof = pd.DataFrame(data=df_bool.sum(axis=1), columns=['count'])
#Creating Dataframe for idf values
idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
df_idf = pd.DataFrame(columns=['idf']).from_dict(dict(idf), orient='index')
df_idf.columns = ['idf']
#Combine words and idf DataFrames
df_merge = df_dof.merge(df_idf, left_index=True, right_index=True)
#Sorting the result of count by ascending order
df_dof_srt = df_merge.sort_values('count', ascending=True)
#Exporting as CSV
df_dof_srt.to_csv("df.csv", encoding="utf-8")

#Creating and saving Word2Vec model. Please comment-out when you load an existing model.
sent = df['text_tokens']
model = word2vec.Word2Vec(sent, size=300, min_count=3, window=5, iter=2000, seed=2000)
model.save("input.model")

#Loading Word2Vec model. Arrange name of the model when you use an existing model.
model = word2vec.Word2Vec.load("input.model")

#Initializing a 300-dimension data-frame
doc_vec = np.zeros((df.shape[0], 300))
#Prepairing an empty list for model coverage rate
coverage = []
#Caluculating average of the vector of words by a document
for i,doc in enumerate(df['text_tokens']):
    feature_vec = np.zeros(300) #Initializing a 300-dimension vector data as 0
    num_words = 0
    no_count = 0
    for word in doc:
        try:
            feature_vec += model.wv[word]
            num_words += 1
        except:
            no_count += 1
    feature_vec = feature_vec / num_words
    doc_vec[i] = feature_vec
    #Caluculating word coverage rate of the model by each document
    cover_rate = num_words / (num_words + no_count)
    coverage.append(cover_rate)

#Print overall word coverage rate of the model
mean_coverage = round(mean(coverage)*100, 2)
print("Word cover-rate: " + str(mean_coverage) + "%")

#Dimensionality reduction by tsne
tsne= TSNE(n_components=2, init='pca', verbose=1, random_state=2000, perplexity=50, learning_rate=200, method='exact', n_iter=1000)
embedding = tsne.fit_transform(doc_vec)
#Storing in DataFrame
embedding = pd.DataFrame(embedding, columns=['x', 'y'])
embedding["keywords"]=pd.DataFrame(fword_list)
embedding["id"]= df.id
embedding["year"]= df.year
embedding["title"]= df.title
embedding["abstract"]= df.abstract

#Export to CSV
embedding.to_csv("output.csv", encoding="utf_8", quoting=csv.QUOTE_ALL)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models
import sys
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
        #if i[1][:2] == 'NN': #Extract only noun
        if i[1] in ["FW", "JJ", "JJR", "JJS", "NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:   # Target FW:foreign word、NN:noun, JJ:adjective, RB:adverb("RB", "RBR"), VB:verb
            t = lemmatizer.lemmatize(i[0])
            if not t in stop_nltk:
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
        if pos[0] in ["名詞", "動詞", "形容詞"]: #target word-class
            if pos[6] != '*': #lemmatized-word is added when it exists
                token_list.append(pos[6])
            else:
                token_list.append(node.surface)        
        node = node.next
    return list(token_list)

#Loading input dataset
df = pd.read_csv('input.csv', encoding="utf-8")
#Combine two columns into a new column
#df['text'] = df['title'].str.cat(df['description'], sep=' ')
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
df['text_clean'] = df.text_clean.map(lambda x: re.sub(r'\d+', '', x))
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

#LDA
np.random.seed(2000)
texts = df['text_tokens'].values
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=20, passes=5, minimum_probability=0)
ldamodel.save('lda_bz.model')
print(ldamodel.print_topics())

#Converting Topic-Model result into numpy matrix
hm = np.array([[y for (x,y) in ldamodel[corpus[i]]] for i in range(len(corpus))])

#Dimensionality reduction by tsne
tsne = TSNE(n_components=3, init='pca', verbose=1, random_state=2000, perplexity=50, method='exact', early_exaggeration=120, learning_rate=200, n_iter=1000)
embedding = tsne.fit_transform(hm)

x_coord = embedding[:, 0]
y_coord = embedding[:, 1]
z_coord = embedding[:, 2]

#RGB変換
def std_norm(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    y = (x-xmean)/xstd
    min = y.min(axis=axis, keepdims=True)
    max = y.max(axis=axis, keepdims=True)
    norm_rgb = (y-min)/(max-min) * 254
    result = norm_rgb.round(0)
    return result

x_rgb = std_norm(x_coord, axis=0)
y_rgb = std_norm(y_coord, axis=0)
z_rgb = std_norm(z_coord, axis=0)

embedding = pd.DataFrame(x_coord, columns=['x'])
embedding['y'] = pd.DataFrame(y_coord)
embedding['z'] = pd.DataFrame(y_coord)
embedding["r"] = pd.DataFrame(x_rgb)
embedding["g"] = pd.DataFrame(y_rgb)
embedding["b"] = pd.DataFrame(z_rgb)
embedding['keywords'] = pd.DataFrame(fword_list)
embedding['id'] = df.id
embedding['name'] = df.title

#export to csv
embedding.to_csv("Output_byLDA_RGB.csv", encoding="utf_8")

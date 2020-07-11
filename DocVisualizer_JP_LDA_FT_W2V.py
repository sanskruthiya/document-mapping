from gensim import corpora, models
from glob import glob
from operator import itemgetter
from scipy.sparse.csgraph import connected_components
from statistics import mean
import fasttext
import hdbscan
import collections
import umap
import os
import sys
import csv
import string
import codecs
import re
import MeCab
import pandas as pd
import numpy as np

#MeCab Tokenizer
def mecab_tokenizer(tx, sw):
    token_list = []
    #tagger = MeCab.Tagger('/usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    tagger = MeCab.Tagger('/usr/local/lib/mecab/dic/ipadic')
    tagger.parse('')
    node = tagger.parseToNode(tx)
    while node:
        if not node.surface in sw:
            pos = node.feature.split(",")
            if not pos[0] in ["記号"]: #target word-class
                if not pos[6] == '*': #lemma is added when it exists
                    token_list.append(pos[6])
                else:
                    token_list.append(node.surface)
        node = node.next
    return list(token_list)

#Feature-word extractor
def fword_extractor(num, docs):
    fword = docs[0][0]
    for c in range(num+1):
        if not c == 0:
            fword = fword + '|' + docs[c][0]
    return fword

#LDA vectorizer
def lda_vectorizer(corps, dict, bsn):
    #Creating LDA model
    topics_input = input('Input the number of topics for LDA model (default=40) >> ')
    try:
        num_topics = int(topics_input)
    except:
        num_topics = 40 #set default value
        print("The number of topics is set as 40, due to invalid input")
    a_weight = num_topics / 10 

    model_file = "02_models/" + bsn + "_lda_" + str(num_topics) + ".model"
    
    if not os.path.exists(model_file):
        ldamodel = models.ldamodel.LdaModel(corps, id2word=dict, num_topics=num_topics, random_state=1, alpha=a_weight/num_topics, minimum_probability=0) #minimum_probability=0にすることで文書数xトピック数の配列ができる
        ldamodel.save(model_file)
        topic_list = []
        with open("02_models/" + bsn + "_lda_" + str(num_topics) + "_topics.txt", 'w', encoding="utf_8") as t:
            for i in range(num_topics):
                i_md = "No." + str(i) + "\t" + ldamodel.print_topic(i)
                topic_list.append(i_md)
                i += 1
            t.write("\n".join(topic_list))
    else:
        pass
    
    #Loading an existing LDA model
    model = models.LdaModel.load(model_file)
    lda_vec = np.array([[y for (x,y) in model[corps[i]]] for i in range(len(corps))])
    print('The number of documents: ' + str(lda_vec.shape[0]))
    return lda_vec

#FastText vectorizer
def ft_vectorizer(tx, bsn):
    model_path = "02_models/cc.ja.300.bin"
    if not os.path.exists(model_path):
        model_txt_path = "02_models/" + bsn + "_ft_corpus" + ".txt"
        model_alt_path = "02_models/" + bsn + "_ft_corpus" + ".bin"
        with open(model_txt_path, 'w', encoding="utf_8") as f:
            text_corpus = ""
            for t in tx:
                text_corpus = " ".join(map(str, t))
                f.write(text_corpus+"\n")
        print("New FastText model is created. See " + model_alt_path)
        model = fasttext.train_unsupervised(model_txt_path, model='cbow', dim=300)
        #model.save_model(model_alt_path)
    else:
        print("Pre-trained model " + model_path + " is used.")
        model = fasttext.load_model(model_path)
    
    ft_vec = np.zeros((df.shape[0], 300))
    #Prepairing an empty list for model coverage rate
    coverage = []
    #Caluculating average of the vector of words by a document
    for i,doc in enumerate(tx):
        feature_vec = np.zeros(300) #Initializing a 300-dimension vector data as 0
        num_words = 0
        no_count = 0
        for word in doc:
            #word_t = "'"+str(word)+"'"
            try:
                feature_vec += model[str(word)]
                num_words += 1
            except:
                no_count += 1
        feature_vec = feature_vec / num_words
        ft_vec[i] = feature_vec
        #Caluculating word coverage rate of the model by each document
        cover_rate = num_words / (num_words + no_count)
        coverage.append(cover_rate)
    #Print overall word coverage rate of the model
    mean_coverage = round(mean(coverage)*100, 2)
    print("Word cover-rate: " + str(mean_coverage) + "%")
    return ft_vec

#Word2Vec vectorizer
def w2v_vectorizer(tx, bsn):
    d_size = 300
    model_file = "02_models/" + bsn + "_w2v_" + str(d_size) + ".model"
    if not os.path.exists(model_file):
        w2vmodel = models.word2vec.Word2Vec(tx, size=d_size, min_count=3, window=5, iter=2000, seed=2000)
        w2vmodel.save(model_file)
    else:
        pass
    #Loading Word2Vec model. Arrange name of the model when you use an existing model.
    model = models.word2vec.Word2Vec.load(model_file)
    #Initializing a 300-dimension data-frame
    w2v_vec = np.zeros((df.shape[0], 300))
    #Prepairing an empty list for model coverage rate
    coverage = []
    #Caluculating average of the vector of words by a document
    for i,doc in enumerate(tx):
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
        w2v_vec[i] = feature_vec
        #Caluculating word coverage rate of the model by each document
        cover_rate = num_words / (num_words + no_count)
        coverage.append(cover_rate)
    #Print overall word coverage rate of the model
    mean_coverage = round(mean(coverage)*100, 2)
    print("Word cover-rate: " + str(mean_coverage) + "%")
    return w2v_vec

#Open input-data and store it into DataFrame
docs_file = input('Input a document-list file (.csv) >> ')
if os.path.exists(docs_file):
    with codecs.open(docs_file, 'r', 'utf-8', 'ignore') as f:
        df = pd.read_csv(f, delimiter='\t')
    base_name = os.path.splitext(os.path.basename(docs_file))[0] #file name without extension
else:
    print('Program cancelled due to invalid input. Please enter correct file name')
    sys.exit()

#Open stopword list
stops_file = input('Input a stopword-list file (.txt) >> ')
if os.path.exists(stops_file):
    with codecs.open(stops_file, 'r', 'utf-8', 'ignore') as f:
        stopwords = f.read().splitlines()
else:
    print('Stopword file is not found. This process continues without stopwords.')
    stopwords = []

os.makedirs("02_models", exist_ok=True)
os.makedirs("03_vector_data", exist_ok=True)

#Combine plural columns into a new column
df['txt'] = df['title'].str.cat(df['abstract'], sep=' | ', na_rep=' - ')
df['text'] = df['txt'].str.cat(df['description'], sep=' | ', na_rep=' - ')

#Selecting Japanese(MeCab) or English(NLTK) based tokenizer
text_tokenizer = mecab_tokenizer

#Vectorize modelling
method = input('Select vectorize method - LDA:1, FastText:2, W2V:3 >> ')

#Remove https-links
df['text_clean'] = df.text.map(lambda x: re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', "", x))
#Remove numerals
df['text_clean'] = df.text_clean.map(lambda x: re.sub(r'\d+', '', x))
#Remove symbols
df['text_clean'] = df.text_clean.map(lambda x: re.sub(r'[「」。、,（）%#\$&\?\(\)~\.=\+\-\[\]\{\}\|\*]+', '', x))
#Converting all letters into lower case
df['text_clean'] = df.text_clean.map(lambda x: x.lower())
#Creating DataFrame for Token-list
df['text_tokens'] = df.text_clean.map(lambda x: text_tokenizer(x, stopwords))

#Creating dictionary and corpus
texts = df['text_tokens'].values
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=3, no_above=0.4)
corpus = [dictionary.doc2bow(text) for text in texts]

#Feature words extraction by tf-idf (option: min_df, max_df)
tfidf_model = models.TfidfModel(corpus, id2word=dictionary.token2id, normalize=False)
tfidf_corpus = list(tfidf_model[corpus])

if method == "1":
    method_name = "LDA"
    doc_vec = lda_vectorizer(tfidf_corpus, dictionary, base_name)
elif method == "2":
    method_name = "FT"
    doc_vec = ft_vectorizer(df['text_tokens'], base_name)
elif method == "3":
    method_name = "W2V"
    doc_vec = w2v_vectorizer(df['text_tokens'], base_name)
else:
    print('Program cancelled due to invalid input. Please enter 1, 2 or 3.')
    sys.exit()

tfidf_texts = []
for doc in tfidf_corpus:
    tfidf_text = []
    for word in doc:
        tfidf_text.append([dictionary[word[0]], word[1]])
        tfidf_text_sort = sorted(tfidf_text, reverse=True, key=itemgetter(1))
    tfidf_texts.append(tfidf_text_sort)

num_fword = 40 #set the number of displayed feature words
fword_list = []
for i in range(len(tfidf_texts)):
    if len(tfidf_texts[i]) > num_fword-1: #If the number of fwords less than set, input N/A.
        fword = fword_extractor(num_fword-1, tfidf_texts[i])
    else:
        fword = fword_extractor(len(tfidf_texts[i])-1, tfidf_texts[i])
    fword_list.append(fword)

#Storing in DataFrame
embedding_u = umap.UMAP(min_dist=0.1, n_neighbors=50, metric='euclidean', spread=1.0).fit_transform(doc_vec)
embedding = pd.DataFrame(embedding_u, columns=['x', 'y'])
embedding['id'] = df.doc_id
embedding['project_id'] = df.prj_id
embedding['title'] = df.title
embedding['keywords'] = fword_list
embedding['published_year'] = df.pub_year
embedding['start_year'] = df.start_year
embedding['end_year'] = df.end_year
embedding['affiliation'] = df.affiliation
embedding['department'] = df.department
embedding['division'] = df.division
embedding['person'] = df.person
embedding['account_type'] = df.account_class
embedding['budget'] = df.budget
embedding['execution'] = df.execution
embedding['link'] = df.link_url

x_coord = embedding_u[:, 0]
y_coord = embedding_u[:, 1]

#clustering
clst_input = input('Input the number for min-size of a cluster (default=100) >> ')
try:
    clusters = int(clst_input)
except:
    clusters = 100 #set default value
    print("The number of cluster is set as 100, due to invalid input")

type_input = input('Input type of clustering - eom:1 or leaf:2 >> ')
if type_input == "1":
    type_c = 'eom'
elif type_input == "2":
    type_c = 'leaf'
else:
    type_c = 'leaf' #set default value
    print("Type of clustering is set as leaf due to invalid input.")

clustering = hdbscan.HDBSCAN(cluster_selection_method=type_c, min_cluster_size=clusters, min_samples=10) #The lower the value, the less noise you’ll get
log_method = "HDBSCAN (minimum size of clusters: " + str(clusters) + " )"

log_list = [] # set an empty list for log
label_list = [] # set an empty list for labelled area file
log_list.append(log_method) #log the method of clustering

#Extract only XY coordinates from DataFrame for clustering calculation
df_xy = embedding.loc[:, ['x', 'y']]

#Convert XY coordicates into numpy array
X = df_xy.to_numpy()
log_doc = "Documents : " + str(X.shape[0]) #log the number of documents
log_list.append(log_doc)
print(log_doc) #display the number of documents

#Execution of clustering (using scikit-learn based modules)
clustering.fit(X)

#Store the label data
labels = clustering.labels_
num_labels = len(set(labels)) - (1 if -1 in labels else 0) #the number of labels
outliers = list(labels).count(-1) #the number of outliers
log_label = "labels : " + str(num_labels) + ", outliers : " + str(outliers) #log the number of labels and outliers
log_list.append(log_label)
print(log_label) #display the number of labels and outliers

#Create DataFrame of ID and Label No
embedding_L1 = pd.DataFrame()
embedding_L1["id"] = df.doc_id
embedding_L1["keywords"] = fword_list
embedding_L1["X"] = x_coord
embedding_L1["Y"] = y_coord
embedding_L1["label"] = labels
#embedding_L1["Area_Label"] = "No." + embedding_L1["label" + 1].astype(str) + "-area"

#Feature words extraction
num = 0
log_head = "area_id" + '\t' + "documents" + '\t' + "keywords" + '\t' + "X" + '\t' + "Y" + '\t' + "label" #log header
log_list.append(log_head)
label_list.append(log_head)
while num < num_labels:
    f_list = [] #empty list for storing all feature-words of documents
    embedding_L2 = embedding_L1.query("label == @num") #filtering the columns by the label
    #convert the filtered feature words into array
    fwords = np.array(embedding_L2['keywords'])
    s = fwords.T
    #add feature words into list removing separator bar
    for i in s:
        try: #exception for avoiding errors due to unexpected value for feature words
            l = [x.strip() for x in i.split('|')]
            f_list.extend(l)
        except:
            pass
    #count the frequency of feature words by each label
    c = collections.Counter(f_list)
    top_c = c.most_common(40)[0:39]
    #count the number of documents in each label
    doc_n = len(embedding_L2)
    #caluculate centroid coordinates
    x_mean = embedding_L2["X"].mean()
    y_mean = embedding_L2["Y"].mean()
    #tag name for each label based on top2 with rate
    l_key1 = top_c[0][0] + "/" + top_c[1][0] + "/" + top_c[2][0]
    l_key2 = [k for k, v in top_c[3:] if v / doc_n > 0.4 and v / doc_n < 0.65]
    if len(l_key2) > 0:
        l_keys = l_key1 + "/" + l_key2[0]
    else:
        l_keys = l_key1
    #create log data of each label
    log_fwords = str(num) + "\t" + str(doc_n) + "\t" + str(top_c) + "\t" + str(round(x_mean, 4)) + "\t" + str(round(y_mean, 4)) + "\t" + str(l_keys) #log the feature words of areas
    log_list.append(log_fwords)
    label_list.append(log_fwords)
    print(log_fwords) #display the feature words of areas
    num += 1

#export to csv
embedding.to_csv("03_vector_data/" + base_name + "_by" + method_name + ".csv", encoding="utf_8", index=False)

#export log
log_file = "log_" + str(num_labels) + ".txt"
with open("03_vector_data/" + base_name + "_by" + method_name + "_" + log_file, 'w', encoding="utf_8") as t:
    t.write("\n".join(log_list))

#export XY centroid of the labelled areas
label_name = "label_" + str(num_labels) + ".tsv"
with open("03_vector_data/" + base_name + "_by" + method_name + "_" + label_name, 'w', encoding="utf_8") as t:
    t.write("\n".join(label_list))

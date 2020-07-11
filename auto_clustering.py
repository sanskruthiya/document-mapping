import hdbscan
import os
import sys
import codecs
import collections
import pandas as pd
import numpy as np

file_name = input('Input csv file >> ')
if os.path.exists(file_name):
    with codecs.open(file_name, 'r', 'utf-8', 'ignore') as f:
        df = pd.read_csv(f)
    base_name = os.path.splitext(os.path.basename(file_name))[0] #file name without extension
else:
    print('Program cancelled due to invalid input. Please enter correct file name')
    sys.exit()

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

#clustering
clustering = hdbscan.HDBSCAN(cluster_selection_method=type_c, min_cluster_size=clusters, min_samples=10) #The lower the value, the less noise you’ll get
log_method = "HDBSCAN (minimum size of clusters: " + str(clusters) + " ) / Type: " + type_c

log_list = [] # set an empty list for log
label_list = [] # set an empty list for labelled area file
log_list.append(log_method) #log the method of clustering

#Extract only XY coordinates from DataFrame for clustering calculation
df_xy = df.loc[:, ['x', 'y']]

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
embedding = pd.DataFrame()
embedding["id"] = df["id"]
embedding["keywords"] = df["keywords"]
embedding["X"] = df["x"]
embedding["Y"] = df["y"]
embedding["label"] = labels
#embedding_L1["Area_Label"] = "No." + embedding_L1["label" + 1].astype(str) + "-area"

#Feature words extraction
num = 0
log_head = "area_id" + '\t' + "documents" + '\t' + "keywords" + '\t' + "X" + '\t' + "Y" + '\t' + "label" #log header
log_list.append(log_head)
label_list.append(log_head)
while num < num_labels:
    f_list = [] #empty list for storing all feature-words of documents
    embedding2 = embedding.query("label == @num") #filtering the columns by the label
    #convert the filtered feature words into array
    fwords = np.array(embedding2['keywords'])
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
    top_c = c.most_common(15)[0:14]
    #count the number of documents in each label
    doc_n = len(embedding2)
    #caluculate centroid coordinates
    x_mean = embedding2["X"].mean()
    y_mean = embedding2["Y"].mean()
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

#export log
log_file = "log_" + str(num_labels) + ".txt"
with open(base_name + "_" + log_file, 'w', encoding="utf_8") as t:
    t.write("\n".join(log_list))

#export XY centroid of the labelled areas
label_name = "label_" + str(num_labels) + ".tsv"
with open(base_name + "_" + label_name, 'w', encoding="utf_8") as t:
    t.write("\n".join(label_list))

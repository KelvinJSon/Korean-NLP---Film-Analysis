# %%
# Importing various packages
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import re 
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns 
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
print("Import Done")
# %%
# Getting the Korean film reviews
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train = pd.read_table('ratings_train.txt') #splitting vetween training and test datasets
test = pd.read_table('ratings_test.txt')
train.head()
# %%
# Dropping numerical ID column as it is not helpful
train = train.drop(['id'],axis=1) 
test = test.drop(['id'],axis=1)
train.info()
# %%
# Checking the training data for any null values and locating them
print(train.isnull().values.any())
train.loc[train.document.isnull()]
# %%
# Dropping null reviews in training
train = train.dropna()
train.info()
# %%
# Creating another column where punctuations are removed for the review
train['word_n_2']=train['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]"," ")
test['document']=test['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]"," ")
test['document'].replace('', np.nan, inplace=True)
test = test.dropna()
train.head(20)
# %%
# Tokenization and Removing Stopwords
import os
'JAVA_HOME' in os.environ
from tqdm import tqdm
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17.0.1'
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
X_train = []
X_test = []
for sentence in tqdm(train['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # Train set tokenization
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # train set stopwords
    X_train.append(stopwords_removed_sentence)
for sentence in tqdm(test['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # Test set tokenization
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # test set stowords
    X_test.append(stopwords_removed_sentence)
print('Max Length for Reviews :',max(len(review) for review in X_train))
print('Average Length of Reviews :',sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('Length of Review Samples')
plt.ylabel('Number of Review samples')
plt.show()
# %%
# Limiting tokens to words that have only shown up more than 4h times 
max_features = 9308 # Number of words that have popped up more than 4 times in the training set
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# %%
# Padding and cutting reviews when they are > 50 words
drop_train = [index for index, sentence in enumerate(X_test) if len(sentence) <1]
X_test = np.delete(X_test, drop_train, axis=0)
max_len = 50
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=50)
# %%
# Creating the model
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, GlobalMaxPool1D
y_train = np.array(train['label'])
y_test = np.array(test['label'])
y_test = np.delete(y_test, drop_train, axis=0)
embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)
# %%
# Taking the best model used for our data
from tensorflow.keras.models import load_model
loaded_model = load_model('best_model.h5')
print("\n Test Accuracy: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
model.summary()
# %%
# Evaluating model performance
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import re
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report
import scikitplot as skplt

# Making predictions using 0.5 as the cutooff
predictions = loaded_model.predict(X_test)
predictions = (predictions >= 0.5)
from scikitplot.metrics import plot_confusion_matrix
plot_confusion_matrix(y_test,predictions)
acc_score = accuracy_score(y_test,predictions)
pre_score = precision_score(y_test,predictions)
rec_score = recall_score(y_test,predictions)
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print("-"*50)
cr = classification_report(y_test,predictions)
print(cr)
# %%
# Plotting ROC curve
predictions_probability = loaded_model.predict(X_test)
fpr,tpr,thresholds = roc_curve(y_test,predictions_probability[:,0])
plt.plot(fpr,tpr)
plt.plot([0,1])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
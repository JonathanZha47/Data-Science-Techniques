from nltk.stem.snowball import SnowballStemmer
from snowballstemmer import stemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
## Read the data
print("Reading the data")
X_train = pd.read_csv("./data/X_train.csv") 
X_test = pd.read_csv("./data/X_test.csv")
X_train['Total'] = X_train['Text'] + X_train['Summary']
X_test['Total'] = X_test['Text'] + X_test['Summary']

## Train Stemming
print("Train Stemming")
df_train_text = pd.DataFrame(X_train['Total'])
stemmer = SnowballStemmer("english", ignore_stopwords=True)

def stem(text):
    return ' '.join([stemmer.stem(word) for word in word_tokenize(text)])

df_train_text['Total'] = df_train_text['Total'].fillna("").apply(stem)
df_train_text.to_csv("./data/df_train_text.csv", index=False)

## Test Stemming
print("Test Stemming")
df_test_text = pd.DataFrame(X_test['Total'])
df_test_text['Total'] = df_test_text['Total'].fillna("").apply(stem)
df_test_text.to_csv("./data/df_test_text.csv", index=False)


stem_train = pd.read_csv("./data/df_train_text.csv")
stem_test = pd.read_csv("./data/df_test_text.csv")

stem_train.fillna(' ',inplace=True)
stem_test.fillna(' ',inplace=True)
## Train TF-IDF
print("Train TF-IDF")
vectorizer = TfidfVectorizer(max_df=.8,min_df=0.15,max_features=1000)
tfidf_train = vectorizer.fit_transform(stem_train['Total'])
train_matrix = pd.DataFrame(tfidf_train.toarray(), columns=vectorizer.get_feature_names())
train_matrix.to_csv("tfidf_train.csv")

## Test TF-IDF
print("Test TF-IDF")
tfidf_test = vectorizer.transform(stem_test['Total'])
test_matrix = pd.DataFrame(tfidf_test.toarray(), columns=vectorizer.get_feature_names())
test_matrix.to_csv("tfidf_test.csv")

## Combine TF-IDF with other features
print("Combine TF-IDF with other features")
train_tfidf = pd.read_csv("tfidf_train.csv")
test_tfidf = pd.read_csv("tfidf_test.csv")
train = pd.concat([X_train,train_tfidf],axis=1)
test = pd.concat([X_test,test_tfidf],axis=1)

## Drop unnecessary columns
print("Drop unnecessary columns")
train_processed = train.drop(columns=['Id','ProductId','UserId','Summary','Text','Unnamed: 0','Time','Total'])
train_processed.fillna(0,inplace=True)
test_processed = test.drop(columns=['Id','ProductId','UserId','Summary','Text','Unnamed: 0','Time','Total'])
test_processed.fillna(0,inplace=True)

## Save the processed data
print("Save the processed data")
train_processed.to_csv("./data/train_processed.csv", index=False)
test_processed.to_csv("./data/test_processed.csv", index=False)



import nltk, string, csv,array, numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
import pandas as pd

def fun(dataaa):
    #to store lemmatized data
    data=""
    l=WordNetLemmatizer()
    text=word_tokenize(dataaa)
    #lemmatizing each word
    for w in text:
        x=l.lemmatize(w)
        data=data+" "+x

    #it is the pre defined stop words so as to compare and not copy
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(data)

    wordsFiltered = ""

    #not copying the stop words
    for w in words:
        if w not in stopWords:
            wordsFiltered=wordsFiltered+" "+w
    #final output
    return wordsFiltered


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

rows=[]
columns=[]

with open(r'G:\IIIT Hyderabad Megathon\NLP\abstract1.csv','rt')as f:
  abstract = csv.reader(f)
  for row in abstract:
        rows=rows+row

with open(r'G:\IIIT Hyderabad Megathon\NLP\files1.csv','rt')as f:
  files = csv.reader(f)
  for column in files:
        columns=columns+column



a = [[0 for x in range(len(rows))] for x in range(len(columns))]
for i in range(len(rows)):
    for j in range(len(columns)):
        a[j][i]=cosine_sim(rows[i],columns[j])

df = pd.DataFrame(a)
df.to_csv(r'G:\IIIT Hyderabad Megathon\NLP\data.csv', index=False)

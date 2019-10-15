from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#to store lemmatized data
data=""
l=WordNetLemmatizer()
text=word_tokenize("better is for the best, how are you.      .Strings are sequences of characters.")
#lemmatizing each word
for w in text:
    x=l.lemmatize(w)
    data=data+" "+x

#it is the pre defined stop words so as to compare and not copy
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)

wordsFiltered = ""
print(wordsFiltered)

#not copying the stop words
for w in words:
    if w not in stopWords:
        wordsFiltered=wordsFiltered+" "+w
#final output
print(wordsFiltered)

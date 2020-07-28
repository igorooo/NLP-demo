from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

stopwords = []
with open('../language/polish.stopwords.txt', 'r') as f:
    for word in f.readlines():
        stopwords.append(word.replace('\n', ''))

vectorizer = CountVectorizer(stop_words=stopwords)
vectorizer2 = TfidfVectorizer()

texts = ['super ale super było super']

XX = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names())
print(XX.toarray())



training_data = ['Pogoda była super, bylo słonecznie, padał deszcz i śnieg',
                 'Wyścigi były szybkie, biegali, były rowery i sport',
                 'Wczoraj wieczorem w pogodzie przewidzieli, że dziś będzie padał deszcze',
                 'Ostatnio zacząłem biegać na rowerze, super sprawa takie sporty']
labels = ['pogoda', 'sport', 'pogoda', 'sport']


X = vectorizer.fit_transform(training_data)
X1 = vectorizer2.fit_transform(training_data)

print(vectorizer.get_feature_names())
print(vectorizer2.get_feature_names())

clf = MultinomialNB()
clf.fit(X1, labels)



test_data = ['Jutro ma być super pogoda, będzie padał deszcz i wgle',
             'jutro będziemy biegali na rowerze w deszczu i śniegu jak będzie padał deszcz']

Xt = vectorizer2.transform(test_data)

print(Xt)

print(clf.predict(Xt.toarray()))



























"""str = ''

with open('../training_data/Albania_59.txt', 'r') as f:
    str = f.read()

str = str.lower()
tokens_raw = nltk.word_tokenize(str,'polish')
stop_words = stopwords.words('polish')

tokens = [w for w in tokens_raw if not w in stop_words]
"""
#tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
"""tokens = wordfreq.tokenize(text=str, lang='pl')
print(tokens)

print(wordfreq.get_language_info('pl'))"""


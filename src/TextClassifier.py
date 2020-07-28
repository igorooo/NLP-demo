from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import stempel
from copy import copy
from random import shuffle


training_set_path = '../training_data'
training_set_path_stemm = '../training_data_stemm'
test_set_path = '../test_data'
test_set_path_stemm = '../test_data_stemm'

def stemming(text):
    marks = ['.', '.', '(', ')']
    text = text.lower()
    for m in marks:
        text = text.replace(m, "")

    stemmer = stempel.StempelStemmer.default()

    words = text.split(" ")
    for word in words:
        if word.isalpha():
            stem = stemmer.stem(word)
            if stem != None:
                text = text.replace(word, stemmer.stem(word))

    return text

def readStopWords():
    stopwords = []
    with open('../language/polish.stopwords.txt', 'r') as f:
        for word in f.readlines():
            word = word.replace('\n', '')
            stopwords.append(word)
    return stopwords

def removeStopWords(text, stopwords):
    txt = copy(text)
    for stopword in stopwords:
        txt = txt.replace(stopword, "")

    return txt

def extractDataSet(path):
    """
    Data structure:
    {
        labels : list of n labels
        content : list of n files content
    }
    :param path: dir path
    :return: dir
    """

    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = {
        'labels' : [],
        'content' : []
    }
    for file in files:

        if not '_' in file:
            continue

        label = file.split("_")[0]
        content = ''
        with open(path + '/' + file, 'r') as f:
            content = f.read()

        data['labels'].append(label)
        data['content'].append(content)

    return data

def NLP(training_data, test_data, vectorizer_type='tfidf', algorithm='nb', stopwords=None):
    if stopwords != None:
        training_data['content'] = [removeStopWords(cont, stopwords) for cont in training_data['content']]
        test_data['content'] = [removeStopWords(cont, stopwords) for cont in test_data['content']]

    vectorizer = None
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer()

    if vectorizer_type == 'count':
        vectorizer = CountVectorizer()

    if vectorizer_type != 'tfidf' and vectorizer_type != 'count':
        print("wrong vectorizer")
        return


    X = vectorizer.fit_transform(training_data['content'])
    Xtest = vectorizer.transform(test_data['content'])

    clf = None

    if algorithm == 'nb':
        clf = MultinomialNB()

    if algorithm == 'dt':
        clf = DecisionTreeClassifier()

    if algorithm != 'dt' and algorithm != 'nb':
        print("wrong algorithm")
        return

    clf.fit(X, training_data['labels'])
    test_result = clf.predict(Xtest.toarray())

    counter = 0
    for i in range(0, len(test_result)):
        if test_result[i] == test_data['labels'][i]:
            counter += 1

    return (counter / len(test_result) * 1.0)


def shuffle_data(training_data, test_data):
    contents = training_data['content'] + test_data['content']
    labels = training_data['labels'] + test_data['labels']

    data = list(zip(contents, labels))

    shuffle(data)

    treshold = int(len(data) * 3/4)

    tr_data = {
        'content' : [x[0] for x in data[:treshold]],
        'labels': [x[1] for x in data[:treshold]]
    }

    te_data = {
        'content': [x[0] for x in data[treshold:]],
        'labels': [x[1] for x in data[treshold:]]
    }

    return tr_data, te_data



training_data = extractDataSet(training_set_path)
test_data = extractDataSet(test_set_path)

training_data, test_data = shuffle_data(training_data, test_data)

training_data_stemm = extractDataSet(training_set_path_stemm)
test_data_stemm = extractDataSet(training_set_path_stemm)

training_data_stemm, test_data_stemm = shuffle_data(training_data_stemm, test_data_stemm)

results = []

"""# Naive Bayes
res = NLP(copy(training_data), copy(test_data), vectorizer_type="tfidf", algorithm='nb')
results.append(('tfidf nb', res))
res = NLP(copy(training_data), copy(test_data), vectorizer_type="count", algorithm='nb')
results.append(('count nb', res))

res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="tfidf", algorithm='nb')
results.append(('tfidf nb stemm', res))
res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="count", algorithm='nb')
results.append(('count nb stemm', res))

res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="tfidf", algorithm='nb', stopwords=readStopWords())
results.append(('tfidf nb stemm stopwords', res))
res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="count", algorithm='nb', stopwords=readStopWords())
results.append(('count nb stemm stopwords', res))"""


ress = []
nn = 20
sum = 0
for i in range(nn):
    training_data_stemm, test_data_stemm = shuffle_data(training_data_stemm, test_data_stemm)
    res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="count", algorithm='nb',
              stopwords=readStopWords())
    print(res)
    sum += res

print(sum/nn*1.0)



"""# Decision tree
res = NLP(copy(training_data), copy(test_data), vectorizer_type="tfidf", algorithm='dt')
results.append(('tfidf dt', res))
res = NLP(copy(training_data), copy(test_data), vectorizer_type="count", algorithm='dt')
results.append(('count dt', res))

res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="tfidf", algorithm='dt')
results.append(('tfidf dt stemm', res))
res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="count", algorithm='dt')
results.append(('count dt stemm', res))

res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="tfidf", algorithm='dt', stopwords=readStopWords())
results.append(('tfidf dt stemm stopwords', res))
res = NLP(copy(training_data_stemm), copy(test_data_stemm), vectorizer_type="count", algorithm='dt', stopwords=readStopWords())
results.append(('count dt stemm stopwords', res))


for result in results:
    print(result)
    
    """



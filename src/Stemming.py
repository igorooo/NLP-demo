from os import listdir
from os.path import isfile, join
import stempel


training_set_path = '../training_data'
training_set_savepath = '../training_data_stemm'
test_set_path = '../test_data'
test_set_savepath = '../test_data_stemm'

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

def stemm_data(path, savepath):

    files = [f for f in listdir(path) if isfile(join(path, f))]

    for file in files:

        if not '_' in file:
            continue

        label = file.split("_")[0]
        content = ''
        with open(path + '/' + file, 'r') as f:
            content = f.read()

        with open(savepath + '/' + file, 'w') as f:
            f.write(stemming(content))
            print("saved")

stemm_data(training_set_path, training_set_savepath)
stemm_data(test_set_path, test_set_savepath)


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('stopwords')
# -----------------------------------------------------------------------------
#                                  Config
# -----------------------------------------------------------------------------

# If the model trains with only the first 50 words of the plot
weak = False

trainPath = "./train.txt"
trainDF = pd.read_csv(trainPath, sep='\t', names=['title', 'from', 'genre', 'director', 'plot'])

testPath = "./test_no_labels.txt"
testDF = pd.read_csv(testPath, sep='\t', header=None, names=['title', 'from', 'director', 'plot'])

# -----------------------------------------------------------------------------
#                                 Main Code
# -----------------------------------------------------------------------------

def apply_preprocessing(text):
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mustn't": "must not",
        "shan't": "shall not",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "wasn't": "was not",
        "we'd": "we would",
        "we're": "we are",
        "weren't": "were not",
        "what's": "what is",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "would've" : "would have",
        "you'd": "you would",
        "you're": "you are",
    }
    lowered = text.lower()
    lowered_re = re.sub(r'[^a-zA-Z\s]', '', lowered)
    words = nltk.word_tokenize(lowered_re)
    
    preproc = ' '.join(contractions.get(word, word) for word in words)
    tokens = word_tokenize(preproc, "english")
    
    for token in tokens:
        if(all(char in string.punctuation for char in token)):
            tokens.remove(token)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)


X_train = None
if not weak:
    X_train = trainDF['plot']
else:
    trainDF = trainDF.to_dict()
    for i in range(len(trainDF['plot'])):
        words = trainDF['plot'][i].split(' ')
        plot = ""
        for j in range(len(words)):
            if j == 50:
                break
            plot += words[j] + " "
        trainDF['plot'][i] = plot
    trainDF = pd.DataFrame.from_dict(trainDF)
    X_train = trainDF['plot']

y_train = trainDF['genre']
X_test = testDF['plot']

# Maybe something to process the plot ????
X_train = X_train.apply(apply_preprocessing)
X_test = X_test.apply(apply_preprocessing)

print(X_train)

# Naive Bayes pipeline with CountVectorizer and TF-IDF
nb_pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer()),
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

# parameters for grid search
param_grid = {
    'count_vectorizer__max_features': [1000, 3000, 5000, 10000, 15000, 20000],
    'tfidf_transformer__use_idf': [True, False],
    'classifier__alpha': [0.1, 0.2, 0.5, 1.0],
}

grid_search = GridSearchCV(nb_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_classifier = grid_search.best_estimator_
y_test_pred = best_classifier.predict(X_test)

# print(f"Best parameters: {grid_search.best_params_} \n")

# writing the results to the file
output_path = 'results.txt'
with open(output_path, 'w') as output:
    for label in y_test_pred:
        output.write(label + '\n')
print(f"Results written to {output_path}")
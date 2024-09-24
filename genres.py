import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


trainPath = "./train.txt"
trainDF = pd.read_csv(trainPath, sep='\t', names=['title', 'from', 'genre', 'director', 'plot'])

testPath = "./test_no_labels.txt"
testDF = pd.read_csv(testPath, sep='\t', header=None, names=['title', 'from', 'director', 'plot'])

X_train = trainDF['plot']
y_train = trainDF['genre']
X_test = testDF['plot']

# Maybe something to process the plot ????


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
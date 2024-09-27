import pandas as pd

testPath = "./test_no_labels.txt"
testDF = pd.read_csv(testPath, sep='\t', header=None, names=['title', 'from', 'director', 'plot'])
testDF = testDF.to_dict()
test = testDF['title']

answers = []
with open('results.txt') as f:
    output = f.read()
    answers = output.split('\n')
    f.close()

df = pd.read_csv("wiki_movie_plots_deduped.csv")
df = df.to_dict()
genresCorrect = df['Genre'] 
titlesCorrect = df['Title']

checked = []
g = 0
correct = 0
for key in test:
    for num in titlesCorrect:
        if test[key] == titlesCorrect[num] and answers[g] == genresCorrect[num] and test[key] not in checked:
            checked.append(test[key])
            correct += 1
    g += 1

print("\naccuracy:", correct / len(answers) * 100, "%\n")
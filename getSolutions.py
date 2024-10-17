import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the test data
testPath = "./test_no_labels.txt"
testDF = pd.read_csv(testPath, sep='\t', header=None, names=['title', 'from', 'director', 'plot'])
testDF = testDF.to_dict()
test = testDF['title']

# Load the answers (predictions)
answers = []
with open('results.txt') as f:
    output = f.read()
    answers = output.split('\n')
    f.close()

# Load the correct genre data
df = pd.read_csv("wiki_movie_plots_deduped.csv")
df = df.to_dict()
genresCorrect = df['Genre']
titlesCorrect = df['Title']

real = []
predicted = []

checked = []
g = 0
correct = 0

# Iterate through the test data and match titles with genres
for key in test:
    for num in titlesCorrect:
        if test[key] == titlesCorrect[num] and test[key] not in checked:
            r = None
            if answers[g] in genresCorrect[num]:
                r = answers[g]
            else:
                if genresCorrect[num] not in ["drama", "comedy", "horror", "action", "romance", 
                                              "western", "animation", "crime", "sci-fi"]:
                    r = "action"
                else:
                    r = genresCorrect[num]
            predicted.append(answers[g])
            real.append(r)

        if test[key] == titlesCorrect[num] and answers[g] in genresCorrect[num] and test[key] not in checked:
            checked.append(test[key])
            correct += 1

    g += 1

# Define the genre labels (ensure this matches the categories you're working with)
genre_labels = ["action", "animation", "comedy", "crime", "drama", "horror", "romance", "sci-fi", "western"]

# Print accuracy
print("\naccuracy:", correct / len(answers) * 100, "%\n")

# Calculate and print the number of correctly predicted "drama" genres
counter = sum(1 for i in range(len(predicted)) if predicted[i] == real[i] and predicted[i] == "drama")
print(counter)

# Create the confusion matrix
matrix = metrics.confusion_matrix(real, predicted, labels=genre_labels)

# Plot the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(10, 7))
cax = ax.matshow(matrix, cmap="Blues")  # Display the matrix with a blue color scale

# Add color bar for reference
plt.colorbar(cax)

# Set axis ticks and labels
ax.set_xticks(np.arange(len(genre_labels)))
ax.set_yticks(np.arange(len(genre_labels)))
ax.set_xticklabels(genre_labels, rotation=45)
ax.set_yticklabels(genre_labels)

threshold = matrix.max() / 2 

# Annotate each cell in the matrix with the actual count (numbers)
for i in range(len(genre_labels)):
    for j in range(len(genre_labels)):
        color = "white" if matrix[i, j] > threshold else "black"
        ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color=color)

# Set axis labels
ax.set_xlabel("Predicted Genre", fontsize=14)
ax.set_ylabel("True Genre", fontsize=14)

# Save the figure
plt.savefig("output.jpg")

# Optionally show the plot
plt.show()

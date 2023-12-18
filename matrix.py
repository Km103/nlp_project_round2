import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def confusionMatrix(true_entities,text):

    true_labels=[elem[1] for elem in true_entities]
    predicted_labels = [ ent.label_ for ent in text.ents]
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def stats(logits):

    probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
    if probabilities.ndim > 1:
        probabilities = probabilities[0]
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    probabilities_normalized = probabilities / np.sum(probabilities)
    percentages = np.round(probabilities_normalized * 100, 2)
    norm = plt.Normalize(vmin=np.min(probabilities_normalized), vmax=np.max(probabilities_normalized))
    colors = plt.cm.coolwarm(norm(probabilities_normalized))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=probabilities_normalized, y=class_labels, palette=colors, orient='h', width=0.2, ax=ax)

    ax.tick_params(colors="white")  
    ax.xaxis.label.set_color("white") 
    ax.yaxis.label.set_color("white") 
    ax.title.set_color("white") 

    ax.set_title("Probabilities Window", fontsize=16)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Predictions")
    
    for i, percentage in enumerate(percentages):
        ax.text(probabilities_normalized[i] + 0.02, i, f'{percentage}%', va='center', fontsize=12, color='white')
    plt.show()

    print(f"Probabilities: {probabilities}")

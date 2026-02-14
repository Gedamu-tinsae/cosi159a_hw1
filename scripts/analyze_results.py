import numpy as np
from scipy import stats

def analyze_predictions(predictions, true_labels):
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    accuracy = np.mean(predictions == true_labels)
    n = len(predictions)
    ci_low, ci_upp = stats.binom.interval(0.95, n, accuracy, loc=0)
    ci_low /= n
    ci_upp /= n
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"95% CI: [{ci_low*100:.2f}%, {ci_upp*100:.2f}%]")

if __name__ == "__main__":
    preds = np.random.randint(0, 10, 1000)
    trues = np.random.randint(0, 10, 1000)
    analyze_predictions(preds, trues)

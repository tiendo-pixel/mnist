import gzip
import struct
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_mnist_images(filepath: str) -> np.ndarray:
    """
    Load MNIST images from a .gz file.

    Parameters:
    - filepath (str): Path to the .gz file containing MNIST images.

    Returns:
    - np.ndarray: Image array of shape (num_samples, 784) with values in range [0, 255].
    """
    with gzip.open(filepath, 'rb') as f:
        # Read magic number, number of images, rows, and columns (big-endian 32-bit integers)
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for MNIST images file.")
        
        # Read the raw byte buffer and convert to uint8 array
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8)
        
        # Reshape into (N, 784)
        return images.reshape(num_images, rows * cols)

def load_mnist_labels(filepath: str) -> np.ndarray:
    """
    Load MNIST labels from a .gz file.

    Parameters:
    - filepath (str): Path to the .gz file containing MNIST labels.

    Returns:
    - np.ndarray: Label array of shape (num_samples,) with values in range [0, 9].
    """
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and number of items (big-endian 32-bit integers)
        magic, num_items = struct.unpack('>II', f.read(8))
        
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for MNIST labels file.")
        
        # Read the raw byte buffer and convert to uint8 array
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8) 
        return labels

def load_mnist_data(data_dir: str = "."):
    """
    Load both training and testing datasets for MNIST.

    Parameters:
    - data_dir (str): Root directory containing the four .gz MNIST files.

    Returns:
    - tuple: (X_train, y_train, X_test, y_test)
    """
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)

    return X_train, y_train, X_test, y_test

# Assuming you have the MNIST dataset files in the current directory, you can load them as follows:
X_train, y_train, X_test, y_test = load_mnist_data(data_dir=".")
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score as the ratio of correctly predicted samples to total samples
    - Formula: Accuracy = Number of correct predictions / Total number of samples.
    - Returns: A float value between 0.0 (worst) and 1.0 (perfect classification).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def precision_score(y_true, y_pred, average: str = "macro"):
    """
    Calculate the precision score as the ratio of true positives to the sum of true positives and false positives
    - "macro" average: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    - None / binary: Calculate precision for the positive class (label 1) only. This is the default behavior.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true) if average == "macro" else [1]

    precision = []
    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))

        # Handle zero division by returning 0 if there are no positive predictions
        precision.append(true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0)
    return np.mean(precision) if average == "macro" else precision[0]

def recall_score(y_true, y_pred, average: str = "macro"):
    """
    Calculate the recall score as the ratio of true positives to the sum of true positives and false negative
    - y_true: Ground truth (correct) target values or labels.
    - y_pred: Estimated target labels returned by the classifier.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true) if average == "macro" else [1]
    recall = []
    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))

        # Handle zero division by returning 0 if there are no positive instances
        recall.append(true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0)
    return np.mean(recall) if average == "macro" else recall[0]

def f1_score(y_true, y_pred, average: str = "macro"):
    """
    Calculate the F1 score as the harmonic mean of precision and recall
    - Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    - Returns: A float score in range [0.0, 1.0] reflecting the model's ability to find all positive instances.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true) if average == "macro" else [1]

    f1_scores = []
    for cls in classes:
        precision = precision_score(y_true == cls, y_pred == cls, average=None)
        recall = recall_score(y_true == cls, y_pred == cls, average=None)
        f1_scores.append(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)
    return np.mean(f1_scores) if average == "macro" else f1_scores[0]

def classification_report(y_true, y_pred, digits: int = 2):
    """
    Build a comprehensive text report showing the main classification metrics for each class.
    - Parameters: y_true (actual labels) and y_pred (predicted labels) of equal length.
    - Returns: A formatted string summarizing precision, recall, f1-score, and support per class.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    report_data = {}
    
    # 1. Calculate metrics for each individual class
    for cls in classes:
        # Support is the number of actual occurrences of this class
        support = int(np.sum(y_true == cls))
        
        # Binary mask for the current class
        y_true_cls = (y_true == cls)
        y_pred_cls = (y_pred == cls)
        
        p = precision_score(y_true_cls, y_pred_cls, average=None)
        r = recall_score(y_true_cls, y_pred_cls, average=None)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        report_data[cls] = {
            "precision": p,
            "recall": r,
            "f1-score": f1,
            "support": support
        }

    # 2. Summary metrics (outside the loop)
    total_samples = len(y_true)
    accuracy = accuracy_score(y_true, y_pred)

    # Macro average (unweighted mean)
    macro_p = float(np.mean([report_data[cls]["precision"] for cls in classes]))
    macro_r = float(np.mean([report_data[cls]["recall"] for cls in classes]))
    macro_f1 = float(np.mean([report_data[cls]["f1-score"] for cls in classes]))

    # Weighted average (weighted by support of each class)
    weighted_p = sum(report_data[cls]["precision"] * report_data[cls]["support"] for cls in classes) / total_samples
    weighted_r = sum(report_data[cls]["recall"] * report_data[cls]["support"] for cls in classes) / total_samples
    weighted_f1 = sum(report_data[cls]["f1-score"] * report_data[cls]["support"] for cls in classes) / total_samples

    # 3. Format text output table
    col_width = digits + 6
    header_fmt = f"{{:>15}} {{:>{col_width}}} {{:>{col_width}}} {{:>{col_width}}} {{:>9}}\n"
    row_fmt = f"{{:>15}} {{:>{col_width}.{digits}f}} {{:>{col_width}.{digits}f}} {{:>{col_width}.{digits}f}} {{:>9}}\n"

    report_str = "\n"
    report_str += header_fmt.format("", "precision", "recall", "f1-score", "support")
    report_str += "\n"

    # Print rows for each digit (0 to 9)
    for cls in classes:
        d = report_data[cls]
        report_str += row_fmt.format(str(cls), d["precision"], d["recall"], d["f1-score"], d["support"])

    report_str += "\n"
    # Accuracy row
    acc_fmt = f"{{:>15}} {{:>{col_width}}} {{:>{col_width}}} {{:>{col_width}.{digits}f}} {{:>9}}\n"
    report_str += acc_fmt.format("accuracy", "", "", accuracy, total_samples)

    # Macro & Weighted avg rows
    report_str += row_fmt.format("macro avg", macro_p, macro_r, macro_f1, total_samples)
    report_str += row_fmt.format("weighted avg", weighted_p, weighted_r, weighted_f1, total_samples)

    return report_str

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix to evaluate the accuracy of a classification.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    num_classes = len(classes)

    # Initialize confusion matrix with zeros
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # fill confusion matrix: cm[actual, predicted]
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_cls) & (y_pred == pred_cls))
    return cm

class GaussianNB():
    def __init__(self, var_smoothing: float = 1e-3):
        """
        Gaussian Naive Bayes classifier.
        - var_smoothing: Portion of the largest variance added to variances for numerical stability.
        """
        self.var_smoothing = var_smoothing
        self.class_priors = None
        self.means = None
        self.variances = None
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Gaussian Naive Bayes model to the training data.
        - X: Training data of shape (num_samples, num_features).
        - y: Target labels of shape (num_samples,).
        """
        self.classes = np.unique(y)
        num_samples, num_features = X.shape
        num_classes = len(self.classes)

        self.class_priors = np.zeros(num_classes)
        self.means = np.zeros((num_classes, num_features))
        self.variances = np.zeros((num_classes, num_features))

        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.class_priors[idx] = X_cls.shape[0] / num_samples
            self.means[idx] = np.mean(X_cls, axis=0)
            # Add smoothing epsilon once to avoid zero variance
            self.variances[idx] = np.var(X_cls, axis=0) + self.var_smoothing

    def log_likelihood_calculations(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate log-likelihood ln P(X|class) using summation across 784 features.
        """
        num_samples = X.shape[0]
        num_classes = len(self.classes)
        log_likelihoods = np.zeros((num_samples, num_classes))
        
        for index in range(num_classes):
            mean = self.means[index]
            var = self.variances[index]
            
            # Log Gaussian PDF formula: -0.5 * ln(2 * pi * var) - ((X - mean)^2) / (2 * var)
            log_pdf = -0.5 * np.log(2.0 * np.pi * var) - ((X - mean) ** 2) / (2.0 * var)
            
            # Sum across all 784 features (prevents underflow)
            log_likelihoods[:, index] = np.sum(log_pdf, axis=1)

        return log_likelihoods

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate class probabilities using Softmax / Log-Sum-Exp.
        """
        # ln P(class | X) ~ ln P(X | class) + ln P(class)
        log_posteriors = self.log_likelihood_calculations(X) + np.log(self.class_priors)

        # Softmax with Log-Sum-Exp trick for numerical stability
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors - max_log)
        return exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        """
        # Pick the class with highest log-posterior
        log_posteriors = self.log_likelihood_calculations(X) + np.log(self.class_priors)
        return self.classes[np.argmax(log_posteriors, axis=1)]

# =====================================================================
# Workflow without dimensionality reduction (Full 784 features)
# =====================================================================

# 1. Load MNIST dataset from .gz files (shape: (N, 784))
X_train, y_train, X_test, y_test = load_mnist_data(data_dir=".")

# 2. Normalize pixel values from [0, 255] to [0.0, 1.0] for numerical stability
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

print(f"Training features shape: {X_train.shape}")  # (60000, 784)
print(f"Testing features shape:  {X_test.shape}")   # (10000, 784)

# 3. Initialize Gaussian Naive Bayes classifier
# var_smoothing adds a small variance value to corner pixels that are always zero
model = GaussianNB(var_smoothing=0.01)

# 4. Train the model on the full 784 features
print("\nTraining GaussianNB model...")
model.fit(X_train, y_train)
print("Training completed!")

# 5. Predict class labels for the test dataset
print("Evaluating on test set...")
y_pred = model.predict(X_test)

# 6. Compute evaluation metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

# 7. Print comprehensive classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# 8. Visualization of the confusion matrix (optional) and prediction results can be added here if needed.
# 8.1: Visualize predicted vs actual labels for a subset of test samples
plt.figure(figsize=(15, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    # Reshape the 784-feature vector back to 28x28 grayscale image
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    
    # Highlight title in green if correct, red if wrong
    is_correct = (y_test[i] == y_pred[i])
    title_color = "green" if is_correct else "red"
    
    plt.title(f"True: {y_test[i]} | Pred: {y_pred[i]}", 
              fontsize=12, fontweight='bold', color=title_color)
    plt.axis('off')

plt.suptitle("Sample Predictions (Green: Correct, Red: Incorrect)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# 2. Save directly to PDF in docs/figures/
output_dir = os.path.join("docs", "figures")
os.makedirs(output_dir, exist_ok=True)

pred_pdf_path = os.path.join(output_dir, "sample_predictions.pdf")
plt.savefig(pred_pdf_path, bbox_inches='tight')
print(f"Successfully saved sample predictions to: {pred_pdf_path}")

# 8.2: Visualize the confusion matrix for the predictions
cm = confusion_matrix(y_test, y_pred)
classes = np.unique(y_test)

# 8.2.1: Setup figure canvas
fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

# 8.2.2: Setup labels and ticks
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=11, fontweight='bold')
ax.set_yticklabels(classes, fontsize=11, fontweight='bold')

# 8.2.3:Titles and axis labels
ax.set_title("MNIST Naive Bayes - Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Predicted Digit", fontsize=12, labelpad=10)
ax.set_ylabel("True Digit", fontsize=12, labelpad=10)

# 8.2.4: Display numbers inside each cell
thresh = cm.max() / 2.0
for i in range(len(classes)):
    for j in range(len(classes)):
        val = cm[i, j]
        color = "white" if val > thresh else "black"
        ax.text(j, i, f"{val:,}", ha="center", va="center", 
                color=color, fontsize=10, fontweight='medium')
plt.tight_layout()

# =====================================================================
# Save to docs/figures/ directory
# =====================================================================
output_dir = os.path.join("docs", "figures")
os.makedirs(output_dir, exist_ok=True)  # Automatically creates directory if it doesn't exist

# 1. Save high-resolution PDF (vector format for reports)
pdf_path = os.path.join(output_dir, "confusion_matrix.pdf")
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Successfully saved PDF to: {pdf_path}")
plt.show()
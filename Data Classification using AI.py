from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42              
TEST_SIZE = 0.2                
CROSS_VAL_FOLDS = 5            
K_SEARCH_RANGE = range(1, 16) 
BANNER_WIDTH = 70
SHOW_PLOTS = True              
NEW_SAMPLE = [[5.0, 3.4, 1.5, 0.2]]
_PENDING_FIGURES: list = []

def print_header(title: str) -> None:
    """Print a bold section header for readable console output."""
    print("\n" + "=" * BANNER_WIDTH)
    print(title.center(BANNER_WIDTH))
    print("=" * BANNER_WIDTH)


def print_step(step_number: int, description: str) -> None:
    """Print a numbered pipeline step marker."""
    print(f"\n[Step {step_number}] {description}")
    print("-" * BANNER_WIDTH)

def load_and_explore_dataset() -> tuple[pd.DataFrame, pd.Series, list[str], np.ndarray]:
    """Load the Iris dataset and print a summary so the data is
    understood before any modeling begins.
    """
    print_step(1, "Loading and understanding the dataset")

    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    labels = pd.Series(iris.target, name="species")
    target_names = iris.target_names

    print(f"Samples loaded       : {features.shape[0]}")
    print(f"Features per sample  : {features.shape[1]} -> {list(features.columns)}")
    print(f"Target classes       : {len(target_names)} -> {list(target_names)}")

    print("\nClass distribution (checking the dataset is balanced):")
    class_counts = labels.value_counts().sort_index()
    for class_index, count in class_counts.items():
        print(f"  {target_names[class_index]:<12} : {count} samples")

    print("\nFirst 5 rows of the dataset:")
    preview = features.copy()
    preview["species"] = labels.map(lambda i: target_names[i])
    print(preview.head().to_string(index=False))

    print("\nStatistical summary of the features:")
    print(features.describe().round(2).to_string())

    return features, labels, list(features.columns), target_names

def visualize_dataset(features: pd.DataFrame, labels: pd.Series, target_names: np.ndarray) -> None:
    """Build a pairplot showing how the three species separate across
    every pair of features. This is purely exploratory -- it runs
    before any scaling or modeling and helps justify why a
    distance-based classifier like KNN works well on this data.
    """
    print_step(2, "Visualizing feature relationships across species")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed -- skipping visualization.")
        return

    plot_data = features.copy()
    plot_data["Species"] = labels.map(lambda i: target_names[i].capitalize())

    sns.set_style("whitegrid")
    pairplot = sns.pairplot(
        plot_data,
        hue="Species",
        palette="Set2",
        diag_kind="kde",
        plot_kws={"alpha": 0.75, "s": 35, "edgecolor": "white", "linewidth": 0.3},
        height=1.9,
    )
    pairplot.figure.suptitle("Iris Feature Relationships by Species", y=1.02, fontsize=14, weight="bold")
    pairplot.figure.canvas.manager.set_window_title("Feature Pairplot")
    pairplot.savefig("feature_pairplot.png", dpi=150, bbox_inches="tight")
    print("Saved feature relationship chart to 'feature_pairplot.png'.")
    _PENDING_FIGURES.append(pairplot.figure)

def preprocess_and_split(
    features: pd.DataFrame, labels: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """Scale features to a common range and split the data into
    training and testing sets.

    Scaling matters for KNN specifically because it is a distance-based
    algorithm -- without it, features with larger numeric ranges would
    unfairly dominate the distance calculation.
    """
    print_step(3, "Preprocessing: feature scaling and train/test split")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("Applied StandardScaler -> every feature now has mean 0, variance 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        scaled_features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,          
        stratify=labels,       
    )

    print(f"Training samples     : {x_train.shape[0]} ({(1 - TEST_SIZE) * 100:.0f}%)")
    print(f"Testing samples      : {x_test.shape[0]} ({TEST_SIZE * 100:.0f}%)")
    print("Split is stratified   -> class balance preserved in both sets.")

    return x_train, x_test, y_train, y_test, scaler

def find_optimal_k(x_train: np.ndarray, y_train: pd.Series) -> int:
    """Search a range of K values using cross-validation on the
    training set only, and return the K with the highest average
    accuracy. Using cross-validation here (instead of the test set)
    avoids leaking test data into model selection.
    """
    print_step(4, "Tuning the model: searching for the best K")

    print(f"{'K':>3} | {'Mean CV Accuracy':>17}")
    print("-" * 24)

    best_k, best_score = 1, 0.0
    for k in K_SEARCH_RANGE:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, x_train, y_train, cv=CROSS_VAL_FOLDS)
        mean_score = scores.mean()
        print(f"{k:>3} | {mean_score:>16.4f}")

        if mean_score > best_score:
            best_k, best_score = k, mean_score

    print(f"\nSelected K = {best_k}  (mean cross-validation accuracy: {best_score:.4f})")
    return best_k

def train_model(x_train: np.ndarray, y_train: pd.Series, k: int) -> KNeighborsClassifier:
    """Train the final KNN classifier on the full training set."""
    print_step(5, f"Training the final KNN classifier (K={k})")

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    print("Model training complete -- the classifier has memorized the training map.")
    return model

def evaluate_model(
    model: KNeighborsClassifier,
    x_test: np.ndarray,
    y_test: pd.Series,
    target_names: np.ndarray,
) -> None:
    """Predict on the held-out test set and report predictions,
    accuracy, a confusion matrix, and a full precision/recall/F1
    breakdown.
    """
    print_step(6, "Predicting on unseen test data and evaluating performance")

    predictions = model.predict(x_test)

    print("First 10 predictions :", [int(value) for value in predictions[:10]])
    print("First 10 actual vals :", [int(value) for value in np.asarray(y_test)[:10]])

    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    print(f"\nAccuracy score        : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"Weighted F1 score      : {weighted_f1:.4f}")
    print("(Accuracy alone can be misleading on imbalanced data, so the")
    print(" F1 score and full report below give a more complete picture.)")

    matrix = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"Actual: {name}" for name in target_names],
        columns=[f"Pred: {name}" for name in target_names],
    )
    print(matrix_frame.to_string())

    print("\nClassification Report (precision / recall / F1 per class):")
    print(classification_report(y_test, predictions, target_names=target_names))

    _plot_confusion_matrix(matrix, target_names, accuracy)


def _plot_confusion_matrix(matrix: np.ndarray, target_names: np.ndarray, accuracy: float) -> None:
    """Render a clean, annotated confusion matrix heatmap and queue it
    for display alongside the other figures.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed -- skipping confusion matrix chart.")
        return

    labels = [name.capitalize() for name in target_names]

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 14, "weight": "bold"},
        ax=axis,
    )
    axis.set_title(f"Confusion Matrix -- KNN on Iris Test Set\nAccuracy: {accuracy * 100:.2f}%",
                    fontsize=13, weight="bold", pad=14)
    axis.set_xlabel("Predicted Species", fontsize=11)
    axis.set_ylabel("Actual Species", fontsize=11)
    figure.canvas.manager.set_window_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved confusion matrix chart to 'confusion_matrix.png'.")
    _PENDING_FIGURES.append(figure)

def predict_new_sample(
    model: KNeighborsClassifier,
    scaler: StandardScaler,
    feature_names: list[str],
    target_names: np.ndarray,
) -> None:
    """Demonstrate the trained model classifying a brand-new sample
    that was never part of the training or test data.
    """
    print_step(7, "Classifying a brand-new, unseen sample")

    sample_frame = pd.DataFrame(NEW_SAMPLE, columns=feature_names)
    print("New measurement:")
    print(sample_frame.to_string(index=False))

    scaled_sample = scaler.transform(sample_frame)
    predicted_class = model.predict(scaled_sample)[0]
    probabilities = model.predict_proba(scaled_sample)[0]

    print(f"\nPredicted species     : {target_names[predicted_class]}")
    print("Class probabilities   :")
    for name, probability in zip(target_names, probabilities):
        print(f"  {name:<12} : {probability * 100:5.1f}%")

def main() -> None:
    """Run the complete supervised classification pipeline end to end."""
    print_header("DATA CLASSIFICATION USING AI -- KNN on the Iris Dataset")

    features, labels, feature_names, target_names = load_and_explore_dataset()
    visualize_dataset(features, labels, target_names)
    x_train, x_test, y_train, y_test, scaler = preprocess_and_split(features, labels)
    best_k = find_optimal_k(x_train, y_train)
    model = train_model(x_train, y_train, best_k)
    evaluate_model(model, x_test, y_test, target_names)
    predict_new_sample(model, scaler, feature_names, target_names)

    print_header("PIPELINE COMPLETE")
    print(
        "The model was trained on labeled data, validated on unseen test\n"
        "data, and used to classify a brand-new sample -- demonstrating the\n"
        "full supervised learning workflow from raw data to prediction.\n"
    )

    if SHOW_PLOTS and _PENDING_FIGURES:
        try:
            import matplotlib.pyplot as plt
            print("Displaying charts... close the chart windows to end the program.")
            plt.show()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
K-Nearest Neighbors (KNN) Classifier using Iris Dataset

- Understand and implement KNN for classification.
- Evaluate the performance of the model with different values of `K`.
- Visualize how the KNN algorithm creates decision boundaries.

Tools Used

- Python
- Pandas
- Scikit-learn (sklearn)
- Matplotlib
- NumPy

Dataset

We use the `Iris.csv` file which contains:
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **Target**: Species (Setosa, Versicolor, Virginica)


How It Works

1. **Load the dataset** from CSV.
2. **Preprocess and normalize** features using `StandardScaler`.
3. **Split** the dataset into training and testing sets.
4. **Train** the KNN model with multiple values of `K`.
5. **Evaluate** accuracy and plot the confusion matrix.
6. **Visualize decision boundaries** using PCA for 2D projection.


Results

- Tried `K = 1, 3, 5, 7, 9`
- Accuracy varies depending on `K`.
- Confusion matrix helps us understand the model’s performance.
- PCA was used to reduce dimensions and visualize decision regions.

Example Output

- Accuracy for K=3: ~97%
- Confusion matrix showing model predictions.
- 2D decision boundary plot that shows how the model classifies the points.

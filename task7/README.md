Breast Cancer Classification using SVM (Linear & Non-Linear)

Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset which contains 569 samples of benign and malignant cell nuclei, with 30 numerical features per sample.

- `M` = Malignant (cancerous)
- `B` = Benign (non-cancerous)

Tools & Libraries

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Steps Performed

1. Data Preparation
- Removed unnecessary columns (`id`)
- Encoded `diagnosis` as 0 (Benign) and 1 (Malignant)
- Selected 2 features: `radius_mean`, `texture_mean`
- Scaled features using `StandardScaler`

2. Model Training
- Trained two SVM models:
  - **Linear SVM**
  - **SVM with RBF (Radial Basis Function) kernel**

3. Visualization
- Plotted decision boundaries for both models in 2D
- Shows how RBF kernel handles complex (non-linear) boundaries

4. Hyperparameter Tuning
- (Optional: included in extended version)
- Use `GridSearchCV` for tuning `C` and `gamma` for RBF kernel

Results

- **Linear SVM** gives a straight boundary
- **RBF SVM** gives curved boundaries, better at fitting real-world data



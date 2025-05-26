import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif # For feature importance

# 1. Fetch the dataset
print("Fetching Heart Disease Dataset...")
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# For the target variable, let's ensure it's binary (0 for no disease, 1 for disease)
# The original dataset might have values 0, 1, 2, 3, 4 for 'num' (diagnosis of heart disease).
# We'll map anything > 0 to 1 (presence of heart disease).
y = y['num'].apply(lambda x: 1 if x > 0 else 0)

print("Dataset fetched successfully.")
print("\nOriginal X head:")
print(X.head())
print("\nOriginal y head:")
print(y.head())

# 2. Preprocessing
print("\nStarting data preprocessing...")

# Check for missing values
print("\nMissing values before handling:")
print(X.isnull().sum())

for col in ['ca', 'thal']:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype == 'object' or X[col].nunique() < 10: # Treat as categorical if few unique values
            X[col] = X[col].fillna(X[col].mode()[0]) # Mode for categorical
        else:
            X[col] = X[col].fillna(X[col].median()) # Median for numerical

print("\nMissing values after handling:")
print(X.isnull().sum())
print("\nX info after missing value handling:")
X.info()

# Identify numerical and categorical features
# We'll use the 'variable information' to guide this.
# According to UCI metadata:
# Numerical: age, trestbps, chol, thalach, oldpeak, (ca, thal - after converting to numeric)
# Categorical: sex, cp, fbs, restecg, exang, slope
# 'thal' and 'ca' can be treated as categorical or ordinal depending on interpretation.
# For simplicity and given the typical values, we'll treat them as numerical but watch their unique values.

numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

print("\nUnique values for 'ca':", X['ca'].unique())
print("Unique values for 'thal':", X['thal'].unique())

if X['ca'].nunique() <= 5: # Small number of unique integer values often indicates categorical/ordinal
    categorical_features.append('ca')
else:
    numerical_features.append('ca')

if X['thal'].nunique() <= 5: # Small number of unique integer values often indicates categorical/ordinal
    categorical_features.append('thal')
else:
    numerical_features.append('thal')

# Remove duplicates from lists if any
numerical_features = list(set(numerical_features) - set(categorical_features))
categorical_features = list(set(categorical_features) - set(numerical_features))


# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler() # Scaling for numerical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # One-hot encoding for categorical

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any (though there shouldn't be for this dataset)
)

print(f"\nNumerical features for scaling: {numerical_features}")
print(f"Categorical features for one-hot encoding: {categorical_features}")

# 3. Feature Importance (before splitting to see raw importance, but selection happens after split)
# We'll use a simple method like SelectKBest with f_classif to get an initial idea of importance.
# This step is mostly for insight, not for direct feature selection before scaling/encoding.
# Feature importance on raw data before preprocessing might be misleading for some features.
# A more robust approach would be to calculate importance on preprocessed data or use
# tree-based models (like RandomForest) for importance.
# For KNN, we generally rely on scaling and don't do explicit feature selection unless performance is an issue.

print("\nAssessing initial feature importance (using f_classif):")

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nData split: Training samples = {len(X_train)}, Testing samples = {len(X_test)}")

# 5. Create a pipeline for preprocessing and KNN
# This ensures that preprocessing steps are consistently applied to both training and test data
# and avoids data leakage.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', KNeighborsClassifier())])

# 6. Hyperparameter Tuning (Finding the best 'k') using GridSearchCV
# We'll also consider 'weights' (uniform vs. distance) and 'metric'.
print("\nStarting hyperparameter tuning for KNN using GridSearchCV...")

param_grid = {
    'classifier__n_neighbors': np.arange(1, 21), # Test k from 1 to 20
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan'] # Common distance metrics for KNN
}

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
# n_jobs=-1 uses all available CPU cores for parallel processing
# cv=10 means 10-fold cross-validation

grid_search.fit(X_train, y_train)

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Get the best model
best_knn_model = grid_search.best_estimator_

# 7. Evaluate the Best Model on the Test Set
print("\nEvaluating the best KNN model on the test set...")
y_pred = best_knn_model.predict(X_test)
y_prob = best_knn_model.predict_proba(X_test)[:, 1] # Probabilities for ROC curve

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


print("\n--- Interpreting Feature Importance for KNN ---")
print("KNN does not inherently provide direct feature importance scores like tree-based models.")
print("Its 'importance' comes from how features contribute to distance in the scaled space.")
print("Features that significantly differentiate between classes (even after scaling) are 'important'.")
print("We applied StandardScaler, which helps all numerical features contribute fairly based on their distribution.")
print("One-Hot Encoding for categorical features creates new binary features, allowing them to contribute to distance.")


try:
    from sklearn.inspection import permutation_importance

    print("\nCalculating Permutation Importance (may take a moment)...")

    # Fit the preprocessor separately to get feature names after transformation
    preprocessor_fitted = preprocessor.fit(X_train)
    transformed_X_train = preprocessor_fitted.transform(X_train)
    transformed_X_test = preprocessor_fitted.transform(X_test)

    # Get feature names after one-hot encoding
    feature_names = numerical_features
    if 'cat' in preprocessor_fitted.named_transformers_:
        ohe_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names.extend(ohe_feature_names)

    # Re-train the best KNN classifier on the *transformed* training data
    # (since permutation_importance expects a fitted model on transformed data)
    final_knn_classifier = grid_search.best_estimator_.named_steps['classifier']
    final_knn_classifier.fit(transformed_X_train, y_train)


    result = permutation_importance(final_knn_classifier, transformed_X_test, y_test,
                                    n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    # Create a DataFrame for better visualization
    perm_importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in sorted_idx],
        'Importance Mean': result.importances_mean[sorted_idx],
        'Importance Std': result.importances_std[sorted_idx]
    })
    print("\nPermutation Importance (on test set):")
    print(perm_importance_df.tail(10)) # Display top 10 important features

    # Plotting permutation importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=[feature_names[i] for i in sorted_idx])
    ax.set_title("Permutation Importances (Test set)")
    fig.tight_layout()
    plt.show()

except ImportError:
    print("Install `scikit-learn` version 0.24 or later for `permutation_importance`.")
    print("Alternatively, install `eli5` (`pip install eli5`) for another permutation importance method.")
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your dataset from the CSV file
df = pd.read_csv('sentiment_analysis.csv')  # Replace with your actual CSV file path

# Debugging: Print DataFrame shape and columns
print("Original dataset shape:", df.shape)
print("Columns in the DataFrame:", df.columns)

# 2. Clean the data (e.g., remove any missing or incorrect data)
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

df.dropna(inplace=True)  # Remove missing values if any
print("Dataset shape after dropping NAs:", df.shape)  # Check shape after dropping NAs

# 3. Encode sentiment labels into numbers
if 'Sentiment' in df.columns:
    label_encoder = LabelEncoder()
    df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])
else:
    print("Error: 'Sentiment' column not found in the DataFrame.")

# 4. Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# 5. Transform text data using TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)  # Convert text to numerical vectors
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# 6. Train Logistic Regression model with a high number of iterations (simulating epochs)
model = LogisticRegression(max_iter=500)  # Increased max_iter to ensure convergence
model.fit(X_train_tfidf, y_train)

# 7. Evaluate the model on test data
y_pred = model.predict(X_test_tfidf)

# 8. Print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 9. Print classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 10. Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Use seaborn to plot a heatmap of the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

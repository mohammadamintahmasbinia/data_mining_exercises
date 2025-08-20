import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
import nltk

# دانلود stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ------------------------
# 1. خواندن دیتاست
# ------------------------
# دیتاست Sentiment140 دارای ستون‌های زیر است:
# 0 - target (0 = negative, 2 = neutral, 4 = positive)
# 5 - text (توییت)
df = pd.read_csv("training.1600000.processed.noemoticon.csv", 
                 encoding="latin-1", header=None)

df = df[[0,5]]
df.columns = ["sentiment", "text"]

# نگاشت labelها
df["sentiment"] = df["sentiment"].replace({0:0, 2:1, 4:2})  # 0=negative,1=neutral,2=positive

# برای تست سریع فقط 100000 رکورد استفاده می‌کنیم (اختیاری)
df = df.sample(100000, random_state=42)

# ------------------------
# 2. پاکسازی متن
# ------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # حذف URL
    text = re.sub(r"@\w+", "", text)  # حذف منشن
    text = re.sub(r"#", "", text)  # حذف هشتگ
    text = re.sub(r"[^a-z\s]", "", text)  # حذف عدد و علائم
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# ------------------------
# 3. تقسیم داده
# ------------------------
X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------
# 4. بردارسازی با TF-IDF
# ------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)  # unigram + bigram
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------
# 5. بالانس داده‌ها با SMOTE
# ------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)

# ------------------------
# 6. مدل Naive Bayes
# ------------------------
model = MultinomialNB(alpha=1.0)
model.fit(X_train_bal, y_train_bal)

# ------------------------
# 7. پیش‌بینی و ارزیابی
# ------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

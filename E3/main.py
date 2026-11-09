import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ===============================
# 1. بارگذاری داده‌ها (نسخه CSV)
# ===============================
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")  # مسیر فایل CSV

# ===============================
# 2. پیش‌پردازش داده‌ها
# ===============================
df = df.dropna(subset=["CustomerID"])   # حذف ردیف‌هایی که CustomerID ندارند
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]  # حذف مقادیر منفی یا صفر

# ایجاد ستون مبلغ کل خرید
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# ===============================
# 3. استخراج ویژگی‌ها برای هر مشتری
# ===============================
features = df.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",        # تعداد خریدها
    "Quantity": "sum",             # تعداد کل کالاها
    "TotalPrice": "sum",           # مجموع هزینه
    "InvoiceDate": ["min", "max"]  # تاریخ اولین و آخرین خرید
})

# تخت کردن MultiIndex ستون‌ها
features.columns = ["NumPurchases", "TotalQuantity", "TotalSpent", "FirstPurchase", "LastPurchase"]

# محاسبه میانگین فاصله زمانی بین خریدها
features["PurchasePeriod"] = (pd.to_datetime(features["LastPurchase"]) - 
                              pd.to_datetime(features["FirstPurchase"])).dt.days / features["NumPurchases"]

# پر کردن مقادیر NaN
features = features.fillna(0)

# ===============================
# 4. آماده‌سازی داده برای خوشه‌بندی
# ===============================
X = features[["NumPurchases", "TotalQuantity", "TotalSpent", "PurchasePeriod"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 5. اجرای الگوریتم DBSCAN
# ===============================
dbscan = DBSCAN(eps=0.65, min_samples=5)  # مقدار eps و min_samples باید بهینه‌سازی بشه
labels = dbscan.fit_predict(X_scaled)

features["Cluster"] = labels

# ===============================
# 6. ارزیابی کیفیت خوشه‌ها
# ===============================
mask = labels != -1  # خوشه -1 نویز هست
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

if n_clusters > 1 and np.sum(mask) > 1:
    sil_score = silhouette_score(X_scaled[mask], labels[mask])
    db_score = davies_bouldin_score(X_scaled[mask], labels[mask])
    print("Silhouette Score:", sil_score)
    print("Davies-Bouldin Index:", db_score)
else:
    print("⚠️ خوشه‌بندی معتبر به‌دست نیامده (DBSCAN همه داده‌ها رو تو یک خوشه یا نویز انداخت).")

# ===============================
# 7. مصورسازی خوشه‌ها
# ===============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="tab10", s=50)
plt.title("DBSCAN Clustering of Customers")
plt.show()

# ===============================
# 8. تحلیل خوشه‌ها
# ===============================
print(features.groupby("Cluster")[["NumPurchases", "TotalQuantity", "TotalSpent", "PurchasePeriod"]].mean())




#----------------------------------------

# from sklearn.neighbors import NearestNeighbors

# min_pts = 5
# neigh = NearestNeighbors(n_neighbors=min_pts)
# nbrs = neigh.fit(X_scaled)
# distances, indices = nbrs.kneighbors(X_scaled)

# distances = np.sort(distances[:, min_pts-1])

# plt.figure(figsize=(8,5))
# plt.plot(distances)
# plt.title("k-distance Graph (برای انتخاب eps)")
# plt.xlabel("Data Points مرتب‌شده")
# plt.ylabel(f"{min_pts}-th Nearest Neighbor Distance")
# plt.show()

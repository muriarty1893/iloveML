# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN sınıflandırıcısı oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = knn.predict(X_test)

# Modelin doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy:.2f}")

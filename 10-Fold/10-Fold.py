# Gerekli kütüphanelerin import edilmesi
import pandas as pd  # Veri analizi ve veri manipülasyonu için
import numpy as np  # Bilimsel hesaplamalar için
import matplotlib.pyplot as plt  # Grafik çizimi için
import seaborn as sns  # İstatistiksel veri görselleştirme için
from sklearn.model_selection import train_test_split, KFold  # Veri bölme ve çapraz doğrulama için
from sklearn.preprocessing import StandardScaler  # Verileri ölçeklendirmek için
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix  # Değerlendirme metrikleri
from tensorflow.keras.models import Sequential  # Keras ile sıralı model oluşturma
from tensorflow.keras.layers import Dense  # Yapay sinir ağı katmanları
from tensorflow.keras.optimizers import Adam, SGD  # Optimizasyon algoritmaları
from tensorflow.keras.utils import plot_model  # Modelin mimarisini görselleştirme

# Veri setinin yüklenmesi
df = pd.read_csv('breast_cancer_data.csv')  # CSV dosyasından veri yükleme
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)  # 'id' ve 'Unnamed: 32' gibi gereksiz sütunlar siliniyor

# Giriş (X) ve hedef (y) değişkenlerinin oluşturulması
X = df.drop(['diagnosis'], axis=1)  # 'diagnosis' sütunu hedef değişken olarak ayrılır
y = df['diagnosis'].map({'B': 1, 'M': 0})  # 'B' (Benign) = 1 ve 'M' (Malignant) = 0 olarak etiketlenir

# Verilerin normalize edilmesi
scaler = StandardScaler()  # StandardScaler nesnesi oluşturulur
X_scaled = scaler.fit_transform(X)  # Giriş verileri ölçeklendirilir

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# y_train numpy dizisine dönüştürülür
y_train = y_train.to_numpy()

# Model oluşturma fonksiyonu tanımlanır
def create_model(hidden_layer_sizes=(64, 32), activation='relu', optimizer='adam'):
    model = Sequential()  # Boş bir sıralı model oluşturulur
    model.add(Dense(hidden_layer_sizes[0], input_dim=X_train.shape[1], activation=activation))  # Giriş katmanı
    model.add(Dense(hidden_layer_sizes[1], activation=activation))  # Gizli katman
    model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı, sigmoid ile iki sınıflı çıktı
    # Optimizasyon algoritması seçilir
    if optimizer == 'adam':
        opt = Adam()
    else:
        opt = SGD()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # Model derlenir
    model.save(f'model{hidden_layer_size}.h5') # modeli kaydermek için
    return model

# 10 katlı çapraz doğrulama (KFold)
kf10 = KFold(n_splits=10, shuffle=True, random_state=42)

# Farklı model konfigürasyonları tanımlanır
models = [
    {'name': 'Model 1', 'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'optimizer': 'adam'},
    {'name': 'Model 2', 'hidden_layer_sizes': (128, 64), 'activation': 'relu', 'optimizer': 'sgd'},
    {'name': 'Model 3', 'hidden_layer_sizes': (64, 64), 'activation': 'tanh', 'optimizer': 'adam'}
]

# Her bir modelin değerlendirilmesi
for model_info in models:
    model_name = model_info['name']
    hidden_layer_size = model_info['hidden_layer_sizes']
    activation = model_info['activation']
    optimizer = model_info['optimizer']
    
    print(f"\nEvaluating {model_name} with hidden_layer_size={hidden_layer_size}, activation={activation}, optimizer={optimizer}")
    
    model = create_model(hidden_layer_size, activation, optimizer)
    print(model.summary())  # Modelin yapısı özetlenir
    # Model mimarisi görselleştirilir ve bir dosyaya kaydedilir
    plot_model(model, to_file=f'{model_name}_architecture.png', show_shapes=True, show_layer_names=True)
    print(f"Saved architecture for {model_name} as {model_name}_architecture.png")
    
    # Çapraz doğrulama başlatılır
    accuracies = []
    for train_index, val_index in kf10.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Model eğitilir
        model.fit(X_train_fold, y_train_fold, batch_size=32, epochs=50, verbose=0)
        
        # Doğrulama setinde tahmin yapılır
        y_val_pred = (model.predict(X_val_fold) > 0.5).astype(int)
        
        # Doğrulama setindeki doğruluk hesaplanır
        accuracies.append(accuracy_score(y_val_fold, y_val_pred))
    
    # Çapraz doğrulama doğruluğu ekrana yazdırılır
    print(f"Cross-validation accuracy for {model_name}: {np.mean(accuracies):.4f}")
    
    # Model tüm eğitim verisi ile eğitilir ve test setinde değerlendirilir
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Test seti üzerinde performans metrikleri hesaplanır
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Sınıflandırma raporu ekrana yazdırılır
    print(f"\n{model_name} - Classification Report:\n", classification_report(y_test, y_pred))

    # Karışıklık matrisi görselleştirilir
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()  # Grafik gösterilir

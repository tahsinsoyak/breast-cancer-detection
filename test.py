# Gerekli kütüphaneleri içe aktarır
import pandas as pd  # Veri işleme ve analiz için kullanılır
import numpy as np  # Sayısal hesaplamalar için kullanılır
import matplotlib.pyplot as plt  # Grafik çizmek için kullanılır
import seaborn as sns  # Gelişmiş grafikler oluşturmak için kullanılır
from sklearn.model_selection import train_test_split, KFold  # Veri bölme ve çapraz doğrulama için kullanılır
from sklearn.preprocessing import StandardScaler  # Veriyi ölçeklendirmek için kullanılır
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix  # Değerlendirme metrikleri
from tensorflow.keras.models import Sequential  # Sıralı model oluşturmak için kullanılır
from tensorflow.keras.layers import Dense  # Tam bağlı (dense) katmanlar oluşturmak için kullanılır
from tensorflow.keras.optimizers import Adam, SGD  # Optimizasyon algoritmaları
from tensorflow.keras.utils import plot_model  # Model mimarisini görselleştirmek için kullanılır

# Veri setini yükler
df = pd.read_csv('breast_cancer_data.csv')  # Göğüs kanseri veri setini okur
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)  # Gereksiz sütunları kaldırır
X = df.drop(['diagnosis'], axis=1)  # Hedef sütunu (diagnosis) çıkarır, kalanlar girdi (feature) olarak kullanılır
y = df['diagnosis'].map({'B': 1, 'M': 0})  # 'B' (iyi huylu) için 1, 'M' (kötü huylu) için 0 değerlerini atar

# Veriyi normalize eder (ölçeklendirir)
scaler = StandardScaler()  # StandardScaler ile veriyi ölçeklendirir
X_scaled = scaler.fit_transform(X)  # Giriş özelliklerini ölçeklendirir

# Veriyi eğitim ve test setlerine ayırır (66% eğitim, 34% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.34, random_state=42)

# y_train ve y_test dizilerini numpy array'ine dönüştürür
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Model oluşturma fonksiyonu
def create_model(hidden_layer_sizes=(64, 32), activation='relu', optimizer='adam'):
    model = Sequential()  # Sıralı bir model oluşturur
    model.add(Dense(hidden_layer_sizes[0], input_dim=X_train.shape[1], activation=activation))  # İlk gizli katman (giriş boyutu X_train'in sütun sayısı)
    model.add(Dense(hidden_layer_sizes[1], activation=activation))  # İkinci gizli katman
    model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı (sigmoid aktivasyon, ikili sınıflandırma için)
    
    # Optimizasyon algoritmasını seçer
    if optimizer == 'adam':
        opt = Adam()
    else:
        opt = SGD()
    
    # Modeli derler
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # İkili çapraz entropi kaybı kullanılır
    model.save(f'model{hidden_layer_size}.h5') # modeli kaydermek için
    return model

# Test edilecek modelleri tanımlar
models = [
    {'name': 'Model 1', 'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'optimizer': 'adam'},
    {'name': 'Model 2', 'hidden_layer_sizes': (128, 64), 'activation': 'relu', 'optimizer': 'sgd'},
    {'name': 'Model 3', 'hidden_layer_sizes': (64, 64), 'activation': 'tanh', 'optimizer': 'adam'}
]

# Modelleri KFold çapraz doğrulama ile değerlendirir
for model_info in models:
    model_name = model_info['name']
    hidden_layer_size = model_info['hidden_layer_sizes']
    activation = model_info['activation']
    optimizer = model_info['optimizer']
    
    print(f"\nEvaluating {model_name} with hidden_layer_size={hidden_layer_size}, activation={activation}, optimizer={optimizer}")
    
    # Modeli oluşturur
    model = create_model(hidden_layer_size, activation, optimizer)
    print(model.summary())  # Modelin özetini yazdırır
    
    # Model mimarisini görselleştirir ve dosyaya kaydeder
    plot_model(model, to_file=f'{model_name}_architecture.png', show_shapes=True, show_layer_names=True)
    print(f"Saved architecture for {model_name} as {model_name}_architecture.png")
    
    # 5 katlı çapraz doğrulama tanımlanır
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []  # Doğruluk skorlarını toplamak için liste
    
    for train_index, val_index in kf5.split(X_train):
        # Eğitim ve doğrulama setlerini ayırır
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Modeli eğitir
        model.fit(X_train_fold, y_train_fold, batch_size=32, epochs=50, verbose=0)
        
        # Doğrulama setinde tahmin yapar
        y_val_pred = (model.predict(X_val_fold) > 0.5).astype(int)
        
        # Doğruluğu hesaplar ve listeye ekler
        accuracies.append(accuracy_score(y_val_fold, y_val_pred))
    
    print(f"Cross-validation accuracy for {model_name}: {np.mean(accuracies):.4f}")
    
    # Modeli tüm eğitim verisi üzerinde eğitir ve test seti üzerinde değerlendirir
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # Test setinde tahmin yapar
    
    # Değerlendirme metriklerini hesaplar
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Sonuçları yazdırır
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Sınıflandırma raporunu yazdırır
    print(f"\n{model_name} - Classification Report:\n", classification_report(y_test, y_pred))

    # Karmaşıklık matrisini çizer ve gösterir
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

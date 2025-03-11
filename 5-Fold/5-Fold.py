import pandas as pd  # Veri manipülasyonu ve analiz için kullanılır.
import numpy as np  # Sayısal hesaplamalar için kullanılır.
import matplotlib.pyplot as plt  # Grafik ve görselleştirme için kullanılır.
import seaborn as sns  # Gelişmiş görselleştirme için kullanılır.
from sklearn.model_selection import train_test_split, KFold  # Veriyi bölme ve çapraz doğrulama işlemleri için.
from sklearn.preprocessing import StandardScaler  # Veriyi standartlaştırmak için.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix  # Sınıflandırma metrikleri için.
from tensorflow.keras.models import Sequential  # Keras ile ardışık model oluşturmak için.
from tensorflow.keras.layers import Dense  # Tam bağlı (dense) katmanları oluşturmak için.
from tensorflow.keras.optimizers import Adam, SGD  # Optimizasyon algoritmaları için.
from tensorflow.keras.utils import plot_model  # Model mimarisini görselleştirmek için.

#Veri Setinin Yüklenmesi ve Ön İşleme
df = pd.read_csv('breast_cancer_data.csv')  # CSV dosyasından veri setini yükler.
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)  # Gereksiz sütunları kaldırır ('id' ve 'Unnamed: 32').
X = df.drop(['diagnosis'], axis=1)  # 'diagnosis' sütununu hedef değişken olarak ayırır, geri kalanlar özellik (feature) olur.
y = df['diagnosis'].map({'B': 1, 'M': 0})  # 'diagnosis' sütunundaki değerleri sayısal hale getirir. Benign: 1, Malignant: 0.

#Veriyi Normalize Etme
scaler = StandardScaler()  # StandardScaler nesnesi oluşturur.
X_scaled = scaler.fit_transform(X)  # Veriyi standartlaştırır (ortalama 0, varyans 1 olacak şekilde).

#Eğitim ve Test Verisinin Ayrılması
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # Veriyi eğitim (%80) ve test (%20) olarak ayırır.
y_train = y_train.to_numpy()  # y_train'i numpy dizisine dönüştürür.

#Model Oluşturma Fonksiyonu
def create_model(hidden_layer_sizes=(64, 32), activation='relu', optimizer='adam'):
    model = Sequential()  # Ardışık model oluşturur.
    model.add(Dense(hidden_layer_sizes[0], input_dim=X_train.shape[1], activation=activation))  # Giriş katmanı.
    model.add(Dense(hidden_layer_sizes[1], activation=activation))  # Gizli katman.
    model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı (sigmoid aktivasyonu ile, çünkü ikili sınıflandırma yapılıyor).
    
    if optimizer == 'adam':
        opt = Adam()  # Adam optimizatörü kullanılır.
    else:
        opt = SGD()  # SGD optimizatörü kullanılır.
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # Model derlenir, binary cross-entropy kaybı ve doğruluk metriği kullanılır.
    model.save(f'model{hidden_layer_size}.h5') # modeli kaydermek için
    return model


# K-Fold Çapraz Doğrulama ve Model Değerlendirme
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 katlı çapraz doğrulama nesnesi oluşturur.

# Farklı Model Konfigürasyonları
models = [
    {'name': 'Model 1', 'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'optimizer': 'adam'},
    {'name': 'Model 2', 'hidden_layer_sizes': (128, 64), 'activation': 'relu', 'optimizer': 'sgd'},
    {'name': 'Model 3', 'hidden_layer_sizes': (64, 64), 'activation': 'tanh', 'optimizer': 'adam'}
]


# Modellerin Eğitimi ve Değerlendirilmesi
for model_info in models:
    model_name = model_info['name']
    hidden_layer_size = model_info['hidden_layer_sizes']
    activation = model_info['activation']
    optimizer = model_info['optimizer']
    
    print(f"\nEvaluating {model_name} with hidden_layer_size={hidden_layer_size}, activation={activation}, optimizer={optimizer}")
    
    model = create_model(hidden_layer_size, activation, optimizer)
    print(model.summary())  # Model özetini yazdırır.
    
    plot_model(model, to_file=f'{model_name}_architecture.png', show_shapes=True, show_layer_names=True)  # Model mimarisini görselleştirir ve bir dosyaya kaydeder.
    
    accuracies = []
    for train_index, val_index in kf5.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold, batch_size=32, epochs=50, verbose=0)  # Modeli eğitim katmanında eğitir.
        
        y_val_pred = (model.predict(X_val_fold) > 0.5).astype(int)  # Tahminleri yapar ve ikili sınıfa dönüştürür.
        
        accuracies.append(accuracy_score(y_val_fold, y_val_pred))  # Doğruluk skorunu hesaplar.
    
    print(f"Cross-validation accuracy for {model_name}: {np.mean(accuracies):.4f}")
    
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)  # Tüm eğitim verisi üzerinde eğitir.
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # Test seti üzerinde tahminler yapar.
    
    # Performans metriklerini hesaplar ve ekrana yazdırır.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    print(f"\n{model_name} - Classification Report:\n", classification_report(y_test, y_pred))  # Sınıflandırma raporunu yazdırır.
    
    # Confusion matrix'i görselleştirir.
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


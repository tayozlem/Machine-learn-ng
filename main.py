"""

MÜŞTERİ CHURN (AYRILMA) TAHMİNİ MAKİNE ÖĞRENMESİ PİPELİNE'I

1. ÖDEVİN AMACI:
    Bu projenin amacı, telekom/hizmet sektöründeki müşteri verilerini kullanarak
    müşterilerin şirketten ayrılıp ayrılmayacağını (Churn: 1, Stay: 0) tahmin 
    eden uçtan uca bir Makine Öğrenmesi (Machine Learning) boru hattı 
    (pipeline) geliştirmektir. 
    
    Proje kapsamında sentetik bir veri seti üretilmiş, veri ön işleme, eksik 
    değer doldurma (imputation), özellik mühendisliği (feature engineering), 
    kategorik kodlama (encoding) ve standartlaştırma (scaling) adımları 
    uygulanmıştır. Modeller %60 Train, %20 Validation ve %20 Test olacak 
    şekilde bölünmüş; Logistic Regression, KNN, Decision Tree ve Random Forest 
    algoritmalarının performansları Accuracy, Precision, Recall, F1-Score, 
    ROC-AUC ve Karmaşıklık Matrisi (Confusion Matrix) metrikleri üzerinden 
    karşılaştırılmıştır.

2. KULLANILAN KÜTÜPHANELER:
    - pandas: Veri yapısı (DataFrame) yönetimi, CSV işlemleri ve öznitelik dönüşümleri.
    - numpy: Sentetik veri üretimi ve vektörel/matematiksel hesaplamalar.
    - scikit-learn (sklearn): 
        * Model Selection: train_test_split
        * Preprocessing: StandardScaler
        * Models: LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier
        * Metrics: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
    - matplotlib & seaborn: Karmaşıklık matrislerinin ve grafiklerin görselleştirilmesi.

3. ÇALIŞTIRMA ADIMLARI:
    Adım 1: Gerekli kütüphaneleri terminal üzerinden yükleyin:
            pip install pandas numpy scikit-learn matplotlib seaborn
            
    Adım 2: Kod dosyasını bir Python ortamında (VS Code, PyCharm, Jupyter Notebook vb.) çalıştırın.
            python main.py
            
    Adım 3: Kod otomatik olarak sentetik veri setini oluşturacak, 
            modelleri eğitecek, metrik sonuç tablosunu terminale basacak ve 2x2 düzeninde 
            Confusion Matrix grafik penceresini ekrana getirecektir.

"""

# Gerekli kütüphaneleri yükleyelim 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# Rastgele üretimin tekrarlanabilir olması için seed sabitlenir
np.random.seed(42)

n_samples = 500

# Özelliklerin (Features) Oluşturulması
musteri_id = [f"CUST_{i:04d}" for i in range(1, n_samples + 1)]
yas = np.random.randint(18, 65, size=n_samples)

# Gelir sütununu oluşturup NaN içerebilmesi için float türüne çeviriyoruz
gelir = np.random.randint(17000, 100000, size=n_samples).astype(float)

# 'gelir' sütununun %10'una rastgele NaN ekleme
mask = np.random.choice([True, False], size=n_samples, p=[0.10, 0.90])
gelir[mask] = np.nan

abonelik_suresi = np.random.randint(1, 60, size=n_samples)  # Ay cinsinden
destek_talebi_sayisi = np.random.poisson(lam=2.5, size=n_samples)  # Poisson dağılımı
sehir = np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Adana', 'Eskişehir'], size=n_samples)
uyelik_tipi = np.random.choice(['Standart', 'Premium', 'VIP'], size=n_samples, p=[0.50, 0.35, 0.15])

# 3. Churn Olasılığının Hesaplanması
gelir_gecici = np.nan_to_num(gelir, nan=np.nanmean(gelir))

churn_olasilik = (
    0.20 
    + 0.08 * destek_talebi_sayisi 
    - 0.008 * abonelik_suresi 
    - 0.000001 * gelir_gecici 
    + np.where(uyelik_tipi == 'Standart', 0.10, -0.05)
)

# Tam %50 - %50 dengeli dağılım üretimi
esik_deger = np.median(churn_olasilik)
churn = (churn_olasilik >= esik_deger).astype(int)

# DataFrame Oluşturma
df = pd.DataFrame({
    'musteri_id': musteri_id,
    'yas': yas,
    'gelir': gelir,
    'abonelik_suresi': abonelik_suresi,
    'destek_talebi_sayisi': destek_talebi_sayisi,
    'sehir': sehir,
    'uyelik_tipi': uyelik_tipi,
    'churn': churn
})

# Veri Setini CSV Olarak Kaydetme
df.to_csv('musteri_churn_veri_seti.csv', index=False, encoding='utf-8-sig')

# Eksik değer doldurma
median_deger = df['gelir'].median()
df['gelir'] = df['gelir'].fillna(median_deger)

# Müşteri id çıkarma
df_model = df.drop(columns=['musteri_id']).copy()

# Ordinal Veri İçin Encoding (uyelik_tipi)
uyelik_siralama = {
    'Standart': 1,
    'Premium': 2,
    'VIP': 3
}
df_model['uyelik_tipi_encoded'] = df_model['uyelik_tipi'].map(uyelik_siralama)
df_model = df_model.drop(columns=['uyelik_tipi'])

# Nominal veri için One-Hot encoding (sehir)
df_model = pd.get_dummies(df_model, columns=['sehir'], drop_first=True, dtype=int)

# Yeni özellik üretme (yas_grubu)
def yas_kategorisi(yas):
    if yas < 30:
        return 'Genç'
    elif 30 <= yas <= 50:
        return 'Orta Yaş'
    else:
        return 'Kıdemli'

df_model["yas_grubu"] = df_model["yas"].apply(yas_kategorisi)

# DÜZELTME 4: Değişken ismi yas_siralama olarak güncellendi
yas_siralama = {
    'Genç': 1,
    'Orta Yaş': 2,
    'Kıdemli': 3
}
df_model['yas_grubu_encoded'] = df_model['yas_grubu'].map(yas_siralama)
df_model = df_model.drop(columns=['yas', 'yas_grubu'])

# Veri Setini (X ve y) Ayırma
X = df_model.drop(columns=['churn'])
y = df_model['churn']

# Train (%60), Validation (%20), Test (%20) Bölünmesi
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# DÜZELTME 1, 2 VE 3: ÖLÇEKLEME UYGULAMASI VE YENİ DEĞİŞKENLERE ATAMA
sayisal_sutunlar = ['gelir', 'abonelik_suresi', 'destek_talebi_sayisi']
scaler_std = StandardScaler()

# Kopya oluşturarak ölçeklenmiş veri setlerini hazırlıyoruz
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[sayisal_sutunlar] = scaler_std.fit_transform(X_train[sayisal_sutunlar])
X_val_scaled[sayisal_sutunlar] = scaler_std.transform(X_val[sayisal_sutunlar])
X_test_scaled[sayisal_sutunlar] = scaler_std.transform(X_test[sayisal_sutunlar])

# Modellerin Tanımlanması
modeller = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

sonuclar = []

# Matris Görselleştirmesi İçin Plot Alanı
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.ravel()

for i, (isim, model) in enumerate(modeller.items()):
    # Modeli Eğitme (ÖLÇEKLENMİŞ Train verisi ile)
    model.fit(X_train_scaled, y_train)
    
    # Validation ve Test tahminleri (ÖLÇEKLENMİŞ veri ile)
    val_tahmin = model.predict(X_val_scaled)
    test_tahmin = model.predict(X_test_scaled)
    test_olasilik = model.predict_proba(X_test_scaled)[:, 1]
    
    # Confusion Matrix Hesaplama
    cm = confusion_matrix(y_test, test_tahmin)
    
    # Matrix Grafiğini Çizdirme
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Kalır (0)', 'Ayrılır (1)'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f'{isim}\nConfusion Matrix', fontsize=11, fontweight='bold')
    
    # Metriklerin hesaplanması
    sonuclar.append({
        'Model': isim,
        'Val Accuracy': accuracy_score(y_val, val_tahmin),
        'Test Accuracy': accuracy_score(y_test, test_tahmin),
        'Test Precision': precision_score(y_test, test_tahmin, zero_division=0),
        'Test Recall': recall_score(y_test, test_tahmin, zero_division=0),
        'Test F1-Score': f1_score(y_test, test_tahmin, zero_division=0),
        'Test ROC-AUC': roc_auc_score(y_test, test_olasilik),
        'TN (Doğru Negatif)': cm[0, 0],
        'FP (Yanlış Pozitif)': cm[0, 1],
        'FN (Yanlış Negatif)': cm[1, 0],
        'TP (Doğru Pozitif)': cm[1, 1]
    })

plt.tight_layout()
plt.show()

# Sonuç Tablosu
sonuc_df = pd.DataFrame(sonuclar)
print(sonuc_df.to_string(index=False))

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
from datetime import datetime

# --- YENİ ÖZELLİK: HAFIZA VE VERİ TABANI YÖNETİMİ ---
DOSYA_GECMIS = "kullanici_gecmisi.csv"

# Bugünün tarihini alıyoruz (Örn: 2026-05-31)
bugun = datetime.now().strftime("%Y-%m-%d")

# Geçmiş kayıtları getiren fonksiyon
def kayitlari_getir():
    if os.path.exists(DOSYA_GECMIS):
        return pd.read_csv(DOSYA_GECMIS)
    else:
        return pd.DataFrame(columns=["Tarih", "İsim", "Gramaj", "Kalori", "Protein", "Karbo", "Yağ"])

# Yeni öğünü kalıcı olarak kaydeden fonksiyon
def kayit_ekle(isim, gramaj, kalori, protein, karbo, yag):
    df = kayitlari_getir()
    yeni_kayit = pd.DataFrame([{
        "Tarih": bugun, 
        "İsim": isim, 
        "Gramaj": gramaj,
        "Kalori": kalori, 
        "Protein": round(protein, 1), 
        "Karbo": round(karbo, 1), 
        "Yağ": round(yag, 1)
    }])
    df = pd.concat([df, yeni_kayit], ignore_index=True)
    df.to_csv(DOSYA_GECMIS, index=False)

# Bugünün kayıtlarını silen fonksiyon
def bugunu_sifirla():
    if os.path.exists(DOSYA_GECMIS):
        df = pd.read_csv(DOSYA_GECMIS)
        # Bugüne ait olmayanları tut (Yani bugünü silmiş oluyoruz)
        df = df[df["Tarih"] != bugun]
        df.to_csv(DOSYA_GECMIS, index=False)

# 1. YAPAY ZEKA MODELİNİ YÜKLEME
@st.cache_resource
def model_yukle():
    return tf.keras.models.load_model("yemek_modeli.h5")

try:
    model = model_yukle()
except Exception as e:
    st.error(f"Dosya orada ama yüklenirken şu GERÇEK hata oluştu: {e}")

siniflar = ['pizza', 'hamburger', 'sushi'] 

if "gecici_analiz" not in st.session_state:
    st.session_state.gecici_analiz = None

# 3. BESİN DEĞERLERİNİ GETİR (CSV'DEN)
def csv_degerlerini_getir(yemek_adi):
    try:
        df = pd.read_csv("nutrition_dataset.csv")
        df['food_name'] = df['food_name'].str.lower()
        eslesme = df[df['food_name'].str.contains(yemek_adi.lower(), na=False)]
        if not eslesme.empty:
            ilk_satir = eslesme.iloc[0]
            return {
                "kalori": float(ilk_satir['calories']), "protein": float(ilk_satir['protein']),
                "karbo": float(ilk_satir['carbs']), "yag": float(ilk_satir['fat'])
            }
    except FileNotFoundError:
        yedek_veriler = {
            "hamburger": {"kalori": 295, "protein": 14, "karbo": 24, "yag": 14},
            "pizza": {"kalori": 266, "protein": 11, "karbo": 33, "yag": 10},
            "sushi": {"kalori": 143, "protein": 4, "karbo": 32, "yag": 4}
        }
        return yedek_veriler.get(yemek_adi, {"kalori": 100, "protein": 5, "karbo": 15, "yag": 2})

porsiyon_tanimlari = {"hamburger": 200, "pizza": 300, "sushi": 150}

# 4. WEB ARAYÜZÜ TASARIMI
st.set_page_config(page_title="AI Diyetisyen", page_icon="🥗", layout="centered")

# --- SOL YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.header("👤 Sağlık Profili")
    
    cinsiyet = st.radio("Cinsiyet:", ["Erkek", "Kadın"])
    yas = st.number_input("Yaş:", min_value=15, max_value=100, value=25)
    boy = st.number_input("Boy (cm):", min_value=100, max_value=250, value=175)
    kilo = st.number_input("Kilo (kg):", min_value=30, max_value=200, value=70)
    
    hedef = st.selectbox("Hedefiniz:", ["Kilo Korumak", "Kilo Vermek", "Kilo Almak"])
    
    if cinsiyet == "Erkek":
        bmr = 88.362 + (13.397 * kilo) + (4.799 * boy) - (5.677 * yas)
    else:
        bmr = 447.593 + (9.247 * kilo) + (3.098 * boy) - (4.330 * yas)
        
    gunluk_ihtiyac = bmr * 1.375 
    
    if hedef == "Kilo Vermek": hedef_kalori = gunluk_ihtiyac - 500
    elif hedef == "Kilo Almak": hedef_kalori = gunluk_ihtiyac + 400
    else: hedef_kalori = gunluk_ihtiyac
        
    vki = kilo / ((boy/100)**2)
    
    st.divider()
    st.metric("⚖️ VKI", f"{vki:.1f}")
    st.metric("🎯 Günlük Hedef", f"{int(hedef_kalori)} kcal")


# --- ANA EKRAN (ÜST KISIM): FOTOĞRAF VE ANALİZ ---
st.markdown("<h1 style='text-align: center;'>🥗 AI Diyetisyen & Akıllı Takip</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("### 📸 Yeni Öğün Ekle")
giris_tipi = st.radio("Miktar:", ["🍽️ Porsiyon", "📏 Gramaj"], horizontal=True)

if giris_tipi == "🍽️ Porsiyon":
    porsiyon_miktari = st.slider("Miktar (Porsiyon):", 0.5, 5.0, 1.0, 0.5)
else:
    gramaj_miktari = st.slider("Miktar (Gram):", 50, 1500, 200, 50)
    
yuklenen_resim = st.file_uploader("Öğününüzün fotoğrafını seçin:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if yuklenen_resim is not None:
    image = Image.open(yuklenen_resim).convert('RGB')
    st.image(image, caption="Analiz Edilecek Yemek", use_container_width=True)
    
    if st.button("🔍 Yemeği Analiz Et", use_container_width=True):
        with st.spinner("Yapay Zeka İnceliyor..."):
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0)
            
            tahminler = model.predict(img_array)
            st.session_state.gecici_analiz = siniflar[np.argmax(tahminler)]
    
    if st.session_state.gecici_analiz:
        st.info(f"🤖 Tahmin: **{st.session_state.gecici_analiz.upper()}**")
        
        dogrulanan_yemek = st.selectbox("Hatalıysa doğrusunu seçin:", siniflar, index=siniflar.index(st.session_state.gecici_analiz))
        
        if st.button("✅ Kalıcı Olarak Günlüğe Ekle", type="primary", use_container_width=True):
            degerler = csv_degerlerini_getir(dogrulanan_yemek)
            
            final_gramaj = porsiyon_miktari * porsiyon_tanimlari[dogrulanan_yemek] if giris_tipi == "🍽️ Porsiyon" else gramaj_miktari
            oran = final_gramaj / 100.0
            
            # Veri tabanına (CSV) kaydediyoruz!
            kayit_ekle(
                isim=dogrulanan_yemek.capitalize(),
                gramaj=int(final_gramaj),
                kalori=int(degerler['kalori'] * oran),
                protein=degerler['protein'] * oran,
                karbo=degerler['karbo'] * oran,
                yag=degerler['yag'] * oran
            )
            
            st.session_state.gecici_analiz = None 
            st.success("Tarihçenize başarıyla eklendi!")
            st.rerun()

st.divider()

# --- ANA EKRAN (ALT KISIM): SEKMELİ ÖZET ALANI ---
df_tum_kayitlar = kayitlari_getir()

# SEKMELER (TABS) OLUŞTURULUYOR
tab1, tab2 = st.tabs(["📊 Bugünün Özeti", "📈 Geçmiş & Raporlar"])

# TAB 1: BUGÜN
with tab1:
    # Sadece bugünün tarihine ait olanları filtrele
    df_bugun = df_tum_kayitlar[df_tum_kayitlar["Tarih"] == bugun]
    
    if df_bugun.empty:
        st.info("Bugün henüz bir kayıt girmediniz.")
    else:
        toplam_kalori = df_bugun['Kalori'].sum()
        toplam_protein = df_bugun['Protein'].sum()
        toplam_karbo = df_bugun['Karbo'].sum()
        toplam_yag = df_bugun['Yağ'].sum()
        
        kalan_kalori = int(hedef_kalori) - toplam_kalori
        oran_bar = min(toplam_kalori / hedef_kalori, 1.0)
        
        st.write(f"**Günlük Alınan Enerji:** {toplam_kalori} / {int(hedef_kalori)} kcal")
        st.progress(oran_bar)
        
        if kalan_kalori > 0:
            st.success(f"💡 Hedefinize ulaşmak için hala **{kalan_kalori} kcal** yiyebilirsiniz.")
        else:
            st.error(f"⚠️ Günlük kalori hedefinizi **{abs(kalan_kalori)} kcal** aştınız!")
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔥 Kalori", f"{toplam_kalori} kcal")
        m2.metric("🥩 Protein", f"{toplam_protein:.1f} g")
        m3.metric("🍞 Karbo", f"{toplam_karbo:.1f} g")
        m4.metric("🥑 Yağ", f"{toplam_yag:.1f} g")
        
        st.write("📋 **Bugün Yediklerim:**")
        # Tarih sütununu gizleyip listeyi gösteriyoruz
        st.dataframe(df_bugun.drop(columns=["Tarih"]), use_container_width=True)
        
        if st.button("🗑️ Bugünün Kayıtlarını Sil"):
            bugunu_sifirla()
            st.rerun()

# TAB 2: GEÇMİŞ HAFIZA
with tab2:
    if df_tum_kayitlar.empty:
        st.info("Henüz geçmişe dönük bir veriniz bulunmuyor.")
    else:
        st.markdown("### 📅 Günlük Kalori Tüketimi Grafiği")
        # Günlere göre kalori toplamını alıp çubuk grafiğe çeviriyoruz
        gunluk_kaloriler = df_tum_kayitlar.groupby("Tarih")["Kalori"].sum()
        st.bar_chart(gunluk_kaloriler, color="#ff4b4b")
        
        st.markdown("### 📂 Tüm Zamanların Veri Tabani")
        st.dataframe(df_tum_kayitlar, use_container_width=True)

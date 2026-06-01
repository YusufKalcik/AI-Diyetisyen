import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
from datetime import datetime

# --- 1. SAYFA AYARLARI (Her şeyden önce gelmeli) ---
st.set_page_config(page_title="AI Diyetisyen V2", page_icon="🥗", layout="centered")


# --- 2. GOD MODE: KÜRESEL SÜRÜM YAMASI ---
def gereksizleri_temizle(kwargs):
    copluk = ['renorm', 'renorm_clipping', 'renorm_momentum', 'quantization_config']
    for kelime in copluk:
        kwargs.pop(kelime, None)
    return kwargs

_orijinal_bn = tf.keras.layers.BatchNormalization.__init__
def _yamali_bn(self, *args, **kwargs):
    _orijinal_bn(self, *args, **gereksizleri_temizle(kwargs))
tf.keras.layers.BatchNormalization.__init__ = _yamali_bn

_orijinal_dense = tf.keras.layers.Dense.__init__
def _yamali_dense(self, *args, **kwargs):
    _orijinal_dense(self, *args, **gereksizleri_temizle(kwargs))
tf.keras.layers.Dense.__init__ = _yamali_dense

_orijinal_conv = tf.keras.layers.Conv2D.__init__
def _yamali_conv(self, *args, **kwargs):
    _orijinal_conv(self, *args, **gereksizleri_temizle(kwargs))
tf.keras.layers.Conv2D.__init__ = _yamali_conv


# --- 3. PROFiL VE HAFIZA YÖNETiMi ---
PROFILLER_DOSYASI = "profiller.csv"
bugun = datetime.now().strftime("%Y-%m-%d")

def profilleri_getir():
    if os.path.exists(PROFILLER_DOSYASI):
        return pd.read_csv(PROFILLER_DOSYASI)
    return pd.DataFrame(columns=["İsim", "Cinsiyet", "Yaş", "Boy", "Kilo", "Hedef"])

def profil_kaydet(isim, cinsiyet, yas, boy, kilo, hedef):
    df = profilleri_getir()
    # Eğer aynı isimde profil varsa güncellemek için eskisini siliyoruz
    df = df[df["İsim"].str.lower() != isim.lower()]
    yeni_profil = pd.DataFrame([{"İsim": isim, "Cinsiyet": cinsiyet, "Yaş": yas, "Boy": boy, "Kilo": kilo, "Hedef": hedef}])
    df = pd.concat([df, yeni_profil], ignore_index=True)
    df.to_csv(PROFILLER_DOSYASI, index=False)

def aktif_gecmis_dosyasi():
    if "aktif_profil" in st.session_state and st.session_state.aktif_profil != "+ Yeni Profil Ekle":
        return f"{st.session_state.aktif_profil.lower()}_gecmisi.csv"
    return "misafir_gecmisi.csv"

def kayitlari_getir():
    dosya = aktif_gecmis_dosyasi()
    if os.path.exists(dosya):
        return pd.read_csv(dosya)
    return pd.DataFrame(columns=["Tarih", "İsim", "Gramaj", "Kalori", "Protein", "Karbo", "Yağ"])

def kayit_ekle(isim, gramaj, kalori, protein, karbo, yag):
    dosya = aktif_gecmis_dosyasi()
    df = kayitlari_getir()
    yeni_kayit = pd.DataFrame([{
        "Tarih": bugun, "İsim": isim, "Gramaj": gramaj, "Kalori": kalori, 
        "Protein": round(protein, 1), "Karbo": round(karbo, 1), "Yağ": round(yag, 1)
    }])
    df = pd.concat([df, yeni_kayit], ignore_index=True)
    df.to_csv(dosya, index=False)

def bugunu_sifirla():
    dosya = aktif_gecmis_dosyasi()
    if os.path.exists(dosya):
        df = pd.read_csv(dosya)
        df = df[df["Tarih"] != bugun]
        df.to_csv(dosya, index=False)


# --- 4. YAPAY ZEKA MODELiNi YÜKLEME ---
@st.cache_resource
def model_yukle():
    return tf.keras.models.load_model("yemek_modeli.h5", compile=False)

try:
    model = model_yukle()
except Exception as e:
    st.error(f"Model yüklenirken bir pürüz oluştu: {e}")

siniflar = ['pizza', 'hamburger', 'sushi'] 
porsiyon_tanimlari = {"hamburger": 200, "pizza": 300, "sushi": 150}

if "gecici_analiz" not in st.session_state:
    st.session_state.gecici_analiz = None


# --- 5. SOL YAN MENÜ: ÇOKLU PROFiL SİSTEMİ ---
df_profiller = profilleri_getir()
profil_listesi = df_profiller["İsim"].tolist() + ["+ Yeni Profil Ekle"]

if "aktif_profil" not in st.session_state:
    st.session_state.aktif_profil = profil_listesi[0]

with st.sidebar:
    st.header("👤 Kullanıcı Profili")
    
    secilen_profil = st.selectbox(
        "Aktif Profil Seçin:", 
        profil_listesi, 
        index=profil_listesi.index(st.session_state.aktif_profil) if st.session_state.aktif_profil in profil_listesi else 0
    )
    st.session_state.aktif_profil = secilen_profil
    st.divider()

    # --- DURUM A: YENİ PROFiL OLUŞTURMA ---
    if secilen_profil == "+ Yeni Profil Ekle":
        st.subheader("🆕 Yeni Profil Oluştur")
        yeni_isim = st.text_input("İsim:")
        cinsiyet = st.radio("Cinsiyet:", ["Erkek", "Kadın"])
        yas = st.number_input("Yaş:", 15, 100, 25)
        boy = st.number_input("Boy (cm):", 100, 250, 175)
        kilo = st.number_input("Kilo (kg):", 30, 200, 70)
        hedef = st.selectbox("Hedefiniz:", ["Kilo Korumak", "Kilo Vermek", "Kilo Almak"])
        
        if st.button("💾 Profili Kaydet", use_container_width=True):
            if yeni_isim.strip():
                profil_kaydet(yeni_isim.strip(), cinsiyet, yas, boy, kilo, hedef)
                st.session_state.aktif_profil = yeni_isim.strip()
                st.success(f"{yeni_isim} profili başarıyla oluşturuldu!")
                st.rerun()
            else:
                st.error("Lütfen geçerli bir isim girin!")

    # --- DURUM B: MEVCUT PROFiLÜ YÜKLEME VE DÜZENLEME ---
    else:
        p_veri = df_profiller[df_profiller["İsim"] == secilen_profil].iloc[0]
        
        st.subheader(f"⚙️ {secilen_profil} Profilini Düzenle")
        cinsiyet = st.radio("Cinsiyet:", ["Erkek", "Kadın"], index=0 if p_veri["Cinsiyet"] == "Erkek" else 1)
        yas = st.number_input("Yaş:", 15, 100, int(p_veri["Yaş"]))
        boy = st.number_input("Boy (cm):", 100, 250, int(p_veri["Boy"]))
        kilo = st.number_input("Kilo (kg):", 30, 200, int(p_veri["Kilo"]))
        hedef = st.selectbox("Hedefiniz:", ["Kilo Korumak", "Kilo Vermek", "Kilo Almak"], index=["Kilo Korumak", "Kilo Vermek", "Kilo Almak"].index(p_veri["Hedef"]))
        
        if st.button("🔄 Bilgileri Güncelle", use_container_width=True):
            profil_kaydet(secilen_profil, cinsiyet, yas, boy, kilo, hedef)
            st.success("Profil bilgileri güncellendi!")
            st.rerun()

        # Kalori Hesaplama Motoru
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

# --- YENİ PROFİL EKLENİRKEN ANA EKRANI DURDURAN VE KARŞILAYAN BLOK (DÜZELTİLDİ) ---
if secilen_profil == "+ Yeni Profil Ekle":
    st.markdown("<h2 style='text-align: center; margin-top: 100px;'>👋 AI Diyetisyene Hoş Geldiniz!</h2>", unsafe_allow_html=True)
    st.info("👈 Uygulamayı kullanmaya başlamak için lütfen sol menüden bilgilerinizi girip 'Profili Kaydet' butonuna basın.")
    st.stop()

# --- 6. ANA EKRAN: FOTOĞRAF VE YAPAY ZEKA ---
st.markdown(f"<h1 style='text-align: center;'>🥗 AI Diyetisyen ({st.session_state.aktif_profil})</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("### 📸 Yeni Öğün Ekle")
giris_tipi = st.radio("Miktar:", ["🍽️ Porsiyon", "📏 Gramaj"], horizontal=True)

if giris_tipi == "🍽️ Porsiyon":
    porsiyon_miktari = st.slider("Miktar (Porsiyon):", 0.5, 5.0, 1.0, 0.5)
else:
    gramaj_miktari = st.slider("Miktar (Gram):", 50, 1500, 200, 50)
    
yuklenen_resim = st.file_uploader("Öğününüzün fotoğrafını seçin:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# CSV'den besin değeri okuma fonksiyonu
def csv_degerlerini_getir(yemek_adi):
    try:
        df = pd.read_csv("nutrition_dataset.csv")
        df['food_name'] = df['food_name'].str.lower()
        eslesme = df[df['food_name'].str.contains(yemek_adi.lower(), na=False)]
        if not eslesme.empty:
            ilk_satir = eslesme.iloc[0]
            return {"kalori": float(ilk_satir['calories']), "protein": float(ilk_satir['protein']), "karbo": float(ilk_satir['carbs']), "yag": float(ilk_satir['fat'])}
    except:
        pass
    yedek_veriler = {"hamburger": {"kalori": 295, "protein": 14, "karbo": 24, "yag": 14}, "pizza": {"kalori": 266, "protein": 11, "karbo": 33, "yag": 10}, "sushi": {"kalori": 143, "protein": 4, "karbo": 32, "yag": 4}}
    return yedek_veriler.get(yemek_adi, {"kalori": 100, "protein": 5, "karbo": 15, "yag": 2})

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
            
            # Seçili profile özel dosyaya kaydeder
            kayit_ekle(
                isim=dogrulanan_yemek.capitalize(), gramaj=int(final_gramaj),
                kalori=int(degerler['kalori'] * oran), protein=degerler['protein'] * oran,
                karbo=degerler['karbo'] * oran, yag=degerler['yag'] * oran
            )
            st.session_state.gecici_analiz = None 
            st.success("Günlüğünüze başarıyla eklendi!")
            st.rerun()

st.divider()


# --- 7. SEKMELİ ANALİZ ALANI ---
df_tum_kayitlar = kayitlari_getir()
tab1, tab2 = st.tabs(["📊 Bugünün Özeti", "📈 Geçmiş & Raporlar"])

with tab1:
    df_bugun = df_tum_kayitlar[df_tum_kayitlar["Tarih"] == bugun]
    if df_bugun.empty:
        st.info(f"{st.session_state.aktif_profil} için bugün henüz bir kayıt girmediniz.")
    else:
        toplam_kalori = df_bugun['Kalori'].sum()
        toplam_protein = df_bugun['Protein'].sum()
        toplam_karbo = df_bugun['Karbo'].sum()
        toplam_yag = df_bugun['Yağ'].sum()
        
        kalan_kalori = int(hedef_kalori) - toplam_kalori
        oran_bar = min(max(toplam_kalori / hedef_kalori, 0.0), 1.0)
        
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
        st.dataframe(df_bugun.drop(columns=["Tarih"]), use_container_width=True)
        
        if st.button("🗑️ Bugünün Kayıtlarını Sil"):
            bugunu_sifirla()
            st.rerun()

with tab2:
    if df_tum_kayitlar.empty:
        st.info("Henüz geçmişe dönük bir veriniz bulunmuyor.")
    else:
        st.markdown("### 📅 Günlük Kalori Tüketimi Grafiği")
        gunluk_kaloriler = df_tum_kayitlar.groupby("Tarih")["Kalori"].sum()
        st.bar_chart(gunluk_kaloriler, color="#ff4b4b")
        
        st.markdown("### 📂 Tüm Zamanların Veri Tabani")
        st.dataframe(df_tum_kayitlar, use_container_width=True)

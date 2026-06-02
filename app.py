import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
from datetime import datetime
import plotly.graph_objects as go

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="AI Diyetisyen V3.7", page_icon="🥗", layout="centered")


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
    # V3.7 Yenilik: "Aktivite" sütunu veri tabanına eklendi
    return pd.DataFrame(columns=["İsim", "Sifre", "Cinsiyet", "Yaş", "Boy", "Kilo", "Hedef", "Aktivite"])

def profil_kaydet(isim, sifre, cinsiyet, yas, boy, kilo, hedef, aktivite):
    df = profilleri_getir()
    df = df[df["İsim"].str.lower() != isim.lower()]
    yeni_profil = pd.DataFrame([{"İsim": isim, "Sifre": str(sifre), "Cinsiyet": cinsiyet, "Yaş": yas, "Boy": boy, "Kilo": kilo, "Hedef": hedef, "Aktivite": aktivite}])
    df = pd.concat([df, yeni_profil], ignore_index=True)
    df.to_csv(PROFILLER_DOSYASI, index=False)

def profil_sil(isim):
    if os.path.exists(PROFILLER_DOSYASI):
        df = pd.read_csv(PROFILLER_DOSYASI)
        df = df[df["İsim"] != isim]
        df.to_csv(PROFILLER_DOSYASI, index=False)
    
    dosya_gecmis = f"{isim.lower()}_gecmisi.csv"
    if os.path.exists(dosya_gecmis): os.remove(dosya_gecmis)
        
    dosya_su = f"{isim.lower()}_su.csv"
    if os.path.exists(dosya_su): os.remove(dosya_su)

def aktif_gecmis_dosyasi():
    if "aktif_kullanici" in st.session_state and st.session_state.aktif_kullanici is not None:
        return f"{st.session_state.aktif_kullanici.lower()}_gecmisi.csv"
    return "misafir_gecmisi.csv"

def aktif_su_dosyasi():
    if "aktif_kullanici" in st.session_state and st.session_state.aktif_kullanici is not None:
        return f"{st.session_state.aktif_kullanici.lower()}_su.csv"
    return "misafir_su.csv"

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

def tekil_kayit_sil(silinecek_index):
    dosya = aktif_gecmis_dosyasi()
    if os.path.exists(dosya):
        df = pd.read_csv(dosya)
        df = df.drop(index=silinecek_index)
        df.to_csv(dosya, index=False)

def bugunu_sifirla():
    dosya = aktif_gecmis_dosyasi()
    if os.path.exists(dosya):
        df = pd.read_csv(dosya)
        df = df[df["Tarih"] != bugun]
        df.to_csv(dosya, index=False)

# SU TAKİP FONKSİYONLARI (EKSİ DEĞER DESTEKLİ)
def su_getir():
    dosya = aktif_su_dosyasi()
    if os.path.exists(dosya):
        df = pd.read_csv(dosya)
        bugun_df = df[df["Tarih"] == bugun]
        # Toplam suyun 0'ın altına düşmesini engelle
        return max(0, bugun_df["Miktar"].sum()) if not bugun_df.empty else 0
    return 0

def su_ekle(miktar):
    dosya = aktif_su_dosyasi()
    df = pd.read_csv(dosya) if os.path.exists(dosya) else pd.DataFrame(columns=["Tarih", "Miktar"])
    yeni_kayit = pd.DataFrame([{"Tarih": bugun, "Miktar": miktar}])
    df = pd.concat([df, yeni_kayit], ignore_index=True)
    df.to_csv(dosya, index=False)

def su_sifirla():
    dosya = aktif_su_dosyasi()
    if os.path.exists(dosya):
        df = pd.read_csv(dosya)
        df = df[df["Tarih"] != bugun]
        df.to_csv(dosya, index=False)

# Tıbbi Güvenlik Duvarı
def tıbbi_hedef_onayi(boy, kilo, hedef):
    vki = kilo / ((boy / 100) ** 2)
    if vki < 18.5 and hedef == "Kilo Vermek": return False, "🚨 TIBBİ UYARI: Vücut Kitle İndeksiniz 'Zayıf'. Sağlığınız için 'Kilo Vermek' hedefini onaylamıyoruz!"
    elif vki >= 25.0 and hedef == "Kilo Almak": return False, "🚨 TIBBİ UYARI: Vücut Kitle İndeksiniz yüksek. Sağlığınız için 'Kilo Almak' hedefini onaylamıyoruz!"
    return True, ""


# --- 4. YAPAY ZEKA MODELiNi YÜKLEME ---
@st.cache_resource
def model_yukle():
    return tf.keras.models.load_model("yemek_modeli.h5", compile=False)

try: model = model_yukle()
except Exception as e: st.error(f"Model yüklenirken bir pürüz oluştu: {e}")

siniflar = ['pizza', 'hamburger', 'sushi'] 
porsiyon_tanimlari = {"hamburger": 200, "pizza": 300, "sushi": 150}
aktivite_secenekleri = ["Hareketsiz / Masa başı iş", "Hafif Aktif / Haftada 1-2 gün spor", "Çok Aktif / Ağır antrenman & Kreatin Kullanımı"]

if "gecici_analiz" not in st.session_state: st.session_state.gecici_analiz = None
if "guven_orani" not in st.session_state: st.session_state.guven_orani = 0.0


# --- 5. SOL YAN MENÜ: ŞİFRELİ GİRİŞ VE PROFiL MOTORU ---
if "aktif_kullanici" not in st.session_state:
    st.session_state.aktif_kullanici = None

with st.sidebar:
    if st.session_state.aktif_kullanici is None:
        st.header("🔐 Sisteme Giriş")
        islem = st.radio("Bir işlem seçin:", ["Giriş Yap", "Yeni Kayıt Ol"], horizontal=True)
        st.divider()
        
        if islem == "Giriş Yap":
            isim_giris = st.text_input("👤 Kullanıcı Adı:")
            sifre_giris = st.text_input("🔑 Şifre:", type="password")
            if st.button("🚀 Giriş Yap", use_container_width=True):
                df_prof = profilleri_getir()
                eslesme = df_prof[(df_prof["İsim"].str.lower() == isim_giris.lower().strip()) & (df_prof["Sifre"].astype(str) == sifre_giris.strip())]
                if not eslesme.empty:
                    st.session_state.aktif_kullanici = eslesme.iloc[0]["İsim"]
                    st.rerun()
                else: st.error("Kullanıcı adı veya şifre hatalı!")
                    
        else:
            yeni_isim = st.text_input("👤 Yeni Kullanıcı Adı:")
            yeni_sifre = st.text_input("🔑 Bir Şifre Belirleyin:", type="password")
            cinsiyet = st.radio("Cinsiyet:", ["Erkek", "Kadın"])
            yas = st.number_input("Yaş:", 15, 100, 25)
            boy = st.number_input("Boy (cm):", 100, 250, 175)
            kilo = st.number_input("Kilo (kg):", 30, 200, 70)
            hedef = st.selectbox("Hedefiniz:", ["Kilo Korumak", "Kilo Vermek", "Kilo Almak"])
            yeni_aktivite = st.selectbox("Fiziksel Aktivite Seviyeniz:", aktivite_secenekleri)
            
            if st.button("💾 Kayıt Ol ve Başla", use_container_width=True):
                df_prof = profilleri_getir()
                if yeni_isim.strip().lower() in df_prof["İsim"].str.lower().values: st.error("⚠️ Bu kullanıcı adı alınmış.")
                elif len(yeni_isim.strip()) == 0 or len(yeni_sifre.strip()) == 0: st.error("⚠️ Boş bırakılamaz!")
                else:
                    onay, mesaj = tıbbi_hedef_onayi(boy, kilo, hedef)
                    if onay:
                        profil_kaydet(yeni_isim.strip(), yeni_sifre.strip(), cinsiyet, yas, boy, kilo, hedef, yeni_aktivite)
                        st.session_state.aktif_kullanici = yeni_isim.strip()
                        st.rerun()
                    else: st.error(mesaj)

    else:
        aktif_isim = st.session_state.aktif_kullanici
        df_prof = profilleri_getir()
        p_veri = df_prof[df_prof["İsim"] == aktif_isim].iloc[0]
        
        st.header(f"⚙️ {aktif_isim} Profili")
        cinsiyet = st.radio("Cinsiyet:", ["Erkek", "Kadın"], index=0 if p_veri["Cinsiyet"] == "Erkek" else 1)
        yas = st.number_input("Yaş:", 15, 100, int(p_veri["Yaş"]))
        boy = st.number_input("Boy (cm):", 100, 250, int(p_veri["Boy"]))
        kilo = st.number_input("Kilo (kg):", 30, 200, int(p_veri["Kilo"]))
        hedef = st.selectbox("Hedefiniz:", ["Kilo Korumak", "Kilo Vermek", "Kilo Almak"], index=["Kilo Korumak", "Kilo Vermek", "Kilo Almak"].index(p_veri["Hedef"]))
        
        # Eğer eski veride aktivite yoksa varsayılan olarak ilkini seç
        akt_idx = aktivite_secenekleri.index(p_veri["Aktivite"]) if "Aktivite" in p_veri and p_veri["Aktivite"] in aktivite_secenekleri else 0
        aktivite = st.selectbox("Fiziksel Aktivite Seviyeniz:", aktivite_secenekleri, index=akt_idx)
        
        if st.button("🔄 Bilgilerimi Güncelle", use_container_width=True):
            onay, mesaj = tıbbi_hedef_onayi(boy, kilo, hedef)
            if onay:
                profil_kaydet(aktif_isim, p_veri["Sifre"], cinsiyet, yas, boy, kilo, hedef, aktivite)
                st.success("Güncellendi!")
                st.rerun()
            else: st.error(mesaj)

        if cinsiyet == "Erkek": bmr = 88.362 + (13.397 * kilo) + (4.799 * boy) - (5.677 * yas)
        else: bmr = 447.593 + (9.247 * kilo) + (3.098 * boy) - (4.330 * yas)
            
        gunluk_ihtiyac = bmr * 1.375 
        if hedef == "Kilo Vermek": hedef_kalori = gunluk_ihtiyac - 500
        elif hedef == "Kilo Almak": hedef_kalori = gunluk_ihtiyac + 400
        else: hedef_kalori = gunluk_ihtiyac
            
        vki = kilo / ((boy/100)**2)
        if vki < 18.5: vki_durum = "Zayıf"
        elif vki < 25.0: vki_durum = "Normal"
        elif vki < 30.0: vki_durum = "Fazla Kilolu"
        else: vki_durum = "Obez"
        
        st.divider()
        st.metric("⚖️ VKI", f"{vki:.1f} ({vki_durum})")
        st.metric("🎯 Günlük Hedef", f"{int(hedef_kalori)} kcal")
        
        st.divider()
        st.markdown("### 🔒 Oturum Yönetimi")
        if st.button("🚪 Güvenli Çıkış Yap", use_container_width=True):
            st.session_state.aktif_kullanici = None
            st.rerun()
            
        with st.expander("🚨 Tehlikeli Alan"):
            if st.button("🗑️ Hesabımı Kalıcı Olarak Sil", type="primary", use_container_width=True):
                profil_sil(aktif_isim)
                st.session_state.aktif_kullanici = None
                st.rerun()

if st.session_state.aktif_kullanici is None:
    st.markdown("<h2 style='text-align: center; margin-top: 100px;'>🔒 AI Diyetisyene Hoş Geldiniz!</h2>", unsafe_allow_html=True)
    st.info("👈 Kişisel diyet günlüğünüze ulaşmak için lütfen sol menüden giriş yapın veya ücretsiz kayıt olun.")
    st.stop()


# --- 6. ANA EKRAN: FOTOĞRAF VE YAPAY ZEKA ---
st.markdown(f"<h1 style='text-align: center;'>🥗 Hoş Geldin, {st.session_state.aktif_kullanici}!</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("### 📸 Yeni Öğün Ekle")
giris_tipi = st.radio("Miktar:", ["🍽️ Porsiyon", "📏 Gramaj"], horizontal=True)

if giris_tipi == "🍽️ Porsiyon": porsiyon_miktari = st.slider("Miktar (Porsiyon):", 0.5, 5.0, 1.0, 0.5)
else: gramaj_miktari = st.slider("Miktar (Gram):", 50, 1500, 200, 50)
    
yuklenen_resim = st.file_uploader("Öğününüzün fotoğrafını seçin:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

def csv_degerlerini_getir(yemek_adi):
    try:
        df = pd.read_csv("nutrition_dataset.csv")
        df['food_name'] = df['food_name'].str.lower()
        eslesme = df[df['food_name'].str.contains(yemek_adi.lower(), na=False)]
        if not eslesme.empty:
            ilk_satir = eslesme.iloc[0]
            return {"kalori": float(ilk_satir['calories']), "protein": float(ilk_satir['protein']), "karbo": float(ilk_satir['carbs']), "yag": float(ilk_satir['fat'])}
    except: pass
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
            
            tahmin_matrisi = model.predict(img_array)[0]
            en_yuksek_index = np.argmax(tahmin_matrisi)
            st.session_state.gecici_analiz = siniflar[en_yuksek_index]
            st.session_state.guven_orani = tahmin_matrisi[en_yuksek_index] * 100
    
    if st.session_state.gecici_analiz:
        st.info(f"🤖 Yapay Zeka Tahmini: **{st.session_state.gecici_analiz.upper()}** *(%{st.session_state.guven_orani:.1f} güven)*")
        dogrulanan_yemek = st.selectbox("Hatalıysa doğrusunu seçin:", siniflar, index=siniflar.index(st.session_state.gecici_analiz))
        
        if st.button("✅ Kalıcı Olarak Günlüğe Ekle", type="primary", use_container_width=True):
            degerler = csv_degerlerini_getir(dogrulanan_yemek)
            final_gramaj = porsiyon_miktari * porsiyon_tanimlari[dogrulanan_yemek] if giris_tipi == "🍽️ Porsiyon" else gramaj_miktari
            oran = final_gramaj / 100.0
            kayit_ekle(isim=dogrulanan_yemek.capitalize(), gramaj=int(final_gramaj), kalori=int(degerler['kalori'] * oran), protein=degerler['protein'] * oran, karbo=degerler['karbo'] * oran, yag=degerler['yag'] * oran)
            st.session_state.gecici_analiz = None 
            st.session_state.guven_orani = 0.0
            st.success("Günlüğünüze eklendi!")
            st.rerun()

st.divider()

# PLOTLY GRAFİK ÇİZİCİ
def makro_grafik(baslik, alinan, hedef, orijinal_renk):
    bar_rengi = "#FF3333" if alinan > hedef else orijinal_renk
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = alinan,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': baslik, 'font': {'size': 16, 'color': 'white'}},
        number = {'font': {'color': 'white', 'size': 24}, 'valueformat': '.1f'},
        gauge = {
            'axis': {'range': [0, max(hedef, alinan)], 'visible': False},
            'bar': {'color': bar_rengi},
            'bgcolor': "rgba(255,255,255,0.1)",
            'steps': [{'range': [0, hedef], 'color': "rgba(255,255,255,0.05)"}],
            'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': hedef}
        }))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def uyari_kutusu_olustur(alinan, hedef):
    fark = abs(hedef - alinan)
    if alinan > hedef:
        return f"<div style='text-align: center; background-color: #FF4B4B; color: white; padding: 5px; border-radius: 5px; font-weight: bold;'>⚠️ {fark:.1f}g Fazla</div>"
    else:
        return f"<div style='text-align: center; background-color: #00FA9A; color: black; padding: 5px; border-radius: 5px; font-weight: bold;'>✅ {fark:.1f}g Alınabilir</div>"


# --- 7. SEKMELİ ANALİZ ALANI ---
df_tum_kayitlar = kayitlari_getir()
tab1, tab2 = st.tabs(["📊 Bugünün Özeti", "📈 Geçmiş & Raporlar"])

with tab1:
    df_bugun = df_tum_kayitlar[df_tum_kayitlar["Tarih"] == bugun]
    
    # 1. KALORİ BARI
    toplam_kalori = df_bugun['Kalori'].sum() if not df_bugun.empty else 0
    kalan_kalori = int(hedef_kalori) - toplam_kalori
    oran_bar = min(max(toplam_kalori / hedef_kalori, 0.0), 1.0)
    
    st.write(f"**Günlük Alınan Enerji:** {toplam_kalori} / {int(hedef_kalori)} kcal")
    st.progress(oran_bar)
    
    if kalan_kalori > 0: st.success(f"💡 Hedefinize ulaşmak için hala **{kalan_kalori} kcal** yiyebilirsiniz.")
    else: st.error(f"⚠️ Günlük kalori hedefinizi **{abs(kalan_kalori)} kcal** aştınız!")
        
    # 2. MAKRO ÇEMBERLERİ
    if not df_bugun.empty:
        toplam_protein = df_bugun['Protein'].sum()
        toplam_karbo = df_bugun['Karbo'].sum()
        toplam_yag = df_bugun['Yağ'].sum()
        
        hedef_protein = (hedef_kalori * 0.30) / 4
        hedef_karbo = (hedef_kalori * 0.50) / 4
        hedef_yag = (hedef_kalori * 0.20) / 9
        
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.plotly_chart(makro_grafik("🥩 Protein", toplam_protein, hedef_protein, "#1f77b4"), use_container_width=True)
            st.markdown(uyari_kutusu_olustur(toplam_protein, hedef_protein), unsafe_allow_html=True)
        with c2: 
            st.plotly_chart(makro_grafik("🍞 Karbo", toplam_karbo, hedef_karbo, "#FFA500"), use_container_width=True)
            st.markdown(uyari_kutusu_olustur(toplam_karbo, hedef_karbo), unsafe_allow_html=True)
        with c3: 
            st.plotly_chart(makro_grafik("🥑 Yağ", toplam_yag, hedef_yag, "#800080"), use_container_width=True)
            st.markdown(uyari_kutusu_olustur(toplam_yag, hedef_yag), unsafe_allow_html=True)
    
    st.divider()

    # 3. V3.7 YENİLİK: UZMAN SİSTEM SU MOTORU (Aktiviteye Göre Dinamik Hesaplama)
    baz_su = kilo * 35
    aktif_seviye = p_veri["Aktivite"] if "Aktivite" in p_veri else "Hareketsiz / Masa başı iş"
    
    if aktif_seviye == "Hafif Aktif / Haftada 1-2 gün spor":
        hedef_su = baz_su + 500
    elif aktif_seviye == "Çok Aktif / Ağır antrenman & Kreatin Kullanımı":
        hedef_su = baz_su + 1000
    else:
        hedef_su = baz_su
        
    st.markdown(f"### 💧 Akıllı Su Takibi *(Profilinize Özel Hedef: {int(hedef_su)} ml)*")
    icilen_su = su_getir()
    oran_su = min(icilen_su / hedef_su, 1.0)
    st.progress(oran_su)
    
    if icilen_su >= hedef_su:
        st.success(f"🎉 Muhteşem! Günlük su hedefinizi tamamladınız. (Toplam: {icilen_su} ml)")
    else:
        st.info(f"Kalan miktar: **{int(hedef_su - icilen_su)} ml** (Şu ana kadar: {icilen_su} ml)")
    
    # V3.7 Yenilik: 4'lü buton düzeni ve -250ml Geri Alma Butonu
    s1, s2, s3, s4 = st.columns(4)
    if s1.button("🥤 +250 ml", use_container_width=True):
        su_ekle(250)
        st.rerun()
    if s2.button("🚰 +500 ml", use_container_width=True):
        su_ekle(500)
        st.rerun()
    if s3.button("📉 -250 ml", use_container_width=True, help="Yanlışlıkla eklediyseniz geri alın"):
        if icilen_su >= 250:
            su_ekle(-250)
            st.rerun()
        else:
            st.error("İçilen su sıfırın altına düşemez!")
    if s4.button("🔄 Suyu Sıfırla", use_container_width=True):
        su_sifirla()
        st.rerun()

    st.divider()
    
    # 4. YEDİKLERİM TABLOSU
    if not df_bugun.empty:
        st.write("📋 **Bugün Yediklerim:**")
        c1, c2, c3, c4, c5, c6, c7 = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
        c1.markdown("**İsim**")
        c2.markdown("**Gramaj**")
        c3.markdown("**Kalori**")
        c4.markdown("**Protein**")
        c5.markdown("**Karbo**")
        c6.markdown("**Yağ**")
        c7.markdown("**İşlem**")
        st.markdown("<hr style='margin: 0px; padding: 0px; border-bottom: 2px solid #444;'>", unsafe_allow_html=True)
        
        for i, row in df_bugun.iterrows():
            c1, c2, c3, c4, c5, c6, c7 = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
            c1.write(f"{row['İsim']}")
            c2.write(f"{row['Gramaj']}g")
            c3.write(f"{row['Kalori']}")
            c4.write(f"{row['Protein']}g")
            c5.write(f"{row['Karbo']}g")
            c6.write(f"{row['Yağ']}g")
            if c7.button("🗑️", key=f"sil_{i}"):
                tekil_kayit_sil(i)
                st.rerun()
            st.markdown("<hr style='margin: 0px; padding: 0px; border-bottom: 1px dotted #333;'>", unsafe_allow_html=True)
        
        st.write("") 
        with st.expander("🧹 Tüm Günü Sıfırla"):
            st.warning("Bugün eklediğiniz tüm kayıtlar silinecektir. Emin misiniz?")
            if st.button("Tüm Kayıtları Temizle", use_container_width=True):
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
        st.dataframe(df_tum_kayitlar, use_container_width=True, hide_index=True)

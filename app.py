import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
from datetime import datetime
import plotly.graph_objects as go

# Sayfa yapılandırması
st.set_page_config(page_title="AI Diyetisyen", page_icon="🥗", layout="centered")

# TensorFlow Keras sürüm uyuşmazlıkları için yama (Monkey Patching)
def clean_kwargs(kwargs):
    unsupported_keys = ['renorm', 'renorm_clipping', 'renorm_momentum', 'quantization_config']
    for key in unsupported_keys:
        kwargs.pop(key, None)
    return kwargs

_original_bn = tf.keras.layers.BatchNormalization.__init__
def _patched_bn(self, *args, **kwargs):
    _original_bn(self, *args, **clean_kwargs(kwargs))
tf.keras.layers.BatchNormalization.__init__ = _patched_bn

_original_dense = tf.keras.layers.Dense.__init__
def _patched_dense(self, *args, **kwargs):
    _original_dense(self, *args, **clean_kwargs(kwargs))
tf.keras.layers.Dense.__init__ = _patched_dense

_original_conv = tf.keras.layers.Conv2D.__init__
def _patched_conv(self, *args, **kwargs):
    _original_conv(self, *args, **clean_kwargs(kwargs))
tf.keras.layers.Conv2D.__init__ = _patched_conv

# Veri tabanı ve dosya yönetimi
PROFILES_FILE = "profiller.csv"
today = datetime.now().strftime("%Y-%m-%d")

def get_profiles():
    if os.path.exists(PROFILES_FILE):
        return pd.read_csv(PROFILES_FILE)
    return pd.DataFrame(columns=["İsim", "Sifre", "Cinsiyet", "Yaş", "Boy", "Kilo", "Hedef", "Aktivite"])

def save_profile(isim, sifre, cinsiyet, yas, boy, kilo, hedef, aktivite):
    df = get_profiles()
    df = df[df["İsim"].str.lower() != isim.lower()]
    new_profile = pd.DataFrame([{"İsim": isim, "Sifre": str(sifre), "Cinsiyet": cinsiyet, "Yaş": yas, "Boy": boy, "Kilo": kilo, "Hedef": hedef, "Aktivite": aktivite}])
    df = pd.concat([df, new_profile], ignore_index=True)
    df.to_csv(PROFILES_FILE, index=False)

def delete_profile(isim):
    if os.path.exists(PROFILES_FILE):
        df = pd.read_csv(PROFILES_FILE)
        df = df[df["İsim"] != isim]
        df.to_csv(PROFILES_FILE, index=False)
    
    history_file = f"{isim.lower()}_gecmisi.csv"
    if os.path.exists(history_file): os.remove(history_file)
        
    water_file = f"{isim.lower()}_su.csv"
    if os.path.exists(water_file): os.remove(water_file)

def get_active_history_file():
    if "aktif_kullanici" in st.session_state and st.session_state.aktif_kullanici is not None:
        return f"{st.session_state.aktif_kullanici.lower()}_gecmisi.csv"
    return "misafir_gecmisi.csv"

def get_active_water_file():
    if "aktif_kullanici" in st.session_state and st.session_state.aktif_kullanici is not None:
        return f"{st.session_state.aktif_kullanici.lower()}_su.csv"
    return "misafir_su.csv"

def get_records():
    file = get_active_history_file()
    if os.path.exists(file):
        return pd.read_csv(file)
    return pd.DataFrame(columns=["Tarih", "İsim", "Gramaj", "Kalori", "Protein", "Karbo", "Yağ"])

def add_record(isim, gramaj, kalori, protein, karbo, yag):
    file = get_active_history_file()
    df = get_records()
    new_record = pd.DataFrame([{
        "Tarih": today, "İsim": isim, "Gramaj": gramaj, "Kalori": kalori, 
        "Protein": round(protein, 1), "Karbo": round(karbo, 1), "Yağ": round(yag, 1)
    }])
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(file, index=False)

def delete_single_record(idx):
    file = get_active_history_file()
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = df.drop(index=idx)
        df.to_csv(file, index=False)

def reset_today():
    file = get_active_history_file()
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = df[df["Tarih"] != today]
        df.to_csv(file, index=False)

# Su takip modülü
def get_water():
    file = get_active_water_file()
    if os.path.exists(file):
        df = pd.read_csv(file)
        today_df = df[df["Tarih"] == today]
        return max(0, today_df["Miktar"].sum()) if not today_df.empty else 0
    return 0

def add_water(amount):
    file = get_active_water_file()
    df = pd.read_csv(file) if os.path.exists(file) else pd.DataFrame(columns=["Tarih", "Miktar"])
    new_record = pd.DataFrame([{"Tarih": today, "Miktar": amount}])
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(file, index=False)

def reset_water():
    file = get_active_water_file()
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = df[df["Tarih"] != today]
        df.to_csv(file, index=False)

# Tıbbi onay motoru
def check_medical_guardrails(boy, kilo, hedef):
    vki = kilo / ((boy / 100) ** 2)
    if vki < 18.5 and hedef == "Kilo Vermek": 
        return False, "🚨 TIBBİ UYARI: Vücut Kitle İndeksiniz 'Zayıf'. Sağlığınız için 'Kilo Vermek' hedefini onaylamıyoruz!"
    elif vki >= 25.0 and hedef == "Kilo Almak": 
        return False, "🚨 TIBBİ UYARI: Vücut Kitle İndeksiniz yüksek. Sağlığınız için 'Kilo Almak' hedefini onaylamıyoruz!"
    return True, ""

# Model yükleme
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("yemek_modeli.h5", compile=False)

try: 
    model = load_model()
except Exception as e: 
    st.error(f"Model yükleme hatası: {e}")

classes = ['pizza', 'hamburger', 'sushi'] 
portion_defs = {"hamburger": 200, "pizza": 300, "sushi": 150}
activity_options = ["Hareketsiz / Masa başı iş", "Hafif Aktif / Haftada 1-2 gün spor", "Çok Aktif / Ağır antrenman & Kreatin Kullanımı"]

if "gecici_analiz" not in st.session_state: st.session_state.gecici_analiz = None
if "guven_orani" not in st.session_state: st.session_state.guven_orani = 0.0

# Kimlik doğrulama ve oturum
if "aktif_kullanici" not in st.session_state:
    st.session_state.aktif_kullanici = None

with st.sidebar:
    if st.session_state.aktif_kullanici is None:
        st.header("🔐 Sisteme Giriş")
        action = st.radio("İşlem seçin:", ["Giriş Yap", "Yeni Kayıt Ol"], horizontal=True)
        st.divider()
        
        if action == "Giriş Yap":
            username_input = st.text_input("👤 Kullanıcı Adı:")
            password_input = st.text_input("🔑 Şifre:", type="password")
            if st.button("🚀 Giriş Yap", use_container_width=True):
                df_prof = get_profiles()
                match = df_prof[(df_prof["İsim"].str.lower() == username_input.lower().strip()) & (df_prof["Sifre"].astype(str) == password_input.strip())]
                if not match.empty:
                    st.session_state.aktif_kullanici = match.iloc[0]["İsim"]
                    st.rerun()
                else: 
                    st.error("Kullanıcı adı veya şifre hatalı!")
                    
        else:
            new_username = st.text_input("👤 Yeni Kullanıcı Adı:")
            new_password = st.text_input("🔑 Bir Şifre Belirleyin:", type="password")
            gender = st.radio("Cinsiyet:", ["Erkek", "Kadın"])
            age = st.number_input("Yaş:", 15, 100, 22)
            height = st.number_input("Boy (cm):", 100, 250, 175)
            weight = st.number_input("Kilo (kg):", 30, 200, 75)
            target = st.selectbox("Hedefiniz:", ["Kilo Korumak", "Kilo Vermek", "Kilo Almak"])
            new_activity = st.selectbox("Fiziksel Aktivite Seviyeniz:", activity_options)
            
            if st.button("💾 Kayıt Ol", use_container_width=True):
                df_prof = get_profiles()
                if new_username.strip().lower() in df_prof["İsim"].str.lower().values: 
                    st.error("⚠️ Bu kullanıcı adı alınmış.")
                elif len(new_username.strip()) == 0 or len(new_password.strip()) == 0: 
                    st.error("⚠️ Boş bırakılamaz!")
                else:
                    is_approved, msg = check_medical_guardrails(height, weight, target)
                    if is_approved:
                        save_profile(new_username.strip(), new_password.strip(), gender, age, height, weight, target, new_activity)
                        st.session_state.aktif_kullanici = new_username.strip()
                        st.rerun()
                    else: 
                        st.error(msg)

    else:
        active_user = st.session_state.aktif_kullanici
        df_prof = get_profiles()
        user_data = df_prof[df_prof["İsim"] == active_user].iloc[0]
        
        st.header(f"⚙️ {active_user} Profili")
        gender = st.radio("Cinsiyet:", ["Erkek", "Kadın"], index=0 if user_data["Cinsiyet"] == "Erkek" else 1)
        age = st.number_input("Yaş:", 15, 100, int(user_data["Yaş"]))
        height = st.number_input("Boy (cm):", 100, 250, int(user_data["Boy"]))
        weight = st.number_input("Kilo (kg):", 30, 200, int(user_data["Kilo"]))
        target = st.selectbox("Hedefiniz:", ["Kilo Korumak", "Kilo Vermek", "Kilo Almak"], index=["Kilo Korumak", "Kilo Vermek", "Kilo Almak"].index(user_data["Hedef"]))
        
        act_idx = activity_options.index(user_data["Aktivite"]) if "Aktivite" in user_data and user_data["Aktivite"] in activity_options else 0
        activity = st.selectbox("Fiziksel Aktivite Seviyeniz:", activity_options, index=act_idx)
        
        if st.button("🔄 Bilgilerimi Güncelle", use_container_width=True):
            is_approved, msg = check_medical_guardrails(height, weight, target)
            if is_approved:
                save_profile(active_user, user_data["Sifre"], gender, age, height, weight, target, activity)
                st.success("Güncellendi!")
                st.rerun()
            else: 
                st.error(msg)

        # Dinamik BMR (Bazal Metabolizma) ve Kalori Hesaplama
        if gender == "Erkek": 
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else: 
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
            
        daily_need = bmr * 1.375 
        
        if target == "Kilo Vermek": 
            target_calories = daily_need - 500
        elif target == "Kilo Almak": 
            target_calories = daily_need + 400
        else: 
            target_calories = daily_need
            
        vki = weight / ((height/100)**2)
        if vki < 18.5: vki_status = "Zayıf"
        elif vki < 25.0: vki_status = "Normal"
        elif vki < 30.0: vki_status = "Fazla Kilolu"
        else: vki_status = "Obez"
        
        st.divider()
        st.metric("⚖️ VKI", f"{vki:.1f} ({vki_status})")
        st.metric("🎯 Günlük Hedef", f"{int(target_calories)} kcal")
        
        st.divider()
        st.markdown("### 🔒 Oturum Yönetimi")
        if st.button("🚪 Çıkış Yap", use_container_width=True):
            st.session_state.aktif_kullanici = None
            st.rerun()
            
        with st.expander("🚨 Hesabı Sil"):
            if st.button("🗑️ Kalıcı Olarak Sil", type="primary", use_container_width=True):
                delete_profile(active_user)
                st.session_state.aktif_kullanici = None
                st.rerun()

if st.session_state.aktif_kullanici is None:
    st.markdown("<h2 style='text-align: center; margin-top: 100px;'>🔒 AI Diyetisyene Hoş Geldiniz</h2>", unsafe_allow_html=True)
    st.info("👈 Lütfen sol menüden giriş yapın veya kayıt olun.")
    st.stop()

# Ana Ekran
st.markdown(f"<h1 style='text-align: center;'>🥗 Hoş Geldin, {st.session_state.aktif_kullanici}!</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("### 📸 Yeni Öğün Ekle")
input_type = st.radio("Miktar Türü:", ["🍽️ Porsiyon", "📏 Gramaj"], horizontal=True)

if input_type == "🍽️ Porsiyon": 
    portion_amount = st.slider("Miktar (Porsiyon):", 0.5, 5.0, 1.0, 0.5)
else: 
    gram_amount = st.slider("Miktar (Gram):", 50, 1500, 200, 50)
    
uploaded_image = st.file_uploader("Yemek fotoğrafı yükleyin:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

def get_csv_values(food_name):
    try:
        df = pd.read_csv("nutrition_dataset.csv")
        df['food_name'] = df['food_name'].str.lower()
        match = df[df['food_name'].str.contains(food_name.lower(), na=False)]
        if not match.empty:
            row = match.iloc[0]
            return {"kalori": float(row['calories']), "protein": float(row['protein']), "karbo": float(row['carbs']), "yag": float(row['fat'])}
    except: 
        pass
    fallback_data = {
        "hamburger": {"kalori": 295, "protein": 14, "karbo": 24, "yag": 14}, 
        "pizza": {"kalori": 266, "protein": 11, "karbo": 33, "yag": 10}, 
        "sushi": {"kalori": 143, "protein": 4, "karbo": 32, "yag": 4}
    }
    return fallback_data.get(food_name, {"kalori": 100, "protein": 5, "karbo": 15, "yag": 2})

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption="Analiz Edilen Görsel", use_container_width=True)
    
    if st.button("🔍 Analiz Et", use_container_width=True):
        with st.spinner("Model analiz ediyor..."):
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = model.predict(img_array)[0]
            highest_idx = np.argmax(predictions)
            st.session_state.gecici_analiz = classes[highest_idx]
            st.session_state.guven_orani = predictions[highest_idx] * 100
    
    if st.session_state.gecici_analiz:
        st.info(f"🤖 Model Tahmini: **{st.session_state.gecici_analiz.upper()}** *(%{st.session_state.guven_orani:.1f} güven)*")
        verified_food = st.selectbox("Hatalıysa düzeltin:", classes, index=classes.index(st.session_state.gecici_analiz))
        
        if st.button("✅ Günlüğe Ekle", type="primary", use_container_width=True):
            vals = get_csv_values(verified_food)
            final_gram = portion_amount * portion_defs[verified_food] if input_type == "🍽️ Porsiyon" else gram_amount
            ratio = final_gram / 100.0
            add_record(
                isim=verified_food.capitalize(), 
                gramaj=int(final_gram), 
                kalori=int(vals['kalori'] * ratio), 
                protein=vals['protein'] * ratio, 
                karbo=vals['karbo'] * ratio, 
                yag=vals['yag'] * ratio
            )
            st.session_state.gecici_analiz = None 
            st.session_state.guven_orani = 0.0
            st.success("Başarıyla eklendi.")
            st.rerun()

st.divider()

# Grafikler
def create_gauge(title, current, target, color):
    bar_color = "#FF3333" if current > target else color
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = current,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16, 'color': 'white'}},
        number = {'font': {'color': 'white', 'size': 24}, 'valueformat': '.1f'},
        gauge = {
            'axis': {'range': [0, max(target, current)], 'visible': False},
            'bar': {'color': bar_color},
            'bgcolor': "rgba(255,255,255,0.1)",
            'steps': [{'range': [0, target], 'color': "rgba(255,255,255,0.05)"}],
            'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': target}
        }))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def create_warning(current, target):
    diff = abs(target - current)
    if current > target:
        return f"<div style='text-align: center; background-color: #FF4B4B; color: white; padding: 5px; border-radius: 5px; font-weight: bold;'>⚠️ {diff:.1f}g Fazla</div>"
    return f"<div style='text-align: center; background-color: #00FA9A; color: black; padding: 5px; border-radius: 5px; font-weight: bold;'>✅ {diff:.1f}g Kaldı</div>"

# Sekmeler
df_all_records = get_records()
tab1, tab2 = st.tabs(["📊 Günlük Özet", "📈 Geçmiş Raporlar"])

with tab1:
    df_today = df_all_records[df_all_records["Tarih"] == today]
    
    # Kalori Barı
    total_cals = df_today['Kalori'].sum() if not df_today.empty else 0
    remaining_cals = int(target_calories) - total_cals
    progress_ratio = min(max(total_cals / target_calories, 0.0), 1.0)
    
    st.write(f"**Günlük Alınan Enerji:** {total_cals} / {int(target_calories)} kcal")
    st.progress(progress_ratio)
    
    if remaining_cals > 0: 
        st.success(f"💡 Kalan kalori hakkınız: **{remaining_cals} kcal**")
    else: 
        st.error(f"⚠️ Günlük kalori hedefini **{abs(remaining_cals)} kcal** aştınız.")
        
    # Makro Grafikleri
    if not df_today.empty:
        total_protein = df_today['Protein'].sum()
        total_carbs = df_today['Karbo'].sum()
        total_fat = df_today['Yağ'].sum()
        
        target_protein = (target_calories * 0.30) / 4
        target_carbs = (target_calories * 0.50) / 4
        target_fat = (target_calories * 0.20) / 9
        
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.plotly_chart(create_gauge("🥩 Protein", total_protein, target_protein, "#1f77b4"), use_container_width=True)
            st.markdown(create_warning(total_protein, target_protein), unsafe_allow_html=True)
        with c2: 
            st.plotly_chart(create_gauge("🍞 Karbo", total_carbs, target_carbs, "#FFA500"), use_container_width=True)
            st.markdown(create_warning(total_carbs, target_carbs), unsafe_allow_html=True)
        with c3: 
            st.plotly_chart(create_gauge("🥑 Yağ", total_fat, target_fat, "#800080"), use_container_width=True)
            st.markdown(create_warning(total_fat, target_fat), unsafe_allow_html=True)
    
    st.divider()

    # Su Motoru
    base_water = weight * 35
    act_level = user_data["Aktivite"] if "Aktivite" in user_data else "Hareketsiz / Masa başı iş"
    
    if act_level == "Hafif Aktif / Haftada 1-2 gün spor":
        target_water = base_water + 500
    elif act_level == "Çok Aktif / Ağır antrenman & Kreatin Kullanımı":
        target_water = base_water + 1000
    else:
        target_water = base_water
        
    st.markdown(f"### 💧 Su Takibi *(Hedef: {int(target_water)} ml)*")
    drank_water = get_water()
    water_ratio = min(drank_water / target_water, 1.0)
    st.progress(water_ratio)
    
    if drank_water >= target_water:
        st.success(f"🎉 Günlük su hedefine ulaşıldı. ({drank_water} ml)")
    else:
        st.info(f"Kalan: **{int(target_water - drank_water)} ml** (İçilen: {drank_water} ml)")
    
    s1, s2, s3, s4 = st.columns(4)
    if s1.button("🥤 +250 ml", use_container_width=True):
        add_water(250)
        st.rerun()
    if s2.button("🚰 +500 ml", use_container_width=True):
        add_water(500)
        st.rerun()
    if s3.button("📉 -250 ml", use_container_width=True):
        if drank_water >= 250:
            add_water(-250)
            st.rerun()
    if s4.button("🔄 Sıfırla", use_container_width=True):
        reset_water()
        st.rerun()

    st.divider()
    
    # Günlük Tablo
    if not df_today.empty:
        st.write("📋 **Bugün Tüketilenler:**")
        c1, c2, c3, c4, c5, c6, c7 = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
        c1.markdown("**İsim**")
        c2.markdown("**Gramaj**")
        c3.markdown("**Kalori**")
        c4.markdown("**Protein**")
        c5.markdown("**Karbo**")
        c6.markdown("**Yağ**")
        c7.markdown("**İşlem**")
        st.markdown("<hr style='margin: 0px; padding: 0px; border-bottom: 2px solid #444;'>", unsafe_allow_html=True)
        
        for i, row in df_today.iterrows():
            c1, c2, c3, c4, c5, c6, c7 = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
            c1.write(f"{row['İsim']}")
            c2.write(f"{row['Gramaj']}g")
            c3.write(f"{row['Kalori']}")
            c4.write(f"{row['Protein']}g")
            c5.write(f"{row['Karbo']}g")
            c6.write(f"{row['Yağ']}g")
            if c7.button("🗑️", key=f"del_{i}"):
                delete_single_record(i)
                st.rerun()
            st.markdown("<hr style='margin: 0px; padding: 0px; border-bottom: 1px dotted #333;'>", unsafe_allow_html=True)
        
        st.write("") 
        with st.expander("🧹 Günlüğü Temizle"):
            if st.button("Tüm Kayıtları Sil", use_container_width=True):
                reset_today()
                st.rerun()

with tab2:
    if df_all_records.empty:
        st.info("Geçmiş veri bulunmamaktadır.")
    else:
        st.markdown("### 📅 Günlük Kalori Tüketimi")
        daily_cals = df_all_records.groupby("Tarih")["Kalori"].sum()
        st.bar_chart(daily_cals, color="#ff4b4b")
        st.markdown("### 📂 Tüm Veriler")
        st.dataframe(df_all_records, use_container_width=True, hide_index=True)

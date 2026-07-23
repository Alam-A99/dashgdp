# ─────────────────────────────────────────────────────────────────────
# RUN: Bismillah
# ─────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from io import BytesIO
import re

st.set_page_config(page_title="📊 Dashboard PDB", layout="wide", page_icon="📈")

# 1. LOAD DATA

@st.cache_data(ttl=3600)
def load_data_from_github(raw_url):
    try:
        res = requests.get(raw_url, timeout=10)
        res.raise_for_status()
        df = pd.read_excel(BytesIO(res.content), header=0)
        df.rename(columns={df.columns[0]: "Sektor"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        st.stop()

@st.cache_data
def generate_dummy_data():
    """dataset dummy 17 sektor lapangan usaha (ADHK)."""
    np.random.seed(42)
    sektor_list = [
        "A. Pertanian, Kehutanan, dan Perikanan",
        "B. Pertambangan dan Penggalian",
        "C. Industri Pengolahan",
        "D. Pengadaan Listrik dan Gas",
        "E. Pengadaan Air, Pengelolaan Sampah",
        "F. Konstruksi",
        "G. Perdagangan Besar dan Eceran",
        "H. Transportasi dan Pergudangan",
        "I. Penyediaan Akomodasi dan Makan Minum",
        "J. Informasi dan Komunikasi",
        "K. Jasa Keuangan dan Asuransi",
        "L. Real Estate",
        "M,N. Jasa Perusahaan",
        "O. Administrasi Pemerintahan",
        "P. Jasa Pendidikan",
        "Q. Jasa Kesehatan dan Kegiatan Sosial",
        "R,S,T,U. Jasa Lainnya",
    ]
    years = list(range(2019, 2025))
    q_cols = [f"{y}_{q}" for y in years for q in range(1, 5)]
    y_cols = [str(y) for y in years]

    base_levels = np.random.uniform(500, 8000, size=len(sektor_list))
    annual_growth = np.random.uniform(-0.02, 0.09, size=len(sektor_list))  # laju tumbuh tahunan dasar
    seasonal = np.array([0.98, 1.00, 1.01, 1.02])  # pola musiman kuartalan ringan

    rows = []
    for i, sek in enumerate(sektor_list):
        row = {"Sektor": sek}
        level = base_levels[i]
        q_values = {}
        for t, col in enumerate(q_cols):
            year_idx = t // 4
            q_idx = t % 4
            # shock khusus 2020 (mensimulasikan efek pandemi) untuk sektor rentan
            shock = 1.0
            if "2020" in col and sek in ["I. Penyediaan Akomodasi dan Makan Minum", "H. Transportasi dan Pergudangan"]:
                shock = 0.72
            growth_factor = (1 + annual_growth[i]) ** (1/4)
            noise = np.random.normal(1.0, 0.015)
            level = level * growth_factor * seasonal[q_idx] * shock * noise
            q_values[col] = round(level, 1)
        row.update(q_values)
        for y in years:
            yq = [row[f"{y}_{q}"] for q in range(1, 5)]
            row[str(y)] = round(sum(yq), 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = ["Sektor"] + q_cols + y_cols
    return df[ordered_cols]

# 2. FUNGSI PREPARASI DATA
def prepare_data(df):
    all_cols = [c for c in df.columns[1:] if isinstance(c, (str, int))]
    q_cols = sorted([c for c in all_cols if re.match(r"^\d{4}_[1-4]$", str(c))], 
                    key=lambda x: (int(str(x).split('_')[0]), int(str(x).split('_')[1])))
    y_cols = sorted([c for c in all_cols if re.match(r"^\d{4}$", str(c))], key=lambda x: int(str(x)))
    sektor_list = df["Sektor"].dropna().tolist()
    return df, q_cols, y_cols, sektor_list

def get_filtered_cols(df, selected_freqs, q_cols, y_cols):
    if "📊 Tahunan" in selected_freqs:
        return y_cols, "Tahunan"
    cols = []
    labels = []
    for freq in selected_freqs:
        q_num = int(freq.split()[-1])
        q_cols_filtered = sorted([c for c in q_cols if str(c).endswith(f"_{q_num}")], 
                                  key=lambda x: int(str(x).split('_')[0]))
        cols.extend(q_cols_filtered)
        labels.append(f"Q{q_num}")
    cols = sorted(cols, key=lambda x: (int(str(x).split('_')[0]), int(str(x).split('_')[1])))
    label_str = " + ".join(labels) if len(labels) > 1 else labels[0] if labels else "Kuartal"
    return cols, label_str

def calculate_growth_data(df, cols, selected_sectors):
    growth_data = []
    for sek in selected_sectors:
        data = df.loc[df["Sektor"] == sek, cols].values.flatten()
        data = data[~np.isnan(data)]
        if len(data) < 2: continue
        g_latest = (data[-1] - data[-2]) / data[-2] * 100
        g_series = pd.Series(data).pct_change().dropna() * 100
        growth_data.append({
            "Sektor": sek, 
            "Growth_Terkini_%": g_latest, 
            "Nilai_Terkini_M": data[-1],
            "Avg_Growth_%": g_series.mean(), 
            "Volatility_σ": g_series.std(),
            "Nilai_Awal_M": data[-2]
        })
    return pd.DataFrame(growth_data).sort_values("Growth_Terkini_%", ascending=False)

def forecast_sector(data, n_forecast=4):
    """Regresi linear sederhana atas data historis untuk proyeksi n periode ke depan,
    plus interval keyakinan berbasis residual (±1.96 SE)."""
    x = np.arange(len(data))
    if len(data) < 3:
        return None
    coeffs = np.polyfit(x, data, 1)
    trend = np.poly1d(coeffs)
    residuals = data - trend(x)
    se = np.std(residuals, ddof=2) if len(data) > 2 else np.std(residuals)
    x_future = np.arange(len(data), len(data) + n_forecast)
    y_future = trend(x_future)
    ci = 1.96 * se
    # R^2 sebagai indikator kekuatan tren
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return {
        "x_future": x_future, "y_future": y_future, "ci": ci,
        "slope": coeffs[0], "r2": r2, "se": se
    }

# 3. SIDEBAR & KONTROL
st.title("📊 Dashboard Analisis PDB")
st.caption("Eksplorasi real-time dataset PDB menurut Lapangan Usaha (ADHK)")

with st.sidebar:
    st.header("⚙️ Konfigurasi")

    # 🎲 SUMBER DATA — dummy sebagai default agar dashboard langsung bisa dipakai
    data_source = st.radio(
        "🗂️ Sumber Data",
        ["🎲 Dataset (Default)", "🔗 URL GitHub"],
        index=0,
        help="Data dummy memuat 17 sektor contoh agar dashboard bisa langsung dieksplorasi tanpa perlu URL."
    )

    if data_source == "🎲 Dataset (Default)":
        df = generate_dummy_data()
        st.success("✅ Menggunakan data (17 sektor)")
    else:
        raw_url = st.text_input(
            "🔗 DATA URL",
            value="input link here",
            help="Gunakan URL 'raw' "
        )
        if not raw_url or "raw" not in raw_url:
            st.warning("⚠️ Masukkan URL yang valid terlebih dahulu.")
            st.stop()
        df = load_data_from_github(raw_url)

    df, q_cols, y_cols, sektor_list = prepare_data(df)
    
    # 📅 Multi-Select Periode
    freq_options = ["Triwulan 1", "Triwulan 2", "Triwulan 3", "Triwulan 4", "📊 Tahunan"]
    selected_freqs = st.multiselect("📅 Pilih Periode", freq_options, default=["Triwulan 1"])
    if not selected_freqs:
        st.warning("⚠️ Pilih minimal satu periode.")
        st.stop()
    
    # 🏷️ Multi-Select Sektor
    selected_sectors = st.multiselect("🏷️ Pilih Sektor", sektor_list, 
                                      default=[sektor_list[0], sektor_list[1] if len(sektor_list)>1 else sektor_list[0]])
    
    # 👁️ Jenis Visualisasi
    viz_type = st.selectbox("👁️ Jenis Visualisasi", [
        "📈 Tren Nilai", "🏆 Perubahan Terbesar (17)", "🔥 Volatilitas", 
        "📊 Komparasi Q1-Q4", "🔍 Deteksi Anomali"
    ])
    
    # 🎚️ Threshold
    threshold = st.slider("🎯 Threshold (%)", 1.0, 20.0, 5.0)
    
    # 📐 PENGATURAN TINGGI & LEBAR GRAFIK (DIPERTAHANKAN)
    st.divider()
    st.subheader("📐 Pengaturan Visual")
    chart_height = st.slider("📏 Tinggi Grafik (pixel)", min_value=300, max_value=1200, value=600, step=50)
    chart_width = st.slider("📐 Lebar Grafik (pixel)", min_value=600, max_value=2000, value=1200, step=100)

# 4. PROSES DATA & VISUALISASI
cols, freq_label = get_filtered_cols(df, selected_freqs, q_cols, y_cols)
df_growth = calculate_growth_data(df, cols, selected_sectors)

tabs = st.tabs(["📊 Visualisasi", "📋 Tabel Perubahan", "🔮 Prediksi & Indikator"])

with tabs[0]:
    st.subheader(f"📈 Analisis: {viz_type} ({freq_label})")
    
    # ─── VISUALISASI: PERUBAHAN TERBESAR (17 SEKTOR) ─────────────────
    if viz_type == "🏆 Perubahan Terbesar (17)":
        df_top17 = df_growth.head(17)
        if df_top17.empty:
            st.warning("⚠️ Tidak cukup data untuk 17 sektor.")
        else:
            # 📊 CHART BAR HORIZONTAL
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_top17["Sektor"].apply(lambda x: x[:45]+"..." if len(x)>45 else x),
                x=df_top17["Growth_Terkini_%"], orientation='h',
                marker_color=df_top17["Growth_Terkini_%"].apply(lambda v: "#2A9D8F" if v >= 0 else "#E63946"),
                marker_line=dict(width=1, color='white'),
                hovertemplate="<b>%{y}</b><br>Perubahan: %{x:.2f}%<br>Nilai: Rp %{customdata:,.0f} M<extra></extra>",
                customdata=df_top17["Nilai_Terkini_M"]
            ))
            fig.update_layout(
                title=f"🏆 Ranking 17 Sektor: Perubahan Terbesar - {freq_label}",
                xaxis_title="Pertumbuhan Periode Terakhir (%)",
                yaxis=dict(autorange="reversed", title="Sektor"),
                template="plotly_white", 
                height=chart_height,
                width=chart_width,
                margin=dict(l=320, r=20, t=50, b=20), 
                bargap=0.4, 
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=False)
            
            # 📋 TABEL RANKING 17 SEKTOR
            st.markdown("### 📋 Tabel Ranking 17 Sektor - Perubahan Terbesar & Terkecil")
            
            # Siapkan data tabel dengan formatting
            df_table = df_top17.copy()
            df_table["Rank"] = range(1, len(df_table)+1)
            df_table["Tren"] = df_table["Growth_Terkini_%"].apply(lambda x: "📈 Naik" if x >= 0 else "📉 Turun")
            df_table["Perubahan_Absolut_M"] = df_table["Nilai_Terkini_M"] - df_table["Nilai_Awal_M"]
            
            # Format kolom untuk display
            df_display = df_table[[
                "Rank", "Sektor", "Tren", "Growth_Terkini_%", 
                "Nilai_Awal_M", "Nilai_Terkini_M", "Perubahan_Absolut_M",
                "Avg_Growth_%", "Volatility_σ"
            ]].copy()
            
            # Formatting angka
            df_display["Growth_Terkini_%"] = df_display["Growth_Terkini_%"].apply(lambda x: f"{x:+.2f}%")
            df_display["Nilai_Awal_M"] = df_display["Nilai_Awal_M"].apply(lambda x: f"{x:,.0f}")
            df_display["Nilai_Terkini_M"] = df_display["Nilai_Terkini_M"].apply(lambda x: f"{x:,.0f}")
            df_display["Perubahan_Absolut_M"] = df_display["Perubahan_Absolut_M"].apply(lambda x: f"{x:+,.0f}")
            df_display["Avg_Growth_%"] = df_display["Avg_Growth_%"].apply(lambda x: f"{x:+.2f}%")
            df_display["Volatility_σ"] = df_display["Volatility_σ"].apply(lambda x: f"{x:.2f}%")
            
            # Rename kolom untuk display yang lebih jelas
            df_display.columns = [
                "🔢", "🏷️ Sektor", "📊 Tren", "📈 Growth (%)", 
                "💰 Nilai Awal (M)", "💰 Nilai Akhir (M)", "🔄 Δ Absolut (M)",
                "📊 Avg Growth", "⚡ Volatilitas"
            ]
            
            # Tampilkan tabel dengan styling
            st.dataframe(
                df_display,
                use_container_width=True,
                height=500,
                hide_index=True
            )
            
            # 📜 STATISTIK FORMAT
            st.markdown("### 📊 Ringkasan Statistik Detail")
            for _, row in df_top17.iterrows():
                tren = "📈 Ekspansi" if row["Growth_Terkini_%"] >= 0 else "📉 Kontraksi"
                with st.expander(f"🏷️ {row['Sektor'][:50]}{'...' if len(row['Sektor'])>50 else ''}"):
                    st.text(f"   ├─ Nilai Terkini: Rp {row['Nilai_Terkini_M']:,.0f} M")
                    st.text(f"   ├─ Growth Terbaru: {row['Growth_Terkini_%']:+.2f}%")
                    st.text(f"   ├─ Rata-rata Growth: {row['Avg_Growth_%']:+.2f}%")
                    st.text(f"   ├─ Volatilitas (σ): {row['Volatility_σ']:.2f}%")
                    st.text(f"   └─ Tren: {tren}")

    # ─── VISUALISASI: TREN NILAI ────────────────────────────────────
    elif viz_type == "📈 Tren Nilai":
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        for i, sek in enumerate(selected_sectors):
            data = df.loc[df["Sektor"] == sek, cols].values.flatten()
            data = data[~np.isnan(data)]
            periods = [f"{str(c).split('_')[0]} Q{str(c).split('_')[1]}" if '_' in str(c) else str(c) for c in cols]
            fig.add_trace(go.Scatter(x=periods[:len(data)], y=data, mode="lines+markers",
                                     name=sek[:40]+"...", line=dict(color=colors[i%len(colors)], width=2.5)))
        fig.update_layout(
            title=f"📈 Tren Nilai - {freq_label}", 
            xaxis_title="Periode", 
            yaxis_title="Nilai (Miliar Rupiah)",
            template="plotly_white", 
            height=chart_height,
            width=chart_width,
            hovermode="x unified", 
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=False)

    # ─── VISUALISASI: VOLATILITAS ───────────────────────────────────
    elif viz_type == "🔥 Volatilitas":
        vol_data = df_growth.sort_values("Volatility_σ", ascending=False).head(15)
        fig = px.bar(vol_data, x="Volatility_σ", y="Sektor", orientation='h',
                     title=f"🔥 15 Sektor Paling Volatil - {freq_label}", 
                     color="Volatility_σ", color_continuous_scale="RdYlGn_r")
        fig.update_layout(
            height=chart_height,
            width=chart_width,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=False)

    # ─── VISUALISASI: KOMPARASI Q1-Q4 ───────────────────────────────
    elif viz_type == "📊 Komparasi Q1-Q4":
        if "📊 Tahunan" in selected_freqs or len(selected_freqs) < 2:
            st.info("💡 Pilih minimal 2 Triwulan untuk komparasi.")
        else:
            latest_year = max(int(str(c).split('_')[0]) for c in cols)
            comp_data = []
            for sek in selected_sectors[:10]:
                for c in cols:
                    if str(c).startswith(f"{latest_year}_"):
                        val = df.loc[df["Sektor"] == sek, c].values[0]
                        if not pd.isna(val):
                            q = str(c).split('_')[1]
                            comp_data.append({"Sektor": sek[:35]+"..." if len(sek)>35 else sek, "Kuartal": f"Q{q}", "Nilai_M": val})
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                fig = px.bar(df_comp, x="Sektor", y="Nilai_M", color="Kuartal", barmode="group",
                             title=f"📊 Komparasi Nilai per Kuartal - Tahun {latest_year}",
                             labels={"Nilai_M": "Nilai (Miliar Rupiah)", "Sektor": "Sektor"},
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(
                    height=chart_height,
                    width=chart_width,
                    xaxis_tickangle=-45, 
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=False)
            else:
                st.warning("⚠️ Data tidak tersedia untuk komparasi.")

    # ─── VISUALISASI: DETEKSI ANOMALI ───────────────────────────────
    elif viz_type == "🔍 Deteksi Anomali":
        st.info(f"🔍 Threshold: {threshold}%")
        for sek in selected_sectors[:5]:
            data = df.loc[df["Sektor"] == sek, cols].values.flatten()
            data = data[~np.isnan(data)]
            if len(data) < 10: continue
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            anomalies = np.where(z_scores > 2)[0]
            status = "✅ Stabil" if len(anomalies) == 0 else f"⚠️ {len(anomalies)} anomali"
            st.markdown(f"**{sek[:50]}**: {status}")
        
        if selected_sectors:
            sek = selected_sectors[0]
            data = df.loc[df["Sektor"] == sek, cols].values.flatten()
            data = data[~np.isnan(data)]
            periods = [str(c) for c in cols][:len(data)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=periods, y=data, mode="lines+markers", name="Nilai"))
            if len(data) >= 10:
                z = np.abs((data - np.mean(data)) / np.std(data))
                anom_idx = np.where(z > 2)[0]
                if len(anom_idx) > 0:
                    fig.add_trace(go.Scatter(
                        x=[periods[i] for i in anom_idx], 
                        y=[data[i] for i in anom_idx],
                        mode="markers", 
                        name="⚠️ Anomali",
                        marker=dict(color="red", size=10, symbol="x")
                    ))
            fig.update_layout(
                title=f"🔍 Deteksi Anomali - {sek[:40]}",
                height=chart_height,
                width=chart_width,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=False)

with tabs[1]:
    st.subheader("📋 Tabel Perubahan Kuartalan/Tahunan")
    change_rows = []
    for sek in selected_sectors:
        data = df.loc[df["Sektor"] == sek, cols].values.flatten()
        data = data[~np.isnan(data)]
        periods = [str(c) for c in cols]
        for i in range(1, len(data)):
            change_rows.append({
                "Sektor": sek, "Periode_Dari": periods[i-1], "Periode_Ke": periods[i],
                "Nilai_Awal_M": data[i-1], "Nilai_Akhir_M": data[i],
                "Perubahan_%": ((data[i]-data[i-1])/data[i-1]*100) if data[i-1] != 0 else 0
            })
    df_changes = pd.DataFrame(change_rows)
    if not df_changes.empty:
        df_display = df_changes[["Sektor", "Periode_Dari", "Periode_Ke", "Nilai_Awal_M", "Nilai_Akhir_M", "Perubahan_%"]].copy()
        df_display["Perubahan_%"] = df_display["Perubahan_%"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(df_display, use_container_width=True, height=500)
    else:
        st.info("Pilih minimal 1 sektor dengan data lengkap.")

# ─── TAB BARU: PREDIKSI & INDIKATOR PROFESIONAL ──────────────────────
with tabs[2]:
    st.subheader(f"🔮 Prediksi Tren & Indikator Profesional ({freq_label})")

    n_forecast = st.slider("📆 Jumlah Periode Proyeksi ke Depan", 1, 8, 4)

    if not selected_sectors:
        st.info("Pilih minimal 1 sektor pada sidebar untuk melihat prediksi.")
    else:
        total_terkini = df.loc[df["Sektor"].isin(selected_sectors), cols[-1]].sum()

        # ── Kartu indikator ringkas per sektor ──
        st.markdown("### 📌 Indikator Ringkas")
        n_show = min(len(selected_sectors), 4)
        metric_cols = st.columns(n_show)
        for i, sek in enumerate(selected_sectors[:n_show]):
            data = df.loc[df["Sektor"] == sek, cols].values.flatten()
            data = data[~np.isnan(data)]
            if len(data) < 3:
                continue
            g_series = pd.Series(data).pct_change().dropna() * 100
            momentum = g_series.iloc[-1] - g_series.mean()  # akselerasi vs rata-rata historis
            share = data[-1] / total_terkini * 100 if total_terkini else 0
            with metric_cols[i]:
                st.metric(
                    label=sek[:28] + ("..." if len(sek) > 28 else ""),
                    value=f"{data[-1]:,.0f} M",
                    delta=f"{g_series.iloc[-1]:+.2f}% growth"
                )
                st.caption(f"⚡ Momentum: {momentum:+.2f} pp · 🥧 Kontribusi: {share:.1f}%")

        st.divider()

        # ── Proyeksi tren linear per sektor ──
        st.markdown("### 📈 Proyeksi Tren (Regresi Linear + Interval Keyakinan 95%)")
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        forecast_summary = []
        for i, sek in enumerate(selected_sectors):
            data = df.loc[df["Sektor"] == sek, cols].values.flatten()
            data = data[~np.isnan(data)]
            if len(data) < 3:
                continue
            fc = forecast_sector(data, n_forecast)
            if fc is None:
                continue
            periods_hist = list(range(len(data)))
            color = colors[i % len(colors)]
            label = sek[:35] + ("..." if len(sek) > 35 else "")

            # historis
            fig.add_trace(go.Scatter(
                x=periods_hist, y=data, mode="lines+markers",
                name=f"{label} (aktual)", line=dict(color=color, width=2.5)
            ))
            # proyeksi
            fig.add_trace(go.Scatter(
                x=list(fc["x_future"]), y=list(fc["y_future"]), mode="lines+markers",
                name=f"{label} (proyeksi)", line=dict(color=color, width=2, dash="dash")
            ))
            # pita keyakinan
            fig.add_trace(go.Scatter(
                x=list(fc["x_future"]) + list(fc["x_future"])[::-1],
                y=list(fc["y_future"] + fc["ci"]) + list(fc["y_future"] - fc["ci"])[::-1],
                fill="toself", fillcolor=color, opacity=0.12,
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))

            arah = "📈 Naik" if fc["slope"] > 0 else "📉 Turun"
            forecast_summary.append({
                "Sektor": sek,
                "Nilai_Terkini_M": data[-1],
                f"Proyeksi_+{n_forecast}_Periode_M": fc["y_future"][-1],
                "Slope_per_Periode_M": fc["slope"],
                "Kekuatan_Tren_R²": fc["r2"],
                "Arah": arah
            })

        fig.update_layout(
            title="🔮 Proyeksi Nilai Sektor Terpilih",
            xaxis_title="Indeks Periode (historis → proyeksi)",
            yaxis_title="Nilai (Miliar Rupiah)",
            template="plotly_white",
            height=chart_height,
            width=chart_width,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=False)

        # ── Tabel ringkasan proyeksi ──
        if forecast_summary:
            st.markdown("### 📋 Ringkasan Proyeksi & Kekuatan Tren")
            df_fc = pd.DataFrame(forecast_summary)
            df_fc_display = df_fc.copy()
            df_fc_display["Nilai_Terkini_M"] = df_fc_display["Nilai_Terkini_M"].apply(lambda x: f"{x:,.0f}")
            df_fc_display[f"Proyeksi_+{n_forecast}_Periode_M"] = df_fc_display[f"Proyeksi_+{n_forecast}_Periode_M"].apply(lambda x: f"{x:,.0f}")
            df_fc_display["Slope_per_Periode_M"] = df_fc_display["Slope_per_Periode_M"].apply(lambda x: f"{x:+,.1f}")
            df_fc_display["Kekuatan_Tren_R²"] = df_fc_display["Kekuatan_Tren_R²"].apply(lambda x: f"{x:.2f}")
            st.dataframe(df_fc_display, use_container_width=True, hide_index=True)
            st.caption("💡 R² mendekati 1 menandakan tren historis konsisten/kuat sehingga proyeksi lebih andal; R² rendah menandakan pergerakan sektor lebih fluktuatif.")

        st.divider()

        # ── Indikator konsentrasi sektor (HHI) ──
        st.markdown("### 🥧 Indikator Konsentrasi Ekonomi (Herfindahl-Hirschman Index)")
        shares = df.loc[df["Sektor"].isin(selected_sectors), cols[-1]]
        shares_pct = shares / shares.sum() * 100 if shares.sum() else shares
        hhi = np.sum(shares_pct ** 2)
        if hhi < 1500:
            hhi_label = "🟢 Terdiversifikasi"
        elif hhi < 2500:
            hhi_label = "🟡 Konsentrasi Sedang"
        else:
            hhi_label = "🔴 Terkonsentrasi Tinggi"
        c1, c2 = st.columns(2)
        c1.metric("HHI (Sektor Terpilih)", f"{hhi:,.0f}", hhi_label)
        c2.metric("Jumlah Sektor Dianalisis", f"{len(selected_sectors)}")
        st.caption("💡 HHI dihitung dari pangsa nilai sektor terpilih pada periode terakhir (skala 0–10.000). Semakin tinggi nilainya, semakin terkonsentrasi struktur ekonominya pada sedikit sektor.")

st.caption("💡 Keep on Learning in wwww.dataaksi.id with ❤️")

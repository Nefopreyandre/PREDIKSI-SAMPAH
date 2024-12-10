import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.dates as mdates

st.title("Prediksi Jumlah Sampah dan Jenis Sampah Menggunakan ARIMA")

# File uploader
uploaded_file = st.file_uploader("Upload file Excel Anda", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, sheet_name='Sheet1')

    # Preprocessing
    data['Tanggal'] = pd.to_datetime(data['Tanggal'])

    # Tampilkan data
    st.subheader("Data yang Diunggah")
    st.dataframe(data.head())

    # Dapatkan kombinasi unik dari 'Provinsi' dan 'Jenis Sampah'
    unique_combinations = data[['Provinsi', 'Jenis Sampah']].drop_duplicates()

    # Set periode prediksi
    forecast_2024_steps = 12
    forecast_2024_dates = pd.date_range(start='2024-01-01', periods=forecast_2024_steps, freq='MS')

    forecast_results = []

    # Loop setiap kombinasi
    for _, row in unique_combinations.iterrows():
        provinsi = row['Provinsi']
        jenis_sampah = row['Jenis Sampah']

        subset_data = data[(data['Provinsi'] == provinsi) & (data['Jenis Sampah'] == jenis_sampah)]
        subset_data = subset_data.sort_values('Tanggal')

        # Split data ke dalam train dan test
        train_subset = subset_data[(subset_data['Tanggal'] >= '2020-01-01') & (subset_data['Tanggal'] <= '2022-12-31')]
        test_subset = subset_data[(subset_data['Tanggal'] >= '2023-01-01') & (subset_data['Tanggal'] <= '2023-12-31')]

        # Visualisasi rolling mean dan std
        rolling_window = 12
        rolling_mean = train_subset['Jumlah Jenis Sampah Perbulan (Ton)'].rolling(window=rolling_window).mean()
        rolling_std = train_subset['Jumlah Jenis Sampah Perbulan (Ton)'].rolling(window=rolling_window).std()

        st.subheader(f"Rata-rata & Std Bergerak untuk {provinsi} - {jenis_sampah}")
        plt.figure(figsize=(12, 6))
        plt.plot(train_subset['Tanggal'], train_subset['Jumlah Jenis Sampah Perbulan (Ton)'], label='Original')
        plt.plot(train_subset['Tanggal'], rolling_mean, color='red', label=f'Rolling Mean ({rolling_window} bulan)')
        plt.plot(train_subset['Tanggal'], rolling_std, color='green', label=f'Rolling Std ({rolling_window} bulan)')
        plt.title(f'Rolling Mean & Std untuk {provinsi} - {jenis_sampah}')
        plt.legend(loc='best')
        plt.grid(True)
        st.pyplot(plt)

        # Uji ADF
        adf_test = adfuller(train_subset['Jumlah Jenis Sampah Perbulan (Ton)'])
        st.write(f"Statistik Uji ADF untuk {provinsi} - {jenis_sampah}: {adf_test[0]}, p-value: {adf_test[1]}")

        # Visualisasi ACF dan PACF dengan differencing jika diperlukan
        st.subheader(f"ACF & PACF untuk {provinsi} - {jenis_sampah}")
        try:
            # Cek jika data tidak stasioner, maka lakukan differencing
            if adf_test[1] > 0.05:  # Jika p-value lebih besar dari 0.05, data tidak stasioner
                differenced_data = train_subset['Jumlah Jenis Sampah Perbulan (Ton)'].diff().dropna()
                st.write("Data tidak stasioner, melakukan differencing...")
            else:
                differenced_data = train_subset['Jumlah Jenis Sampah Perbulan (Ton)']

            # Tentukan jumlah maksimal lags yang bisa digunakan (maksimal setengah ukuran data)
            max_lags = min(20, len(differenced_data) // 2)

            # Plot ACF
            fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
            plot_acf(differenced_data, lags=max_lags, ax=ax_acf)
            plt.title(f"ACF untuk {provinsi} - {jenis_sampah}")
            plt.grid(True)
            st.pyplot(fig_acf)

            # Plot PACF
            fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
            plot_pacf(differenced_data, lags=max_lags, ax=ax_pacf)
            plt.title(f"PACF untuk {provinsi} - {jenis_sampah}")
            plt.grid(True)
            st.pyplot(fig_pacf)

        except ValueError as e:
            st.write(f"Error dalam plot ACF/PACF: {e}")

        # Latih model ARIMA
        try:
            model_subset = ARIMA(train_subset['Jumlah Jenis Sampah Perbulan (Ton)'], order=(1, 2, 2))
            model_fit_subset = model_subset.fit()

            forecast_2023_subset = model_fit_subset.forecast(steps=len(test_subset))
            mape = mean_absolute_percentage_error(test_subset['Jumlah Jenis Sampah Perbulan (Ton)'], forecast_2023_subset) * 100
            st.write(f"MAPE untuk {provinsi} - {jenis_sampah} tahun 2023: {mape:.2f}%")

            # Simpan hasil prediksi
            for date, actual, predicted in zip(test_subset['Tanggal'], test_subset['Jumlah Jenis Sampah Perbulan (Ton)'], forecast_2023_subset):
                forecast_results.append({
                    'Provinsi': provinsi,
                    'Jenis Sampah': jenis_sampah,
                    'Tanggal': date,
                    'Actual_Jumlah_Sampah': actual,
                    'Predicted_Jumlah_Sampah_2023': predicted
                })

            # Visualisasi perbandingan data aktual dan prediksi 2023
            plt.figure(figsize=(12, 6))
            plt.plot(test_subset['Tanggal'], test_subset['Jumlah Jenis Sampah Perbulan (Ton)'], label=f'Actual {jenis_sampah}', color='blue')
            plt.plot(test_subset['Tanggal'], forecast_2023_subset, label=f'Predicted {jenis_sampah}', color='red', linestyle='--')
            plt.title(f'Perbandingan Prediksi dengan Data Aktual (2023) - {jenis_sampah}')
            plt.xlabel('Tanggal')
            plt.ylabel('Jumlah Sampah (Ton)')
            plt.legend(loc='best')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            st.pyplot(plt)

            # Prediksi untuk 2024
            forecast_2024_subset = model_fit_subset.forecast(steps=forecast_2024_steps)
            for month, prediction in zip(forecast_2024_dates, forecast_2024_subset):
                forecast_results.append({
                    'Provinsi': provinsi,
                    'Jenis Sampah': jenis_sampah,
                    'Tanggal': month,
                    'Predicted_Jumlah_Sampah': prediction
                })

        except Exception as e:
            st.write(f"Error memproses {provinsi} - {jenis_sampah}: {e}")
            continue

    # Konversi ke DataFrame
    forecast_df = pd.DataFrame(forecast_results)

    # Tampilkan hasil prediksi untuk 2024
    unique_provinces = forecast_df['Provinsi'].unique()
    for province in unique_provinces:
        st.subheader(f'Prediksi Jumlah Sampah untuk {province} Tahun 2024')
        subset_df_2024 = forecast_df[(forecast_df['Provinsi'] == province) & (forecast_df['Tanggal'].dt.year == 2024)]

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=subset_df_2024, x='Tanggal', y='Predicted_Jumlah_Sampah', hue='Jenis Sampah', marker='o')
        plt.title(f'Prediksi Jumlah Sampah untuk {province} Tahun 2024')
        plt.grid(True)
        st.pyplot(plt)

    # Tampilkan hasil evaluasi untuk 2023
    forecast_2023_df = pd.DataFrame([result for result in forecast_results if 'Actual_Jumlah_Sampah' in result])
    if not forecast_2023_df.empty:
        st.subheader("Tabel Prediksi Tahun 2023 (dengan nilai aktual)")
        st.dataframe(forecast_2023_df[['Provinsi', 'Jenis Sampah', 'Tanggal', 'Actual_Jumlah_Sampah', 'Predicted_Jumlah_Sampah_2023']])

    # Visualisasi Total Sampah per Tahun
    st.subheader("Visualisasi Total Sampah Tahun 2023 dan Prediksi 2024")

    # Hitung total sampah per bulan untuk tahun 2023 (aktual dan prediksi)
    total_sampah_2023 = forecast_2023_df.groupby('Tanggal')[['Actual_Jumlah_Sampah', 'Predicted_Jumlah_Sampah_2023']].sum()

    # Visualisasi total sampah tahun 2023
    plt.figure(figsize=(12, 6))
    plt.plot(total_sampah_2023.index, total_sampah_2023['Actual_Jumlah_Sampah'], label='Total Actual Sampah 2023', color='blue')
    plt.plot(total_sampah_2023.index, total_sampah_2023['Predicted_Jumlah_Sampah_2023'], label='Total Predicted Sampah 2023', color='red', linestyle='--')
    plt.title('Total Sampah Tahun 2023: Actual vs Prediksi')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Sampah (Ton)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    st.pyplot(plt)

    # Total Sampah Prediksi untuk Tahun 2024
    total_sampah_2024 = forecast_df[forecast_df['Tanggal'].dt.year == 2024].groupby('Tanggal')['Predicted_Jumlah_Sampah'].sum()

    # Visualisasi total sampah prediksi untuk tahun 2024
    plt.figure(figsize=(12, 6))
    plt.plot(total_sampah_2024.index, total_sampah_2024, label='Total Predicted Sampah 2024', color='green', marker='o')
    plt.title('Prediksi Total Sampah Tahun 2024')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Sampah (Ton)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    st.pyplot(plt)

    # Tampilkan nilai MAPE untuk setiap provinsi dan jenis sampah di tabel
    st.subheader("Tabel Evaluasi MAPE per Provinsi dan Jenis Sampah")
    if not forecast_2023_df.empty:
        forecast_2023_df['MAPE'] = forecast_2023_df.apply(lambda row: mean_absolute_percentage_error([row['Actual_Jumlah_Sampah']], [row['Predicted_Jumlah_Sampah_2023']]), axis=1)
        st.dataframe(forecast_2023_df[['Provinsi', 'Jenis Sampah', 'Tanggal', 'Actual_Jumlah_Sampah', 'Predicted_Jumlah_Sampah_2023', 'MAPE']])

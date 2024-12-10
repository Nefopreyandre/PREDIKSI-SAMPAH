import streamlit as st

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

def main():
    # Dataframe selection
    st.markdown("<h1 align='center'> <b> Sistem Prediksi Limbah Sampah Masyarat Provinsi Aceh", unsafe_allow_html=True)
    new_line(1)
    st.markdown("Aplikasi prediksi ini dirancang untuk menyederhanakan proses membangun dan mengevaluasi model ARIMA Agar Lebih Mudah.", unsafe_allow_html=True)
    
    st.divider()
    
    # Overview
    new_line()
    st.markdown("<h2 style='text-align: center;'>PROSES PREDIKSI</h2>", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""
    Ketika membangun model Prediksi, ada serangkaian langkah untuk menyiapkan data dan membangun model. Berikut ini adalah langkah-langkah utama dalam proses Machine Learning:
    
    -  Pengumpulan Data : <div>
                        Proses pengumpulan data dari sumber terpercaya yaitu Website Sistem Informasi Pengelolaan Sampah Nasional (SIPSN) dan Badan Pusat Statistik (BPS).<br> <br>
    -  Data Cleaning :  <div>
                        Proses pembersihan data dengan menghapus duplikasi, menangani nilai yang hilang dll. Langkah ini sangat penting karena seringkali data tidak bersih dan mengandung banyak nilai yang hilang dan outlier. <br> <br>
    -  Data Preprocessing : <div>
                        Proses mengubah data ke dalam format yang sesuai untuk analisis. Hal ini termasuk menangani fitur kategorikal, fitur numerik, penskalaan dan transformasi, dll. <br> <br>
    -  Splitting the Data : <div>
                        Proses membagi data menjadi set pelatihan, validasi, dan pengujian. Set pelatihan digunakan untuk melatih model, set validasi digunakan untuk menyetel hiperparameter, dan set pengujian digunakan untuk mengevaluasi model. <br> <br>
    -  Building Machine Learning Models : <div>
                        Model yang digunakan pada aplikasi ini adalah ARIMA (AutoRegressive Integrated Moving Average). Model ARIMA sangat populer dalam analisis deret waktu untuk memprediksi data masa depan berdasarkan pola dari data historis. <br> <br>
    -  Evaluating Machine Learning Models : <div>
                        Proses mengevaluasi model prediksi dengan menggunakan metrik Mean Absolute Percentage Error (MAPE). <br> <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Pada bagian membangun model, pengguna memasukkan nilai masing-masing hyperparameter. Hiperparameter adalah variabel yang secara signifikan mempengaruhi proses pelatihan model:
    
    - P : Adalah parameter berupa nilai integer yang menentukan jumlah lag pengamatan untuk model ARIMA. <br> <br>
    - D : Adalah jumlah diferensiasi yang diperlukan untuk membuat data menjadi stasioner, jika data stasioner maka nilai D adalah 0. <br> <br>
    - Q : Adalah jumlah lag dari komponen moving average. <br> <br>
    """, unsafe_allow_html=True)
    new_line()
    

    
    st.markdown("""

    
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

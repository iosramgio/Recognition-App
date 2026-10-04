# BISINDO Sign Language Recognition App

Aplikasi desktop untuk menerjemahkan isyarat BISINDO secara real-time dari sarung tangan IoT menjadi teks dan suara, menggunakan model CNN-LSTM.

## Cara Kerja

1. Menerima data sensor (flex + IMU) dari ESP32 via MQTT
2. Menyusun data ke dalam sliding window (100 timestep × 11 fitur)
3. Menormalisasi data dan menjalankan inferensi model CNN-LSTM
4. Menampilkan hasil klasifikasi sebagai teks (jika confidence ≥ 70%)
5. Mengucapkan hasil klasifikasi via Google Text-to-Speech (gTTS)

## Requirements

- customtkinter
- paho-mqtt
- tensorflow
- numpy
- joblib
- matplotlib
- gTTS
- pygame


## File yang Dibutuhkan


- `model_bisindo_final.keras` — model CNN-LSTM terlatih
- `scaler.pkl` — parameter normalisasi Z-Score
- `label_encoder.pkl` — encoder label kelas

## Konfigurasi

| Parameter | Nilai |
|---|---|
| MQTT Broker | `localhost:1883` |
| MQTT Topic | `bisindo/sensor/data` |
| Timesteps | 100 frame |
| Jumlah Fitur | 11 (5 flex + 6 IMU) |
| Confidence Threshold | 0.70 |
| Jumlah Kelas | 10 kosakata BISINDO |

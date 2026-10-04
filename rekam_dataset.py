import paho.mqtt.client as mqtt
import json
import csv
import os
import time

# 1. PENGATURAN MQTT & DATASET
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "bisindo/sensor/data"

NAMA_FILE_CSV = "dataset_bisindo_mentah.csv"
FRAMES_PER_GESTURE = 100  # 100 frame @ 50Hz = 1 Detik perekaman per isyarat

# Variabel Global untuk Kontrol Perekaman
is_recording = False
frames_recorded = 0
current_label = ""
current_sequence_id = 0
temp_data_buffer = []

# 2. BUAT FILE CSV & HEADER
def inisialisasi_csv():
    file_exists = os.path.isfile(NAMA_FILE_CSV)
    with open(NAMA_FILE_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Header kolom dataset
            writer.writerow(['sequence_id', 'label', 'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5', 
                             'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
            print(f"📄 File {NAMA_FILE_CSV} baru berhasil dibuat!")

# 3. FUNGSI PENYADAP MQTT (BERJALAN DI BACKGROUND)
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"❌ Gagal terhubung ke Broker! (Kode: {rc})")

def on_message(client, userdata, msg):
    global is_recording, frames_recorded, temp_data_buffer
    
    if is_recording and frames_recorded < FRAMES_PER_GESTURE:
        try:
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            
            f = data.get('flex', [0, 0, 0, 0, 0])
            a = data.get('accel', [0, 0, 0])
            g = data.get('gyro', [0, 0, 0])
            
            baris_data = [current_sequence_id, current_label, 
                          f[0], f[1], f[2], f[3], f[4], 
                          a[0], a[1], a[2], 
                          g[0], g[1], g[2]]
            
            temp_data_buffer.append(baris_data)
            frames_recorded += 1
            
            if frames_recorded % 10 == 0:
                print(f"   [+] Terrekam: {frames_recorded}/{FRAMES_PER_GESTURE} frame...")
                
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            # ✅ PERBAIKAN: Tangkap error lebih spesifik untuk debugging
            print(f"   [!] Error parsing data: {e}")
            pass
            
    elif is_recording and frames_recorded >= FRAMES_PER_GESTURE:
        is_recording = False
        simpan_ke_csv()

def simpan_ke_csv():
    global temp_data_buffer
    with open(NAMA_FILE_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(temp_data_buffer)
    print(f"✅ Tersimpan! ({len(temp_data_buffer)} baris data untuk isyarat '{current_label}')\n")
    temp_data_buffer = []  # Kosongkan buffer untuk rekaman selanjutnya

# 4. PROGRAM UTAMA (KONTROL USER)
inisialisasi_csv()

# Jalankan MQTT di background (Thread terpisah)
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start() 

print("\n" + "="*50)
print("🎬 SISTEM PEREKAMAN DATASET BISINDO (METODE MENGALIR)")
print("="*50)

try:
    while True:
        # User input kata yang mau direkam 
        kata = input("\nMasukkan kata BISINDO (atau ketik 'exit' untuk keluar): ").strip().upper()
        
        if kata == 'EXIT':
            break
        if kata == "":
            continue
            
        current_label = kata
        current_sequence_id = int(time.time()) # Pakai waktu sbg ID unik rekaman
        
        print(f"⏳ SIAP-SIAP! Posisikan tanganmu rileks di depan dada.")
        input("👉 Tekan [ENTER] untuk mulai merekam, dan langsung peragakan isyaratnya!")
        
        # Mulai merekam
        print("🔴 MEREKAM (1 Detik)... Peragakan sekarang!")
        frames_recorded = 0
        temp_data_buffer = [] # Pastikan buffer kosong sebelum mulai
        is_recording = True
        
        # Timer Timeout
        waktu_mulai = time.time()
        
        # Tunggu sampai MQTT selesai merekam 100 frame ATAU waktu habis
        while is_recording:
            time.sleep(0.1)
            
            # Kalau sudah lebih dari 5 detik tapi belum 100 frame, BATALKAN!
            if time.time() - waktu_mulai > 5.0:
                print(f"\n⚠️ WAKTU HABIS! Rekaman macet di {frames_recorded}/{FRAMES_PER_GESTURE}.")
                print("🔄 Kemungkinan sinyal Wi-Fi putus atau ESP32 hang. Silakan ulangi.")
                is_recording = False
                temp_data_buffer = [] # Buang data yang tidak lengkap
                break

except KeyboardInterrupt:
    print("\n🛑 Dihentikan paksa.")
finally:
    client.loop_stop()
    print("👋 Perekaman selesai. File tersimpan di:", NAMA_FILE_CSV)
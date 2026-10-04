"""
============================================================
APLIKASI DESKTOP — SISTEM PENGENALAN ISYARAT BISINDO
============================================================
Skripsi: Rancang Bangun Sistem Pengenalan Isyarat BISINDO
         Berbasis Sarung Tangan IoT Menggunakan CNN-LSTM

Fitur:
  - Real-time MQTT subscriber (multithreading)
  - Inferensi model CNN-LSTM
  - Output teks + suara (gTTS)
  - Grafik probabilitas semua kelas (live bar chart)
  - Status sensor real-time
  - Riwayat 5 isyarat terakhir

Struktur file yang dibutuhkan (1 folder):
  app_bisindo.py
  model_bisindo_final.keras
  scaler.pkl
  label_encoder.pkl

Install library:
  pip install customtkinter paho-mqtt tensorflow
  pip install joblib numpy gtts pygame matplotlib
============================================================
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import queue
import time
import os
import sys
import numpy as np
import joblib
from collections import deque
import json

import paho.mqtt.client as mqtt
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from gtts import gTTS
import pygame
import tempfile

def resource_path(filename):
    """Selalu mengarah ke folder tempat .exe (atau .py) berada, apapun working directory-nya."""
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)

# ============================================================
# KONFIGURASI
# ============================================================
MQTT_BROKER   = "localhost"
MQTT_PORT     = 1883
MQTT_TOPIC    = "bisindo/sensor/data"

MODEL_PATH    = resource_path("model_bisindo_final.keras")
SCALER_PATH   = resource_path("scaler.pkl")
ENCODER_PATH  = resource_path("label_encoder.pkl")

TIMESTEPS     = 100
N_FEATURES    = 11
CONFIDENCE_THRESHOLD = 0.70   # minimal 70% confidence untuk ditampilkan



FEATURE_COLS = [
    'flex_1','flex_2','flex_3','flex_4','flex_5',
    'acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'
]

LABEL_NAMES = [
    'Apa Kabar','Baik','Hallo','Kamu','Maaf',
    'Sama-Sama','Sampai Jumpa','Saya','Terima Kasih','Tolong'
]

# Warna tema terang
THEME = {
    'bg'          : '#F5F5F5',
    'card'        : '#FFFFFF',
    'primary'     : '#1565C0',
    'primary_light': '#E3F2FD',
    'accent'      : '#2E7D32',
    'accent_light': '#E8F5E9',
    'warning'     : '#F57F17',
    'error'       : '#C62828',
    'text'        : '#212121',
    'text_light'  : '#757575',
    'border'      : '#E0E0E0',
    'connected'   : '#2E7D32',
    'disconnected': '#C62828',
}


# ============================================================
# KELAS UTAMA APLIKASI
# ============================================================
class BISINDOApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ── Konfigurasi Jendela ────────────────────────────
        self.title("Sistem Pengenalan Isyarat BISINDO — Sarung Tangan IoT")
        self.geometry("1200x750")
        self.minsize(1100, 680)
        self.configure(fg_color=THEME['bg'])

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # ── State Aplikasi ─────────────────────────────────
        self.is_running      = False
        self.mqtt_connected  = False
        self.sensor_buffer   = deque(maxlen=TIMESTEPS)
        self.result_queue    = queue.Queue()
        self.history_list    = []
        self.last_prediction = ""
        self.frame_count     = 0
        self.last_probs      = np.zeros(len(LABEL_NAMES))

        # ── Load Model & Scaler ────────────────────────────
        self.model   = None
        self.scaler  = None
        self.encoder = None
        self._load_model()

        # ── Inisialisasi Audio ─────────────────────────────
        pygame.mixer.init()

        # ── Bangun UI ──────────────────────────────────────
        self._build_ui()

        # ── Update loop ────────────────────────────────────
        self._update_ui_loop()

        # ── Handle close ──────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ============================================================
    # LOAD MODEL
    # ============================================================
    def _load_model(self):
        try:
            self.model   = tf.keras.models.load_model(MODEL_PATH)
            self.scaler  = joblib.load(SCALER_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
            print("✅ Model, scaler, dan encoder berhasil dimuat!")
        except Exception as e:
            messagebox.showerror(
                "Error Loading Model",
                f"Gagal memuat file model:\n{e}\n\n"
                "Pastikan file berikut ada dalam 1 folder:\n"
                "• model_bisindo_final.keras\n"
                "• scaler.pkl\n"
                "• label_encoder.pkl"
            )

    # ============================================================
    # BANGUN UI
    # ============================================================
    def _build_ui(self):
        # ── Header ────────────────────────────────────────
        self._build_header()

        # ── Content Area (2 kolom) ─────────────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        # Kolom kiri
        left = ctk.CTkFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(1, weight=1)

        self._build_result_card(left)
        self._build_sensor_card(left)

        # Kolom kanan
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.grid_rowconfigure(0, weight=1)

        self._build_chart_card(right)
        self._build_history_card(right)

    # ── Header ────────────────────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=THEME['primary'],
                              corner_radius=0, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        # Judul
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=20)

        ctk.CTkLabel(
            title_frame,
            text="🤟 Sistem Pengenalan Isyarat BISINDO",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="white"
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame,
            text="  |  Sarung Tangan IoT × CNN-LSTM",
            font=ctk.CTkFont(size=12),
            text_color="#90CAF9"
        ).pack(side="left")

        # Kontrol kanan
        ctrl = ctk.CTkFrame(header, fg_color="transparent")
        ctrl.pack(side="right", padx=20)

        # Status MQTT
        self.lbl_mqtt_status = ctk.CTkLabel(
            ctrl,
            text="⬤  Terputus",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#EF9A9A"
        )
        self.lbl_mqtt_status.pack(side="left", padx=(0, 16))

        # Tombol Start/Stop
        self.btn_start = ctk.CTkButton(
            ctrl,
            text="▶  Mulai",
            width=110, height=36,
            fg_color="#2E7D32",
            hover_color="#1B5E20",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle_system
        )
        self.btn_start.pack(side="left", padx=4)

        # Tombol Reset History
        ctk.CTkButton(
            ctrl,
            text="🗑  Reset",
            width=80, height=36,
            fg_color="#546E7A",
            hover_color="#37474F",
            font=ctk.CTkFont(size=12),
            command=self._reset_history
        ).pack(side="left", padx=4)

    # ── Card Hasil Isyarat ─────────────────────────────────
    def _build_result_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=THEME['card'],
                            corner_radius=12,
                            border_width=1, border_color=THEME['border'])
        card.pack(fill="x", pady=(0, 8))

        # Label judul
        ctk.CTkLabel(
            card,
            text="HASIL PENGENALAN",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME['text_light']
        ).pack(anchor="w", padx=20, pady=(16, 4))

        # Label hasil utama
        self.lbl_result = ctk.CTkLabel(
            card,
            text="—",
            font=ctk.CTkFont(size=52, weight="bold"),
            text_color=THEME['primary']
        )
        self.lbl_result.pack(pady=(4, 0))

        # Label confidence
        self.lbl_confidence = ctk.CTkLabel(
            card,
            text="Keyakinan model: —",
            font=ctk.CTkFont(size=13),
            text_color=THEME['text_light']
        )
        self.lbl_confidence.pack(pady=(0, 4))

        # Progress bar confidence
        self.bar_confidence = ctk.CTkProgressBar(
            card, width=320, height=10,
            fg_color=THEME['border'],
            progress_color=THEME['accent']
        )
        self.bar_confidence.set(0)
        self.bar_confidence.pack(pady=(0, 16))

    # ── Card Status Sensor ─────────────────────────────────
    def _build_sensor_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=THEME['card'],
                            corner_radius=12,
                            border_width=1, border_color=THEME['border'])
        card.pack(fill="x", pady=(0, 0))

        ctk.CTkLabel(
            card,
            text="STATUS SENSOR REAL-TIME",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME['text_light']
        ).pack(anchor="w", padx=20, pady=(14, 8))

        sensor_grid = ctk.CTkFrame(card, fg_color="transparent")
        sensor_grid.pack(fill="x", padx=20, pady=(0, 14))

        # Flex sensors
        flex_frame = ctk.CTkFrame(sensor_grid, fg_color=THEME['primary_light'],
                                  corner_radius=8)
        flex_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        ctk.CTkLabel(flex_frame, text="Flex Sensor",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=THEME['primary']).pack(pady=(8, 4))

        self.flex_bars = []
        flex_names = ["Ibu", "Telunjuk", "Tengah", "Manis", "Kelingking"]
        for i, name in enumerate(flex_names):
            row = ctk.CTkFrame(flex_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(row, text=f"{name:10s}",
                         font=ctk.CTkFont(size=9),
                         text_color=THEME['text'], width=60).pack(side="left")
            bar = ctk.CTkProgressBar(row, height=8,
                                     fg_color=THEME['border'],
                                     progress_color=THEME['primary'])
            bar.set(0)
            bar.pack(side="left", fill="x", expand=True, padx=4)
            self.flex_bars.append(bar)

        ctk.CTkFrame(flex_frame, fg_color="transparent", height=8).pack()

        # IMU sensors
        imu_frame = ctk.CTkFrame(sensor_grid, fg_color=THEME['accent_light'],
                                 corner_radius=8)
        imu_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

        ctk.CTkLabel(imu_frame, text="IMU MPU-6500",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=THEME['accent']).pack(pady=(8, 4))

        self.imu_labels = {}
        imu_names = ["Acc X","Acc Y","Acc Z","Gyr X","Gyr Y","Gyr Z"]
        for name in imu_names:
            row = ctk.CTkFrame(imu_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(row, text=f"{name}:",
                         font=ctk.CTkFont(size=9),
                         text_color=THEME['text'], width=40).pack(side="left")
            lbl = ctk.CTkLabel(row, text="0.000",
                               font=ctk.CTkFont(size=9, weight="bold"),
                               text_color=THEME['accent'])
            lbl.pack(side="left")
            self.imu_labels[name] = lbl

        ctk.CTkFrame(imu_frame, fg_color="transparent", height=8).pack()

        # Buffer progress
        buf_frame = ctk.CTkFrame(card, fg_color="transparent")
        buf_frame.pack(fill="x", padx=20, pady=(0, 14))

        ctk.CTkLabel(buf_frame, text="Buffer window:",
                     font=ctk.CTkFont(size=10),
                     text_color=THEME['text_light']).pack(side="left")

        self.lbl_buffer = ctk.CTkLabel(
            buf_frame, text="0/100 frame",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME['primary']
        )
        self.lbl_buffer.pack(side="left", padx=8)

        self.bar_buffer = ctk.CTkProgressBar(
            buf_frame, height=8,
            fg_color=THEME['border'],
            progress_color=THEME['primary']
        )
        self.bar_buffer.set(0)
        self.bar_buffer.pack(side="left", fill="x", expand=True)

    # ── Card Bar Chart Probabilitas ────────────────────────
    def _build_chart_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=THEME['card'],
                            corner_radius=12,
                            border_width=1, border_color=THEME['border'])
        card.pack(fill="both", expand=True, pady=(0, 8))

        ctk.CTkLabel(
            card,
            text="DISTRIBUSI PROBABILITAS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME['text_light']
        ).pack(anchor="w", padx=20, pady=(14, 4))

        # Matplotlib figure
        self.fig = Figure(figsize=(4, 3.2), dpi=96,
                          facecolor=THEME['card'])
        self.ax  = self.fig.add_subplot(111)
        self._init_chart()

        self.canvas = FigureCanvasTkAgg(self.fig, master=card)
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=12, pady=(0, 12))

    def _init_chart(self):
        self.ax.clear()
        colors = ['#E3F2FD'] * len(LABEL_NAMES)
        self.bars = self.ax.barh(
            LABEL_NAMES,
            np.zeros(len(LABEL_NAMES)),
            color=colors, edgecolor='none', height=0.6
        )
        self.ax.set_xlim(0, 1)
        self.ax.set_xlabel('Probabilitas', fontsize=8,
                           color=THEME['text_light'])
        self.ax.tick_params(axis='y', labelsize=8,
                            colors=THEME['text'])
        self.ax.tick_params(axis='x', labelsize=7,
                            colors=THEME['text_light'])
        self.ax.set_facecolor(THEME['card'])
        self.fig.patch.set_facecolor(THEME['card'])
        for spine in self.ax.spines.values():
            spine.set_color(THEME['border'])
        self.ax.xaxis.label.set_color(THEME['text_light'])
        self.fig.tight_layout(pad=1.5)

    def _update_chart(self, probs):
        max_idx = np.argmax(probs)
        colors  = []
        for i, p in enumerate(probs):
            if i == max_idx and p >= CONFIDENCE_THRESHOLD:
                colors.append(THEME['primary'])
            elif p > 0.1:
                colors.append('#90CAF9')
            else:
                colors.append('#E3F2FD')

        for bar, p, c in zip(self.bars, probs, colors):
            bar.set_width(p)
            bar.set_color(c)

        self.canvas.draw_idle()

    # ── Card Riwayat ───────────────────────────────────────
    def _build_history_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=THEME['card'],
                            corner_radius=12,
                            border_width=1, border_color=THEME['border'])
        card.pack(fill="x")

        ctk.CTkLabel(
            card,
            text="RIWAYAT ISYARAT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME['text_light']
        ).pack(anchor="w", padx=20, pady=(14, 8))

        self.history_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.history_frame.pack(fill="x", padx=16, pady=(0, 14))

        self._render_history()

    def _render_history(self):
        for w in self.history_frame.winfo_children():
            w.destroy()

        if not self.history_list:
            ctk.CTkLabel(
                self.history_frame,
                text="Belum ada isyarat terdeteksi...",
                font=ctk.CTkFont(size=11),
                text_color=THEME['text_light']
            ).pack()
            return

        for item in reversed(self.history_list[-5:]):
            row = ctk.CTkFrame(self.history_frame,
                               fg_color=THEME['primary_light'],
                               corner_radius=6, height=30)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)

            ctk.CTkLabel(
                row,
                text=f"🤟  {item['label']}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=THEME['primary']
            ).pack(side="left", padx=12)

            ctk.CTkLabel(
                row,
                text=f"{item['conf']*100:.1f}%  •  {item['time']}",
                font=ctk.CTkFont(size=10),
                text_color=THEME['text_light']
            ).pack(side="right", padx=12)

    # ============================================================
    # KONTROL SISTEM
    # ============================================================
    def _toggle_system(self):
        if not self.is_running:
            self._start_system()
        else:
            self._stop_system()

    def _start_system(self):
        if self.model is None:
            messagebox.showerror("Error", "Model belum dimuat!")
            return

        self.is_running = True
        self.btn_start.configure(
            text="■  Stop",
            fg_color=THEME['error'],
            hover_color="#8B0000"
        )

        # Start MQTT thread
        self.mqtt_thread = threading.Thread(
            target=self._mqtt_worker, daemon=True)
        self.mqtt_thread.start()

    def _stop_system(self):
        self.is_running = False
        self.mqtt_connected = False
        self.btn_start.configure(
            text="▶  Mulai",
            fg_color="#2E7D32",
            hover_color="#1B5E20"
        )
        self.lbl_mqtt_status.configure(
            text="⬤  Terputus", text_color="#EF9A9A")
        try:
            self.mqtt_client.disconnect()
        except:
            pass

    def _reset_history(self):
        self.history_list  = []
        self.last_prediction = ""
        self.lbl_result.configure(text="—", text_color=THEME['primary'])
        self.lbl_confidence.configure(text="Keyakinan model: —")
        self.bar_confidence.set(0)
        self._init_chart()
        self.canvas.draw()
        self._render_history()

    # ============================================================
    # MQTT WORKER (Thread terpisah)
    # ============================================================
    def _mqtt_worker(self):
        try:
            from paho.mqtt.enums import CallbackAPIVersion
            self.mqtt_client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION1)
        except ImportError:
            # Fallback jika memakai paho-mqtt versi lama
            self.mqtt_client = mqtt.Client()

        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_forever()
        except Exception as e:
            self.result_queue.put({'type': 'error', 'msg': str(e)})

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(MQTT_TOPIC)
            self.mqtt_connected = True
            self.result_queue.put({'type': 'connected'})
        else:
            self.result_queue.put({
                'type': 'error',
                'msg': f'Gagal konek MQTT (rc={rc}). Cek broker Mosquitto.'
            })

    def _on_mqtt_disconnect(self, client, userdata, rc):
        self.mqtt_connected = False
        self.result_queue.put({'type': 'disconnected'})

    def _on_mqtt_message(self, client, userdata, msg):
        # 1. X-Ray Detektor: Cetak setiap kali ada ketukan di pintu MQTT
        

        if not self.is_running:
            return
        
        try:
            # 2. X-Ray Isi Data: Cetak 100 karakter pertama dari data ESP32
            mentahan = msg.payload.decode('utf-8')
            print(f"📦 [ISI DATA] {mentahan[:100]}...") 

            data = json.loads(mentahan)

            # Ekstrak 11 fitur
            f = data.get('flex', [0, 0, 0, 0, 0])
            a = data.get('accel', [0.0, 0.0, 0.0])
            g = data.get('gyro', [0.0, 0.0, 0.0])

            sample = f + a + g  # list 11 elemen
            self.sensor_buffer.append(sample)
            self.frame_count += 1

            # Kirim data sensor ke UI queue untuk update display
            self.result_queue.put({
                'type'  : 'sensor',
                'flex'  : f,
                'accel' : a,
                'gyro'  : g,
                'buffer': len(self.sensor_buffer)
            })

            # Inferensi saat buffer penuh
            if len(self.sensor_buffer) == TIMESTEPS:
                self._run_inference()

        except Exception as e:
            # 3. X-Ray Error: Jangan diam kalau ada yang salah!
            print(f"❌ [ERROR FATAL BACA DATA] {e}")

    # ============================================================
    # INFERENSI CNN-LSTM
    # ============================================================
    def _run_inference(self):
        try:
            window = np.array(self.sensor_buffer,
                              dtype=np.float32)        # (100, 11)

            # --- START UL-02: Waktu Preprocessing ---
            start_prep = time.perf_counter()
            
            # Z-Score normalization
            window_scaled = self.scaler.transform(window) # (100, 11)

            # Reshape untuk model
            window_input = window_scaled.reshape(1, TIMESTEPS, N_FEATURES)
            
            waktu_prep_ms = (time.perf_counter() - start_prep) * 1000
            print(f"[UL-02] Waktu Preprocessing : {waktu_prep_ms:.2f} ms")
            # --- END UL-02 ---

            # --- START UL-03: Waktu Inferensi AI ---
            t0    = time.perf_counter()
            probs = self.model.predict(window_input, verbose=0)[0]
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"[UL-03] Waktu Inferensi AI  : {latency_ms:.2f} ms")
            # --- END UL-03 ---

            pred_idx  = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])
            pred_label = self.encoder.classes_[pred_idx]

            # Kirim hasil ke UI queue
            self.result_queue.put({
                'type'     : 'prediction',
                'label'    : pred_label,
                'conf'     : pred_conf,
                'probs'    : probs,
                'latency'  : latency_ms
            })

            # Kosongkan buffer setelah prediksi
            self.sensor_buffer.clear()

        except Exception as e:
            print(f"Inferensi error: {e}")

    # ============================================================
    # TEXT-TO-SPEECH (Thread terpisah)
    # ============================================================
    def _speak(self, text):

        kata_untuk_suara = text.lower()

        if kata_untuk_suara == "hallo":
            kata_untuk_suara = "halo"
            
        def _tts_worker():
            try:
                # Masukkan variabel kata_untuk_suara yang sudah diperbaiki ke gTTS
                tts = gTTS(text=kata_untuk_suara, lang='id', slow=False)
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix='.mp3') as fp:
                    tmp_path = fp.name
                tts.save(tmp_path)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                os.unlink(tmp_path)
            except Exception as e:
                print(f"TTS error: {e}")

        tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        tts_thread.start()

    # ============================================================
    # UPDATE UI LOOP (polling queue setiap 50ms)
    # ============================================================
    def _update_ui_loop(self):
        try:
            while not self.result_queue.empty():
                item = self.result_queue.get_nowait()

                # ── Koneksi MQTT ───────────────────────────
                if item['type'] == 'connected':
                    self.lbl_mqtt_status.configure(
                        text="⬤  Terhubung",
                        text_color="#A5D6A7"
                    )

                elif item['type'] == 'disconnected':
                    self.lbl_mqtt_status.configure(
                        text="⬤  Terputus",
                        text_color="#EF9A9A"
                    )

                elif item['type'] == 'error':
                    messagebox.showerror("Koneksi Error", item['msg'])
                    self._stop_system()

                # ── Update Sensor Display ──────────────────
                elif item['type'] == 'sensor':
                    flex = item['flex']
                    a    = item['accel']
                    g    = item['gyro']
                    buf  = item['buffer']

                    # Update flex progress bars (Data ESP32: 0 - 100)
                    for i, bar in enumerate(self.flex_bars):
                        # Bagi 100 supaya jadi 0.0 sampai 1.0 untuk CustomTkinter
                        mapped_val = flex[i] / 100.0
                        
                        # Clamping: Jaga-jaga kalau sensor ESP32 ngirim nilai minus atau > 100
                        mapped_val = max(0.0, min(1.0, mapped_val))
                        
                        bar.set(mapped_val)

                    # Update IMU labels
                    imu_vals = {
                        'Acc X': a[0], 'Acc Y': a[1], 'Acc Z': a[2],
                        'Gyr X': g[0], 'Gyr Y': g[1], 'Gyr Z': g[2]
                    }
                    for name, val in imu_vals.items():
                        if name in self.imu_labels:
                            self.imu_labels[name].configure(
                                text=f"{val:+.3f}")

                    # Update buffer progress
                    ratio = buf / TIMESTEPS
                    self.bar_buffer.set(ratio)
                    self.lbl_buffer.configure(
                        text=f"{buf}/{TIMESTEPS} frame")

                # ── Update Hasil Prediksi ──────────────────
                elif item['type'] == 'prediction':
                    # --- START UL-04: Waktu Render UI & Suara ---
                    start_ui = time.perf_counter()

                    label   = item['label']
                    conf    = item['conf']
                    probs   = item['probs']
                    latency = item['latency']

                    # Update chart probabilitas
                    self._update_chart(probs)

                    # Hanya tampilkan jika confidence cukup
                    if conf >= CONFIDENCE_THRESHOLD:
                        self.lbl_result.configure(
                            text=label,
                            text_color=THEME['primary']
                        )
                        self.lbl_confidence.configure(
                            text=f"Keyakinan: {conf*100:.1f}%  "
                                 f"|  Latensi: {latency:.1f} ms"
                        )
                        self.bar_confidence.set(conf)

                        # Tambah ke history
                        self.history_list.append({
                            'label': label,
                            'conf' : conf,
                            'time' : time.strftime('%H:%M:%S')
                        })
                        self._render_history()

                        # TTS jika isyarat baru
                        if label != self.last_prediction:
                            self.last_prediction = label
                            self._speak(label)
                    else:
                        self.lbl_result.configure(
                            text="...",
                            text_color=THEME['text_light']
                        )
                        self.lbl_confidence.configure(
                            text=f"Confidence rendah: {conf*100:.1f}% "
                                 f"(threshold {CONFIDENCE_THRESHOLD*100:.0f}%)"
                        )
                        self.bar_confidence.set(conf)

                    waktu_ui_ms = (time.perf_counter() - start_ui) * 1000
                    print(f"[UL-04] Waktu Render UI     : {waktu_ui_ms:.2f} ms")
                    print("-" * 45) # Garis pembatas di terminal agar rapi
                    # --- END UL-04 ---

        except Exception as e:
            print(f"UI update error: {e}")

        # Jadwalkan ulang setiap 50ms
        if not self._check_destroyed():
            self.after(50, self._update_ui_loop)

    def _check_destroyed(self):
        try:
            self.winfo_exists()
            return False
        except:
            return True

    # ============================================================
    # CLOSE
    # ============================================================
    def _on_close(self):
        self.is_running = False
        try:
            self.mqtt_client.disconnect()
        except:
            pass
        pygame.mixer.quit()
        self.destroy()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Cek file yang dibutuhkan
    required = [MODEL_PATH, SCALER_PATH, ENCODER_PATH]
    missing  = [f for f in required if not os.path.exists(f)]

    if missing:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "File Tidak Ditemukan",
            f"File berikut tidak ditemukan:\n\n" +
            "\n".join(f"• {f}" for f in missing) +
            "\n\nPastikan semua file ada dalam 1 folder yang sama."
        )
        root.destroy()
    else:
        app = BISINDOApp()
        app.mainloop()

"""
==============================================================================
 APLIKASI METODE SECANT - Pencari Akar Persamaan Non-Linear
==============================================================================
 Instalasi Dependensi:
   pip install customtkinter matplotlib numpy sympy

 Cara Menjalankan:
   python secant_method_app.py
==============================================================================
"""

# ─── STANDARD & THIRD-PARTY IMPORTS ──────────────────────────────────────────
import customtkinter as ctk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")                        # Backend wajib untuk embed di Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sympy as sp
import math

# ─── KONFIGURASI TEMA GLOBAL ──────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS 1 ─ LOGIKA ALGORITMA SECANT
#  Bertanggung jawab HANYA untuk perhitungan numerik.
#  Tidak ada ketergantungan pada elemen GUI di sini.
# ═══════════════════════════════════════════════════════════════════════════════
class SecantSolver:
    """
    Kelas yang mengenkapsulasi logika Metode Secant.

    Metode Secant adalah metode pencari akar yang menggunakan formula:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    """

    def __init__(self, func_str: str):
        """
        Inisialisasi solver dengan string fungsi dari pengguna.
        Menggunakan sympy untuk parsing ekspresi matematika yang aman.
        """
        x = sp.Symbol('x')
        # Parse string fungsi menggunakan sympy agar aman dan fleksibel
        self.sym_func = sp.sympify(func_str)
        # Konversi ke fungsi numerik yang cepat dengan numpy sebagai backend
        self.f = sp.lambdify(x, self.sym_func, modules=['numpy', 'math'])

    def solve(self, x0: float, x1: float, tol: float, max_iter: int) -> dict:
        """
        Jalankan iterasi Metode Secant.

        Returns:
            dict: Berisi 'iterations' (list detail per langkah),
                  'root' (akar yang ditemukan), 'converged' (bool).
        """
        iterations = []
        converged = False

        for n in range(1, max_iter + 1):
            fx0 = self.f(x0)
            fx1 = self.f(x1)

            # ── GUARD: Cegah pembagian dengan nol ──────────────────────────
            if (fx1 - fx0) == 0:
                raise ZeroDivisionError(
                    f"f(x0) = f(x1) = {fx1:.6f} pada iterasi ke-{n}.\n"
                    "Metode Secant berhenti karena pembagi menjadi nol.\n"
                    "Coba ubah nilai tebakan awal x₀ dan x₁."
                )

            # ── RUMUS UTAMA METODE SECANT ───────────────────────────────────
            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            fx2 = self.f(x2)

            # Hitung error relatif (|x2 - x1| / |x2|) dalam persen
            error = abs(x2 - x1) / (abs(x2) + 1e-15) * 100 if x2 != 0 else abs(x2 - x1)

            # Simpan data iterasi untuk ditampilkan di log dan grafik
            iterations.append({
                'n':    n,
                'x2':   x2,
                'fx2':  fx2,
                'error': error
            })

            # ── CEK KONVERGENSI ─────────────────────────────────────────────
            if error < tol:
                converged = True
                break

            # Geser titik untuk iterasi berikutnya
            x0, x1 = x1, x2

        return {
            'iterations': iterations,
            'root': x2 if iterations else None,
            'converged': converged
        }

    def get_curve_data(self, x_center: float, span: float = 5.0) -> tuple:
        """
        Hasilkan titik-titik (x, y) untuk memplot kurva f(x).
        Digunakan sebagai latar belakang plot pada grafik konvergensi.
        """
        x_vals = np.linspace(x_center - span, x_center + span, 500)
        try:
            y_vals = self.f(x_vals)
            # Batasi nilai ekstrem agar grafik tetap rapi
            y_vals = np.clip(y_vals, -50, 50)
        except Exception:
            y_vals = np.zeros_like(x_vals)
        return x_vals, y_vals


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS 2 ─ ANTARMUKA GRAFIS (GUI)
#  Bertanggung jawab HANYA untuk tata letak dan interaksi pengguna.
#  Mendelegasikan kalkulasi ke SecantSolver.
# ═══════════════════════════════════════════════════════════════════════════════
class SecantApp(ctk.CTk):
    """
    Kelas utama aplikasi GUI menggunakan CustomTkinter.
    Mewarisi dari ctk.CTk (jendela utama).
    """

    # ── Konstanta Warna & Gaya ─────────────────────────────────────────────
    COLOR_BG      = "#0f1117"
    COLOR_PANEL   = "#1a1d27"
    COLOR_CARD    = "#21253a"
    COLOR_ACCENT  = "#4f8ef7"
    COLOR_SUCCESS = "#2ecc71"
    COLOR_WARNING = "#f39c12"
    COLOR_ERROR   = "#e74c3c"
    COLOR_TEXT    = "#e8eaf0"
    COLOR_MUTED   = "#7b82a0"

    FONT_TITLE    = ("Consolas", 18, "bold")
    FONT_LABEL    = ("Consolas", 12)
    FONT_ENTRY    = ("Consolas", 13)
    FONT_MONO     = ("Courier New", 11)

    def __init__(self):
        super().__init__()
        self._configure_window()
        self._build_layout()
        self._build_input_panel()
        self._build_log_panel()
        self._build_graph_panel()
        self._setup_matplotlib()

    # ────────────────────────────────────────────────────────────────────────
    #  KONFIGURASI JENDELA
    # ────────────────────────────────────────────────────────────────────────
    def _configure_window(self):
        self.title("⚙  Metode Secant  ─  Numerical Root Finder")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.configure(fg_color=self.COLOR_BG)

    # ────────────────────────────────────────────────────────────────────────
    #  LAYOUT UTAMA (3 kolom: Input | Log | Graph)
    # ────────────────────────────────────────────────────────────────────────
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=290)  # Panel input
        self.grid_columnconfigure(1, weight=0, minsize=320)  # Panel log
        self.grid_columnconfigure(2, weight=1)               # Panel grafik
        self.grid_rowconfigure(0, weight=0)                  # Header
        self.grid_rowconfigure(1, weight=1)                  # Konten utama

        # ── Header Bar ──────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=self.COLOR_PANEL,
                              corner_radius=0, height=56)
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        header.grid_propagate(False)

        ctk.CTkLabel(
            header,
            text="  ∫  SECANT METHOD  ─  Non-Linear Equation Solver",
            font=("Consolas", 15, "bold"),
            text_color=self.COLOR_ACCENT
        ).pack(side="left", padx=20, pady=14)

        ctk.CTkLabel(
            header,
            text="Numerical Analysis  v1.0",
            font=("Consolas", 11),
            text_color=self.COLOR_MUTED
        ).pack(side="right", padx=20)

    # ────────────────────────────────────────────────────────────────────────
    #  PANEL KIRI: INPUT FORM
    # ────────────────────────────────────────────────────────────────────────
    def _build_input_panel(self):
        panel = ctk.CTkFrame(self, fg_color=self.COLOR_PANEL, corner_radius=0)
        panel.grid(row=1, column=0, sticky="nsew", padx=(0, 1))
        panel.grid_rowconfigure(10, weight=1)

        # ── Judul Panel ─────────────────────────────────────────────────────
        ctk.CTkLabel(panel, text="[ INPUT PARAMETER ]",
                     font=("Consolas", 12, "bold"),
                     text_color=self.COLOR_ACCENT).pack(
                         anchor="w", padx=20, pady=(20, 4))

        # Garis pemisah
        ctk.CTkFrame(panel, height=1, fg_color=self.COLOR_CARD).pack(
            fill="x", padx=20, pady=(0, 16))

        # ── Definisi Input Fields ────────────────────────────────────────────
        fields = [
            ("f(x)",    "Fungsi f(x)",    "2*x**3 - x - exp(-x)"),
            ("x0",      "x₀  (tebakan awal 1)", "0"),
            ("x1",      "x₁  (tebakan awal 2)", "1"),
            ("tol",     "Toleransi  ε  (%)",    "0.0001"),
            ("maxiter", "Iterasi Maks  N",       "50"),
        ]

        self.entries = {}
        for key, label, default in fields:
            self._make_input_row(panel, key, label, default)

        # Spasi
        ctk.CTkFrame(panel, height=1, fg_color=self.COLOR_CARD).pack(
            fill="x", padx=20, pady=(20, 16))

        # ── Tombol Aksi ──────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(panel, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20)
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.btn_calc = ctk.CTkButton(
            btn_frame,
            text="▶  HITUNG",
            font=("Consolas", 12, "bold"),
            fg_color=self.COLOR_ACCENT,
            hover_color="#3a7bd5",
            corner_radius=6,
            height=40,
            command=self._on_calculate
        )
        self.btn_calc.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.btn_reset = ctk.CTkButton(
            btn_frame,
            text="↺  RESET",
            font=("Consolas", 12, "bold"),
            fg_color=self.COLOR_CARD,
            hover_color="#2d3251",
            text_color=self.COLOR_MUTED,
            corner_radius=6,
            height=40,
            command=self._on_reset
        )
        self.btn_reset.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        # ── Status Bar ───────────────────────────────────────────────────────
        self.lbl_status = ctk.CTkLabel(
            panel, text="Siap menghitung...",
            font=("Consolas", 10),
            text_color=self.COLOR_MUTED,
            wraplength=250
        )
        self.lbl_status.pack(anchor="w", padx=20, pady=(14, 0))

        # ── Result Card ──────────────────────────────────────────────────────
        result_frame = ctk.CTkFrame(panel, fg_color=self.COLOR_CARD,
                                    corner_radius=8)
        result_frame.pack(fill="x", padx=20, pady=(12, 20))

        ctk.CTkLabel(result_frame, text="HASIL AKAR",
                     font=("Consolas", 10), text_color=self.COLOR_MUTED).pack(
                         anchor="w", padx=12, pady=(10, 0))

        self.lbl_root = ctk.CTkLabel(
            result_frame, text="─",
            font=("Consolas", 22, "bold"),
            text_color=self.COLOR_SUCCESS
        )
        self.lbl_root.pack(anchor="w", padx=12)

        self.lbl_froot = ctk.CTkLabel(
            result_frame, text="f(x*) = ─",
            font=("Consolas", 11),
            text_color=self.COLOR_MUTED
        )
        self.lbl_froot.pack(anchor="w", padx=12, pady=(0, 10))

    def _make_input_row(self, parent, key, label_text, default):
        """Helper: buat satu baris label + entry field."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(row, text=label_text,
                     font=("Consolas", 11),
                     text_color=self.COLOR_MUTED,
                     anchor="w").pack(fill="x")

        entry = ctk.CTkEntry(
            row,
            font=("Consolas", 13),
            fg_color=self.COLOR_CARD,
            border_color="#2d3251",
            border_width=1,
            text_color=self.COLOR_TEXT,
            height=34,
            corner_radius=6
        )
        entry.insert(0, default)
        entry.pack(fill="x")
        self.entries[key] = entry

    # ────────────────────────────────────────────────────────────────────────
    #  PANEL TENGAH: LOG ITERASI
    # ────────────────────────────────────────────────────────────────────────
    def _build_log_panel(self):
        panel = ctk.CTkFrame(self, fg_color=self.COLOR_PANEL, corner_radius=0)
        panel.grid(row=1, column=1, sticky="nsew", padx=(1, 1))
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(panel, text="[ LOG ITERASI ]",
                     font=("Consolas", 12, "bold"),
                     text_color=self.COLOR_ACCENT).grid(
                         row=0, column=0, sticky="w", padx=20, pady=(20, 4))

        ctk.CTkFrame(panel, height=1, fg_color=self.COLOR_CARD).grid(
            row=0, column=0, sticky="ew", padx=20, pady=(40, 0))

        # Header kolom tabel log
        header_frame = ctk.CTkFrame(panel, fg_color=self.COLOR_CARD,
                                    corner_radius=4)
        header_frame.grid(row=0, column=0, sticky="ew",
                          padx=12, pady=(50, 0))
        for i, (col, w) in enumerate([
            ("n", 30), ("x₂", 90), ("f(x₂)", 90), ("Error%", 75)
        ]):
            ctk.CTkLabel(header_frame, text=col,
                         font=("Consolas", 10, "bold"),
                         text_color=self.COLOR_ACCENT,
                         width=w, anchor="center").grid(
                             row=0, column=i, padx=2, pady=6)

        # Area scrollable untuk log
        self.log_box = ctk.CTkTextbox(
            panel,
            font=("Courier New", 11),
            fg_color=self.COLOR_BG,
            text_color=self.COLOR_TEXT,
            border_color=self.COLOR_CARD,
            border_width=1,
            corner_radius=4,
            wrap="none",
            state="disabled"
        )
        self.log_box.grid(row=1, column=0, sticky="nsew",
                          padx=12, pady=(0, 12))

        # Konfigurasi tag warna untuk log
        self.log_box.configure(state="normal")
        self.log_box.tag_config("header",
                                foreground=self.COLOR_ACCENT)
        self.log_box.tag_config("converge",
                                foreground=self.COLOR_SUCCESS)
        self.log_box.tag_config("warning",
                                foreground=self.COLOR_WARNING)
        self.log_box.tag_config("error_tag",
                                foreground=self.COLOR_ERROR)
        self.log_box.configure(state="disabled")

    # ────────────────────────────────────────────────────────────────────────
    #  PANEL KANAN: GRAFIK MATPLOTLIB
    # ────────────────────────────────────────────────────────────────────────
    def _build_graph_panel(self):
        self.graph_panel = ctk.CTkFrame(
            self, fg_color=self.COLOR_PANEL, corner_radius=0)
        self.graph_panel.grid(row=1, column=2, sticky="nsew")
        self.graph_panel.grid_rowconfigure(1, weight=1)
        self.graph_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.graph_panel, text="[ VISUALISASI ]",
                     font=("Consolas", 12, "bold"),
                     text_color=self.COLOR_ACCENT).grid(
                         row=0, column=0, sticky="w", padx=20, pady=(20, 4))

    # ────────────────────────────────────────────────────────────────────────
    #  SETUP MATPLOTLIB ─ Embed Figure ke dalam CTkFrame
    # ────────────────────────────────────────────────────────────────────────
    def _setup_matplotlib(self):
        """
        Inisialisasi Figure matplotlib dan embed ke dalam GUI menggunakan
        FigureCanvasTkAgg. Figure dibuat dengan dua subplot (atas-bawah):
          - ax1 (atas): Konvergensi estimasi akar per iterasi
          - ax2 (bawah): Penurunan nilai error per iterasi
        """
        # Warna latar untuk konsistensi dengan tema dark
        BG = self.COLOR_BG
        PANEL = self.COLOR_PANEL

        # Buat Figure dengan rasio tinggi berbeda antar subplot
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1,
            figsize=(6, 7),
            gridspec_kw={'height_ratios': [1.4, 1]},
            facecolor=BG
        )
        self.fig.tight_layout(pad=3.0)
        self.fig.subplots_adjust(left=0.12, right=0.97,
                                  top=0.93, bottom=0.08, hspace=0.45)

        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor(self.COLOR_CARD)
            ax.tick_params(colors=self.COLOR_MUTED, labelsize=8)
            ax.xaxis.label.set_color(self.COLOR_MUTED)
            ax.yaxis.label.set_color(self.COLOR_MUTED)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.COLOR_CARD)
            ax.grid(True, linestyle='--', alpha=0.2,
                    color=self.COLOR_MUTED)

        self._reset_axes()

        # ── Embed Figure ke GUI ──────────────────────────────────────────────
        # FigureCanvasTkAgg menghubungkan objek Figure matplotlib
        # dengan widget Tkinter, sehingga grafik tampil di dalam jendela
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_panel)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(bg=BG, highlightthickness=0)
        canvas_widget.grid(row=1, column=0, sticky="nsew",
                           padx=12, pady=(4, 12))

        # Render awal (kosong)
        self.canvas.draw()

    def _reset_axes(self):
        """Kosongkan dan set ulang label default untuk kedua subplot."""
        self.ax1.cla()
        self.ax2.cla()

        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor(self.COLOR_CARD)
            ax.tick_params(colors=self.COLOR_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.COLOR_CARD)
            ax.grid(True, linestyle='--', alpha=0.2, color=self.COLOR_MUTED)

        # Placeholder teks di tengah plot
        self.ax1.text(0.5, 0.5, "Tekan  ▶ HITUNG  untuk melihat grafik",
                      ha='center', va='center',
                      transform=self.ax1.transAxes,
                      color=self.COLOR_MUTED, fontsize=10,
                      fontstyle='italic')
        self.ax2.text(0.5, 0.5, "Grafik Error akan muncul di sini",
                      ha='center', va='center',
                      transform=self.ax2.transAxes,
                      color=self.COLOR_MUTED, fontsize=10,
                      fontstyle='italic')

        self.ax1.set_title("Konvergensi Estimasi Akar  X_{i+1}",
                           color=self.COLOR_TEXT, fontsize=10, pad=8)
        self.ax2.set_title("Penurunan Error per Iterasi  (%)",
                           color=self.COLOR_TEXT, fontsize=10, pad=8)

    # ────────────────────────────────────────────────────────────────────────
    #  EVENT HANDLER: TOMBOL HITUNG
    # ────────────────────────────────────────────────────────────────────────
    def _on_calculate(self):
        """Validasi input, jalankan solver, lalu tampilkan hasil."""
        # ── 1. Ambil & Validasi Input ────────────────────────────────────────
        try:
            func_str = self.entries["f(x)"].get().strip()
            x0       = float(self.entries["x0"].get().strip())
            x1       = float(self.entries["x1"].get().strip())
            tol      = float(self.entries["tol"].get().strip())
            max_iter = int(self.entries["maxiter"].get().strip())

            if not func_str:
                raise ValueError("Fungsi f(x) tidak boleh kosong.")
            if tol <= 0:
                raise ValueError("Toleransi harus bernilai positif.")
            if max_iter < 1:
                raise ValueError("Iterasi maksimum minimal 1.")

        except ValueError as e:
            messagebox.showerror("Input Tidak Valid",
                                 f"❌ Kesalahan Input:\n\n{e}")
            return

        # ── 2. Inisialisasi Solver ───────────────────────────────────────────
        try:
            solver = SecantSolver(func_str)
        except Exception as e:
            messagebox.showerror("Fungsi Tidak Valid",
                                 f"❌ Gagal mem-parsing f(x):\n\n{e}\n\n"
                                 "Contoh format yang benar:\n"
                                 "  2*x**3 - x - exp(-x)\n"
                                 "  sin(x) - x/2\n"
                                 "  x**2 - 4")
            return

        # ── 3. Jalankan Iterasi Secant ───────────────────────────────────────
        try:
            result = solver.solve(x0, x1, tol, max_iter)
        except ZeroDivisionError as e:
            messagebox.showerror("Pembagian Nol", str(e))
            return
        except Exception as e:
            messagebox.showerror("Kesalahan Kalkulasi",
                                 f"❌ Error tidak terduga:\n\n{e}")
            return

        # ── 4. Tampilkan peringatan jika tidak konvergen ─────────────────────
        if not result['converged']:
            messagebox.showwarning(
                "Tidak Konvergen",
                f"⚠  Iterasi mencapai batas maksimum ({max_iter} iterasi)\n"
                f"namun solusi belum konvergen dengan toleransi ε = {tol}.\n\n"
                "Saran:\n"
                "  • Naikkan batas iterasi maksimum\n"
                "  • Perkecil toleransi\n"
                "  • Coba tebakan awal yang berbeda"
            )

        # ── 5. Update tampilan log, grafik, dan label hasil ──────────────────
        self._update_log(result, tol)
        self._update_graphs(result, solver)
        self._update_result_labels(result, solver)

        status_icon = "✅" if result['converged'] else "⚠"
        n_iter = len(result['iterations'])
        self.lbl_status.configure(
            text=f"{status_icon}  Selesai dalam {n_iter} iterasi",
            text_color=self.COLOR_SUCCESS if result['converged'] else self.COLOR_WARNING
        )

    # ────────────────────────────────────────────────────────────────────────
    #  EVENT HANDLER: TOMBOL RESET
    # ────────────────────────────────────────────────────────────────────────
    def _on_reset(self):
        """Bersihkan semua output tanpa mengubah nilai input."""
        self._clear_log()
        self._reset_axes()
        self.canvas.draw_idle()
        self.lbl_root.configure(text="─", text_color=self.COLOR_SUCCESS)
        self.lbl_froot.configure(text="f(x*) = ─",
                                  text_color=self.COLOR_MUTED)
        self.lbl_status.configure(text="Siap menghitung...",
                                   text_color=self.COLOR_MUTED)

    # ────────────────────────────────────────────────────────────────────────
    #  UPDATE LOG ITERASI
    # ────────────────────────────────────────────────────────────────────────
    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _update_log(self, result: dict, tol: float):
        """Tulis setiap baris iterasi ke dalam text box log."""
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")

        # Header tabel
        sep = "─" * 54 + "\n"
        header = f"{'n':>3}  {'x₂':>14}  {'f(x₂)':>13}  {'Error%':>10}\n"
        self.log_box.insert("end", sep)
        self.log_box.insert("end", header, "header")
        self.log_box.insert("end", sep)

        for row in result['iterations']:
            n     = row['n']
            x2    = row['x2']
            fx2   = row['fx2']
            err   = row['error']
            is_last = (n == len(result['iterations']))

            line = (f"{n:>3}  {x2:>14.8f}  {fx2:>13.6e}  "
                    f"{err:>10.6f}\n")

            tag = "converge" if (is_last and result['converged']) else None
            if tag:
                self.log_box.insert("end", line, tag)
            else:
                self.log_box.insert("end", line)

        self.log_box.insert("end", sep)

        # Pesan kesimpulan
        if result['converged']:
            msg = (f"✅ KONVERGEN  →  Akar x* ≈ "
                   f"{result['root']:.10f}\n")
            self.log_box.insert("end", msg, "converge")
        else:
            msg = f"⚠  Tidak konvergen dalam {len(result['iterations'])} iterasi.\n"
            self.log_box.insert("end", msg, "warning")

        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    # ────────────────────────────────────────────────────────────────────────
    #  UPDATE GRAFIK MATPLOTLIB
    # ────────────────────────────────────────────────────────────────────────
    def _update_graphs(self, result: dict, solver: SecantSolver):
        """
        Perbarui kedua subplot dengan data hasil iterasi.

        ax1 (atas): Plot nilai estimasi akar x_{i+1} per iterasi.
                    Menampilkan kecepatan konvergensi secara visual.
        ax2 (bawah): Plot nilai Error (%) per iterasi dalam skala log.
                     Menampilkan penurunan residual secara eksponensial.
        """
        # Ekstrak data dari list iterasi
        iters  = [r['n']    for r in result['iterations']]
        x_vals = [r['x2']   for r in result['iterations']]
        errors = [r['error'] for r in result['iterations']]

        # ── Reset subplot ────────────────────────────────────────────────────
        self._reset_axes()

        # ── SUBPLOT 1: Konvergensi Nilai Akar ───────────────────────────────
        self.ax1.plot(iters, x_vals,
                      color=self.COLOR_ACCENT,
                      linewidth=2, marker='o', markersize=5,
                      markerfacecolor='white',
                      markeredgecolor=self.COLOR_ACCENT,
                      label='Estimasi x*', zorder=3)

        # Garis horizontal akar final (referensi)
        if result['root'] is not None:
            self.ax1.axhline(y=result['root'],
                             color=self.COLOR_SUCCESS,
                             linestyle='--', linewidth=1.2,
                             alpha=0.7, label=f"Akar ≈ {result['root']:.6f}")

        # Anotasi titik konvergensi terakhir
        if x_vals:
            self.ax1.annotate(
                f"x* ≈ {x_vals[-1]:.6f}",
                xy=(iters[-1], x_vals[-1]),
                xytext=(-40, 12),
                textcoords='offset points',
                color=self.COLOR_SUCCESS,
                fontsize=8,
                arrowprops=dict(arrowstyle="->",
                                color=self.COLOR_SUCCESS,
                                lw=1)
            )

        self.ax1.set_title("Konvergensi Estimasi Akar  X_{i+1}",
                           color=self.COLOR_TEXT, fontsize=10, pad=8)
        self.ax1.set_xlabel("Iterasi ke-n", fontsize=9)
        self.ax1.set_ylabel("Nilai Estimasi Akar", fontsize=9)
        self.ax1.legend(fontsize=8, facecolor=self.COLOR_PANEL,
                        labelcolor=self.COLOR_TEXT, edgecolor='none')
        self.ax1.tick_params(colors=self.COLOR_MUTED, labelsize=8)

        # ── SUBPLOT 2: Penurunan Error ───────────────────────────────────────
        # Gunakan skala logaritmik agar penurunan eksponensial terlihat jelas
        safe_errors = [max(e, 1e-15) for e in errors]

        self.ax2.semilogy(iters, safe_errors,
                          color=self.COLOR_WARNING,
                          linewidth=2, marker='s', markersize=5,
                          markerfacecolor='white',
                          markeredgecolor=self.COLOR_WARNING,
                          label='Error (%)', zorder=3)

        # Garis toleransi sebagai referensi
        tol_val = float(self.entries["tol"].get())
        self.ax2.axhline(y=tol_val,
                         color=self.COLOR_ERROR,
                         linestyle=':', linewidth=1.5,
                         alpha=0.8, label=f"Toleransi ε = {tol_val}")

        # Isi area di bawah kurva error
        self.ax2.fill_between(iters, safe_errors,
                              alpha=0.12, color=self.COLOR_WARNING)

        self.ax2.set_title("Penurunan Error per Iterasi  (%)",
                           color=self.COLOR_TEXT, fontsize=10, pad=8)
        self.ax2.set_xlabel("Iterasi ke-n", fontsize=9)
        self.ax2.set_ylabel("Error (%) — skala log", fontsize=9)
        self.ax2.legend(fontsize=8, facecolor=self.COLOR_PANEL,
                        labelcolor=self.COLOR_TEXT, edgecolor='none')
        self.ax2.tick_params(colors=self.COLOR_MUTED, labelsize=8)

        # Styling spine & grid ulang setelah cla()
        for ax in [self.ax1, self.ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3251")
            ax.grid(True, linestyle='--', alpha=0.18,
                    color=self.COLOR_MUTED)

        # ── Render ulang kanvas ──────────────────────────────────────────────
        # draw_idle() lebih efisien dari draw() untuk update berulang
        self.fig.tight_layout(pad=2.5)
        self.fig.subplots_adjust(left=0.13, right=0.97,
                                  top=0.93, bottom=0.08, hspace=0.5)
        self.canvas.draw_idle()

    # ────────────────────────────────────────────────────────────────────────
    #  UPDATE LABEL HASIL
    # ────────────────────────────────────────────────────────────────────────
    def _update_result_labels(self, result: dict, solver: SecantSolver):
        """Perbarui kartu hasil di panel input."""
        if result['root'] is not None:
            root = result['root']
            try:
                froot = solver.f(root)
            except Exception:
                froot = float('nan')

            self.lbl_root.configure(
                text=f"x* = {root:.10f}",
                text_color=self.COLOR_SUCCESS if result['converged']
                           else self.COLOR_WARNING
            )
            self.lbl_froot.configure(
                text=f"f(x*) = {froot:.4e}",
                text_color=self.COLOR_MUTED
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SecantApp()
    app.mainloop()
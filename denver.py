# denver.py

import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from customtkinter import CTkImage
import secrets, random, base64, os, io, struct, hmac, hashlib, colorsys, json
from Crypto.Cipher import ChaCha20_Poly1305, AES
from Crypto.Protocol.KDF import scrypt
from cryptography.fernet import Fernet
from PIL import Image
import win32clipboard
from hkdf import Hkdf

# DCT Steganography Dependencies
import numpy as np
import cv2

# --- File & Crypto Constants ---
CONFIG_FILE = "config.json"
lic_ext = ".reclusekey"
default_lic = f"master_key{lic_ext}"
key_path = default_lic

# Salts and Nonces
salt_len = 16
chacha_nonce_len = 12
aes_nonce_len = 16
tag_len = 16
fernet_salt_len = 16
hkdf_salt_len = 16

# License File Structure
magic_begin = "-----BEGIN LICENSE-----"
magic_sentinel = "LICENSING-V15"
magic_end = "-----END LICENSE-----"
magic_footer = "# magic"
lic_ver = "15"

# Session cache
pass_cache = {"passphrase": None}

# --- custom context menu ---
class CustomContextMenu(ctk.CTkToplevel):
    def __init__(self, master, theme_colors):
        super().__init__(master)
        self.overrideredirect(True)
        self.lift()
        self.grab_set()

        # Dynamically set colors from the theme dictionary
        bg_color = theme_colors.get("dialog_fg", "#1A102B")
        self.hover_color = theme_colors.get("button_hover", "#432D6D")
        self.text_color = theme_colors.get("text_color", "#E0E0E0")
        border_color = theme_colors.get("main_frame_border_color", "#6A11CB")

        self.bind("<FocusOut>", lambda e: self.destroy())
        self.bind("<Escape>", lambda e: self.destroy())

        self.main_frame = ctk.CTkFrame(self, fg_color=bg_color,
                                       border_color=border_color,
                                       border_width=2, corner_radius=12)
        self.main_frame.pack(expand=True, fill="both")

    def add_command(self, label, command):
        # Use themed colors for the buttons
        btn = ctk.CTkButton(self.main_frame, text=label, fg_color="transparent",
                            hover_color=self.hover_color, text_color=self.text_color,
                            corner_radius=8, anchor="w",
                            command=lambda: self._do_command(command))
        btn.pack(fill="x", padx=5, pady=4)

    def _do_command(self, command):
        self.destroy()
        if command:
            command()

    def popup(self, x, y):
        self.geometry(f"+{x}+{y}")
        self.update_idletasks()

# --- themed dialogs ---
class CenteredToplevel(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lift()
        self.attributes("-topmost", True)
        self.after(10, self.grab_set)

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

class CustomDialog(CenteredToplevel):
    def __init__(self, title="Dialog", text="...", theme_colors=None, **kwargs):
        super().__init__(**kwargs)
        self.title(title)
        self.result = None
        
        dialog_fg = theme_colors.get("dialog_fg", theme_colors.get("fg_color", "#2D2D2D"))
        self.text_color = theme_colors.get("dialog_text", theme_colors.get("text_color", "#FFFFFF"))
        
        if dialog_fg == "transparent":
            dialog_fg = "#2D2D2D" if int(self.text_color[1:], 16) > 0x888888 else "#FCE4EC"
            
        self.configure(fg_color=dialog_fg)
        
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(expand=True, fill="both", padx=30, pady=20)
        
        self.label = ctk.CTkLabel(self.main_frame, text=text, wraplength=450, justify="center", font=ctk.CTkFont(size=14), text_color=self.text_color)
        self.label.pack(pady=(0, 15), expand=True, fill="x")

        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
    
    def wait(self):
        self.master.wait_window(self)
        return self.result

class CustomMessagebox(CustomDialog):
    def __init__(self, title="Info", text="...", theme_colors=None, buttons=["OK"], **kwargs):
        super().__init__(title=title, text=text, theme_colors=theme_colors, **kwargs)
        
        self.button_frame.pack(pady=(10,0))
        
        for btn_text in buttons:
            btn_lower = btn_text.lower()
            
            fg_color = theme_colors.get(f"{btn_lower}_btn_bg", theme_colors.get("popup_btn_bg", theme_colors.get("button_color")))
            hover_color = theme_colors.get(f"{btn_lower}_btn_hover", theme_colors.get("popup_btn_hover", theme_colors.get("button_hover")))
            border_color = theme_colors.get(f"{btn_lower}_btn_border", theme_colors.get("popup_btn_border"))
            border_width = theme_colors.get(f"{btn_lower}_btn_border_width", theme_colors.get("popup_btn_border_width", 0))
            text_color = theme_colors.get(f"{btn_lower}_btn_text", self.text_color)

            btn = ctk.CTkButton(self.button_frame, text=btn_text, command=lambda t=btn_text: self.on_button(t), 
                                fg_color=fg_color, hover_color=hover_color,
                                border_color=border_color, border_width=border_width, text_color=text_color)
            btn.pack(side="left", padx=5)
        
        self.resizable(False, False)
        self.center_window()

    def on_button(self, text):
        self.result = text.lower()
        self.destroy()

class CustomAskString(CustomDialog):
    def __init__(self, title="Input", text="Enter value:", theme_colors=None, show="", **kwargs):
        super().__init__(title=title, text=text, theme_colors=theme_colors, **kwargs)
        
        entry_fg = theme_colors.get("dialog_entry_fg", theme_colors.get("entry_fg", "#343638"))
        
        self.entry = ctk.CTkEntry(self.main_frame, width=350, show=show, 
                                  fg_color=entry_fg, text_color=self.text_color,
                                  border_color=theme_colors.get("popup_btn_hover"))
        self.entry.pack(pady=(0, 10))
        
        if show == "*":
            self.show_var = tk.BooleanVar(value=False)
            checkbox_fg = theme_colors.get("ok_btn_border", theme_colors.get("popup_btn_border", theme_colors.get("button_color")))
            checkbox = ctk.CTkCheckBox(self.main_frame, text="Show Passphrase",
                                       variable=self.show_var, command=self._toggle_show,
                                       text_color=self.text_color, fg_color=checkbox_fg)
            checkbox.pack(pady=(0, 15), anchor="w", padx=5)

        self.button_frame.pack(pady=(10,0))

        ok_fg = theme_colors.get("ok_btn_bg", theme_colors.get("popup_btn_bg", theme_colors.get("button_color")))
        ok_hover = theme_colors.get("ok_btn_hover", theme_colors.get("popup_btn_hover", theme_colors.get("button_hover")))
        ok_border = theme_colors.get("ok_btn_border", theme_colors.get("popup_btn_border"))
        ok_border_width = theme_colors.get("ok_btn_border_width", theme_colors.get("popup_btn_border_width", 0))
        
        self.ok_button = ctk.CTkButton(self.button_frame, text="OK", command=self.on_ok, 
                                       fg_color=ok_fg, hover_color=ok_hover,
                                       border_color=ok_border, border_width=ok_border_width)
        self.ok_button.pack(side="left", padx=5)
        
        cancel_fg = theme_colors.get("cancel_btn_bg", theme_colors.get("no_btn_bg", "#555555"))
        cancel_hover = theme_colors.get("cancel_btn_hover", theme_colors.get("no_btn_hover", "#444444"))
        cancel_border = theme_colors.get("cancel_btn_border", theme_colors.get("popup_btn_border"))
        cancel_border_width = theme_colors.get("cancel_btn_border_width", theme_colors.get("popup_btn_border_width", 0))

        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self.on_cancel, 
                                           fg_color=cancel_fg, hover_color=cancel_hover,
                                           border_color=cancel_border, border_width=cancel_border_width)
        self.cancel_button.pack(side="left", padx=5)

        self.entry.bind("<Return>", lambda event: self.on_ok())
        self.bind("<Escape>", lambda event: self.on_cancel())
        self.resizable(False, False)
        
        self.after(50, lambda: self.entry.focus_set())
        self.center_window()
        
    def _toggle_show(self):
        self.entry.configure(show="" if self.show_var.get() else "*")

    def on_ok(self):
        self.result = self.entry.get()
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

# --- cryptographic functions ---

def is_weak(pw: str) -> bool:
    return len(pw) < 12 or pw.isalpha()

def get_key_material(pw: str, salt: bytes) -> bytes:
    return scrypt(pw.encode("utf-8"), salt, 32, N=2**15, r=8, p=1)

def derive_keys(key_material: bytes, hkdf_salt: bytes) -> tuple:
    hkdf = Hkdf(salt=hkdf_salt, input_key_material=key_material)
    key1 = hkdf.expand(info=b"chacha-layer-1", length=32)
    key2 = hkdf.expand(info=b"aes-layer-2", length=32)
    key3 = hkdf.expand(info=b"chacha-layer-3", length=32)
    return key1, key2, key3

def chacha_encrypt(pt: bytes, key: bytes) -> bytes:
    nonce = secrets.token_bytes(chacha_nonce_len)
    cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(pt)
    return nonce + ct + tag

def chacha_decrypt(blob: bytes, key: bytes) -> bytes:
    if len(blob) < chacha_nonce_len + tag_len: raise ValueError("ChaCha20 ciphertext too short")
    nonce, tag, ct = blob[:chacha_nonce_len], blob[-tag_len:], blob[chacha_nonce_len:-tag_len]
    cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
    return cipher.decrypt_and_verify(ct, tag)

def aes_gcm_encrypt(pt: bytes, key: bytes) -> bytes:
    nonce = secrets.token_bytes(aes_nonce_len)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(pt)
    return nonce + ct + tag

def aes_gcm_decrypt(blob: bytes, key: bytes) -> bytes:
    if len(blob) < aes_nonce_len + tag_len: raise ValueError("AES-GCM ciphertext too short")
    nonce, tag, ct = blob[:aes_nonce_len], blob[-tag_len:], blob[aes_nonce_len:-tag_len]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ct, tag)

def encrypt_with_pass(pt: str, master_pw: str) -> bytes:
    if is_weak(master_pw):
        raise ValueError("Password is too weak")
    
    scrypt_salt = secrets.token_bytes(salt_len)
    hkdf_salt = secrets.token_bytes(hkdf_salt_len)
    
    key_material = get_key_material(master_pw, scrypt_salt)
    k1, k2, k3 = derive_keys(key_material, hkdf_salt)

    layer1_ct = chacha_encrypt(pt.encode("utf-8"), k1)
    layer2_ct = aes_gcm_encrypt(layer1_ct, k2)
    layer3_ct = chacha_encrypt(layer2_ct, k3)
    
    return scrypt_salt + hkdf_salt + layer3_ct

def decrypt_with_pass(payload: bytes, master_pw: str) -> str:
    if len(payload) < salt_len + hkdf_salt_len: raise ValueError("Payload too short")
    
    scrypt_salt, hkdf_salt, rest = payload[:salt_len], payload[salt_len:salt_len+hkdf_salt_len], payload[salt_len+hkdf_salt_len:]
    
    key_material = get_key_material(master_pw, scrypt_salt)
    k1, k2, k3 = derive_keys(key_material, hkdf_salt)

    layer2_ct = chacha_decrypt(rest, k3)
    layer1_ct = aes_gcm_decrypt(layer2_ct, k2)
    pt = chacha_decrypt(layer1_ct, k1)
    
    return pt.decode("utf-8", errors="strict")

# --- license file stuff ---
def fernet_key(phrase: str, salt: bytes) -> bytes:
    return base64.urlsafe_b64encode(scrypt(phrase.encode("utf-8"), salt, 32, N=2**15, r=8, p=1))

def do_hmac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()

def b64_e(b: bytes) -> str: return base64.b64encode(b).decode("utf-8")
def b64_d(s: str) -> bytes: return base64.b64decode(s.encode("utf-8"))
def b32_e(b: bytes) -> str: return base64.b32encode(b).decode("utf-8")
def b32_d(s: str) -> bytes: return base64.b32decode(s.upper().encode("utf-8"))

def build_license(ver: str, id: str, salt64: str, token64: str, mac64: str) -> str:
    return "\n".join([magic_begin, magic_sentinel, f"ver={ver}", f"lic_id={id}", f"kdf_salt={salt64}", f"enc_token={token64}", f"mac={mac64}", magic_footer, magic_end, ""])

def make_license(key_text: str, phrase: str) -> str:
    salt = secrets.token_bytes(fernet_salt_len)
    fkey = fernet_key(phrase, salt)
    token = Fernet(fkey).encrypt(key_text.encode("utf-8"))
    id = base64.urlsafe_b64encode(secrets.token_bytes(9)).decode("utf-8").rstrip("=")
    payload = f"ver={lic_ver}\nlic_id={id}\nkdf_salt={b64_e(salt)}\nenc_token={b64_e(token)}\n"
    mac = do_hmac(fkey, payload.encode("utf-8"))
    return build_license(lic_ver, id, b64_e(salt), b64_e(token), b64_e(mac))

def read_license(lic_text: str, phrase: str) -> str:
    lines = lic_text.strip().splitlines()
    if len(lines) < 8 or lines[0] != magic_begin or lines[-1] != magic_end: raise ValueError("bad license format")
    kv = {k.strip(): v.strip() for line in lines if "=" in line for k, v in [line.split("=", 1)]}
    if kv.get("ver") != lic_ver: raise ValueError("version mismatch")
    try:
        salt, token, mac = b64_d(kv["kdf_salt"]), b64_d(kv["enc_token"]), b64_d(kv["mac"])
    except KeyError: raise ValueError("license is missing fields")
    except Exception as e: raise ValueError(f"error decoding license fields: {e}")
    fkey = fernet_key(phrase, salt)
    payload = f"ver={kv['ver']}\nlic_id={kv['lic_id']}\nkdf_salt={kv['kdf_salt']}\nenc_token={kv['enc_token']}\n".encode("utf-8")
    if not hmac.compare_digest(do_hmac(fkey, payload), mac): raise ValueError("mac invalid, tampered file?")
    return Fernet(fkey).decrypt(token).decode("utf-8")

def save_key(master_key: str, phrase: str, path: str):
    with open(path, "w", encoding="utf-8") as f: f.write(make_license(master_key, phrase))

def load_key(phrase: str, path: str) -> str:
    with open(path, "r", encoding="utf-8") as f: data = f.read()
    return read_license(data, phrase)

# --- DCT core---

def hide_data_dct(img: Image.Image, data: bytes) -> Image.Image:
    header = struct.pack('>I', len(data))
    data_bits = "".join(f"{b:08b}" for b in (header + data))
    
    img_np = np.array(img.convert("RGB"))
    height, width, _ = img_np.shape
    
    if height < 16 or width < 16:
        raise ValueError("Image must be at least 16x16 pixels for DCT steganography.")
        
    img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(img_ycrcb)
    
    channels_to_use = [cr_channel, cb_channel]
    
    capacity = (height // 8) * (width // 8) * 2
    if len(data_bits) > capacity:
        raise ValueError(f"Data is too large for this image. Max size: {capacity // 8} bytes.")

    bit_index = 0
    for channel in channels_to_use:
        for i in range(0, height // 8):
            for j in range(0, width // 8):
                if bit_index >= len(data_bits): break
                block = channel[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                dct_block = cv2.dct(block)
                if data_bits[bit_index] == '1':
                    dct_block[4, 1] = max(5.0, abs(dct_block[4, 1]))
                else:
                    dct_block[4, 1] = min(-5.0, -abs(dct_block[4, 1]))
                idct_block = cv2.idct(dct_block)
                channel[i*8:(i+1)*8, j*8:(j+1)*8] = np.clip(idct_block, 0, 255).astype(np.uint8)
                bit_index += 1
            if bit_index >= len(data_bits): break
        if bit_index >= len(data_bits): break

    img_merged = cv2.merge((y_channel, cr_channel, cb_channel))
    img_final_np = cv2.cvtColor(img_merged, cv2.COLOR_YCrCb2RGB)
    
    return Image.fromarray(img_final_np)

def find_data_dct(img: Image.Image) -> bytes:
    img_np = np.array(img.convert("RGB"))
    height, width, _ = img_np.shape
    
    if height < 16 or width < 16:
        raise ValueError("Image is too small to contain data.")

    img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    _, cr_channel, cb_channel = cv2.split(img_ycrcb)
    channels_to_use = [cr_channel, cb_channel]

    extracted_bits = []
    
    for channel in channels_to_use:
        for i in range(0, height // 8):
            for j in range(0, width // 8):
                block = channel[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                dct_block = cv2.dct(block)
                if dct_block[4, 1] > 0:
                    extracted_bits.append('1')
                else:
                    extracted_bits.append('0')
    
    if len(extracted_bits) < 32:
        raise ValueError("No valid data header found in image.")
        
    header_bits = "".join(extracted_bits[:32])
    data_len = int(header_bits, 2)
    
    total_bits_needed = 32 + data_len * 8
    if len(extracted_bits) < total_bits_needed:
        raise ValueError("Image data appears to be corrupted or incomplete.")

    data_bits = "".join(extracted_bits[32:total_bits_needed])
    
    output_bytes = bytearray()
    for i in range(0, len(data_bits), 8):
        byte = data_bits[i:i+8]
        if len(byte) == 8:
            output_bytes.append(int(byte, 2))
            
    return bytes(output_bytes)

# --- clipboard helpers ---
def copy_img(img: Image.Image):
    output = io.BytesIO()
    img.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    finally:
        win32clipboard.CloseClipboard()

def paste_img() -> Image.Image:
    win32clipboard.OpenClipboard()
    try:
        data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
        if not data: raise ValueError("No image data on clipboard (CF_DIB format).")
        
        # prepend a BMP file header to the DIB data.
        info_header_size = struct.unpack_from("<I", data, 0)[0]
        file_header = b'BM' + struct.pack('<IHHI', 14 + len(data), 0, 0, 14 + info_header_size)
        return Image.open(io.BytesIO(file_header + data))
    except (TypeError, struct.error):
        # fallback for other image types on clipboard
        try:
            img = ImageGrab.grabclipboard()
            if isinstance(img, Image.Image):
                return img
            else:
                raise ValueError("Clipboard data is not a valid image.")
        except Exception:
            raise ValueError("Clipboard data is not a valid image.")
    finally:
        win32clipboard.CloseClipboard()

# --- gui ---
class App(ctk.CTk):
    THEMES = {
        "dark": {
            "bg_color": "#242424", "fg_color": "#2D2D2D", "text_color": "#FFFFFF", 
            "button_color": "#2874A6", "button_hover": "#1F618D", 
            "entry_fg": "#343638", "menu_bg": "#2D2D2D", "menu_fg": "#FFFFFF", "menu_active": "#3B8ED0",
            "main_frame_corner_radius": 20, "main_frame_border_width": 2, "main_frame_border_color": "#1F618D",
            "dialog_fg": "#2D2D2D"
        },
        "purple_gradient": {
            "bg_color": ["#20002c", "#0f0c29"], "fg_color": "transparent", "text_color": "#E0E0E0", 
            "button_color": "transparent", "button_hover": "#432D6D", "button_border": "#7D3C98", 
            "entry_fg": "#2C1E4A", "menu_bg": "#0f0c29", "menu_fg": "#E0E0E0", "menu_active": "#6a11cb",
            
            "dialog_fg": "#1A1130", "dialog_text": "#E0E0E0", "dialog_entry_fg": "#2C1E4A",
            "popup_btn_bg": "transparent", "popup_btn_hover": "#432D6D", 
            "popup_btn_border": "#4A235A", "popup_btn_border_width": 2,
            
            "ok_btn_border": "#9673E2", "yes_btn_border": "#9673E2", "retry_btn_border": "#9673E2",
            
            "no_btn_bg": "#C62828", "no_btn_hover": "#B71C1C", "no_btn_border_width": 0,
            "cancel_btn_bg": "transparent", "cancel_btn_hover": "#432D6D",
            "main_frame_corner_radius": 20, "main_frame_border_width": 2, "main_frame_border_color": "#7D3C98"
        }
    }
    
    @staticmethod
    def load_initial_config():
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError):
            pass
        return {}
    
    INITIAL_CONFIG = load_initial_config()

    def __init__(self):
        super().__init__()
        self.withdraw() 

        self.title("Recluse Encryptor v15")
        
        self.win_width, self.win_height = 950, 900
        x_pos = (self.winfo_screenwidth() / 2) - (self.win_width / 2)
        y_pos = (self.winfo_screenheight() / 2) - (self.win_height / 2)
        self.geometry(f'{self.win_width}x{self.win_height}+{int(x_pos)}+{int(y_pos)}')
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.background_label = None
        self._resize_job = None

        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)
        self.settings_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Settings", menu=self.settings_menu)
        
        self.theme_var = tk.StringVar(value=self.INITIAL_CONFIG.get("theme", "dark"))
        self.theme_submenu = tk.Menu(self.settings_menu, tearoff=0)
        self.settings_menu.add_cascade(label="Theme", menu=self.theme_submenu)
        self.theme_submenu.add_radiobutton(label="Dark Mode", variable=self.theme_var, value="dark", command=self.on_theme_change)
        self.theme_submenu.add_radiobutton(label="Dark Purple Gradient", variable=self.theme_var, value="purple_gradient", command=self.on_theme_change)
        
        self.rainbow_mode_var = tk.BooleanVar(value=False)
        self.settings_menu.add_checkbutton(label="Rainbow Mode", onvalue=True, offvalue=False, variable=self.rainbow_mode_var, command=self._toggle_rainbow_mode)
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(8, weight=1)
        self.main_frame.grid_rowconfigure(10, weight=1)

        self.l_master_key = ctk.CTkLabel(self.main_frame, text="Master Key", font=ctk.CTkFont(size=18, weight="bold"))
        self.l_master_key.grid(row=0, column=0, padx=20, pady=(10, 5))
        self.pw_box = ctk.CTkEntry(self.main_frame, width=500, show="*", placeholder_text="Enter your master key", font=("Segoe UI", 14))
        self.pw_box.grid(row=1, column=0, padx=20, pady=5)
        self.pw_box.bind("<Button-3>", self._show_context_menu)
        
        self.show_pw = tk.BooleanVar(value=False)
        self.show_pw_checkbox = ctk.CTkCheckBox(self.main_frame, text="Show Key", variable=self.show_pw, command=self.toggle_pw)
        self.show_pw_checkbox.grid(row=2, column=0, padx=20, pady=(0, 10))
        
        self.key_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.key_frame.grid(row=3, column=0, pady=10)
        self.b_rand_key = ctk.CTkButton(self.key_frame, text="🔑 Generate Random", command=self.rand_key)
        self.b_save_key = ctk.CTkButton(self.key_frame, text="💾 Save Key to File", command=self.save_key_file)
        self.b_load_key = ctk.CTkButton(self.key_frame, text="📂 Load Key from File", command=self.load_key_file)
        self.b_rand_key.pack(side="left", padx=10, pady=10)
        self.b_save_key.pack(side="left", padx=10, pady=10)
        self.b_load_key.pack(side="left", padx=10, pady=10)

        self.b_rekey = ctk.CTkButton(self.main_frame, text="🔒 New Key & Encrypt Old Key to Clipboard", command=self.rekey)
        self.b_rekey.grid(row=4, column=0, pady=5)

        self.img_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.img_frame.grid(row=5, column=0, pady=10)
        self.b_pick_file = ctk.CTkButton(self.img_frame, text="📂 Select Image File", command=self.pick_file)
        self.b_pick_folder = ctk.CTkButton(self.img_frame, text="📁 Select Image Folder", command=self.pick_folder)
        self.b_paste_img = ctk.CTkButton(self.img_frame, text="📋 Paste Image from Clipboard", command=self.get_pasted_img)
        self.b_pick_file.pack(side="left", padx=10, pady=10)
        self.b_pick_folder.pack(side="left", padx=10, pady=10)
        self.b_paste_img.pack(side="left", padx=10, pady=10)
        
        self.chip_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.chip_frame.grid(row=6, column=0, pady=5)
        
        self.dot = ctk.CTkFrame(self.chip_frame, width=14, height=14, corner_radius=7)
        
        self.dot.pack(side="left", padx=(0, 6))
        self.img_name = ctk.CTkLabel(self.chip_frame, text="No Image Selected", fg_color="transparent")
        self.img_name.pack(side="left", padx=(0, 10))
        self.b_clear_img = ctk.CTkButton(self.chip_frame, text="❌ Clear Image", command=self.clear_img, width=140)
        self.b_clear_img.pack(side="left")

        self.l_input = ctk.CTkLabel(self.main_frame, text="Input", font=ctk.CTkFont(size=16))
        self.l_input.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="sw")
        self.in_box = ctk.CTkTextbox(self.main_frame, font=("Consolas", 14))
        self.in_box.grid(row=8, column=0, padx=20, sticky="nsew")
        self.in_box.bind("<Button-3>", self._show_context_menu)
        
        self.l_output = ctk.CTkLabel(self.main_frame, text="Output", font=ctk.CTkFont(size=16))
        self.l_output.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="sw")
        self.out_box = ctk.CTkTextbox(self.main_frame, font=("Consolas", 14))
        self.out_box.grid(row=10, column=0, padx=20, sticky="nsew")
        self.out_box.bind("<Button-3>", self._show_context_menu)
        
        self.action_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.action_frame.grid(row=11, column=0, pady=10)
        self.b_encrypt = ctk.CTkButton(self.action_frame, text="🔐 Encrypt", command=self.do_encrypt, height=40)
        self.b_decrypt = ctk.CTkButton(self.action_frame, text="🔓 Paste & Decrypt", command=self.do_decrypt, height=40)
        self.b_encrypt.pack(side="left", padx=10, pady=10)
        self.b_decrypt.pack(side="left", padx=10, pady=10)
        
        self.status = ctk.CTkLabel(self.main_frame, text="", font=("Segoe UI", 12), fg_color="transparent")
        self.status.grid(row=12, column=0, padx=20, pady=(0, 10), sticky="w")
        
        self.all_buttons = [self.b_rand_key, self.b_save_key, self.b_load_key, self.b_rekey, self.b_pick_file, self.b_pick_folder, self.b_paste_img, self.b_encrypt, self.b_decrypt, self.b_clear_img]
        self.folder, self.file, self.pasted = None, None, None
        self.rainbow_hue = 0.0

        self.bind("<Configure>", self._debounce_resize)
        self._set_theme()
        ctk.set_appearance_mode("dark")

    def run_startup_sequence(self):
        """Simplified startup sequence without license checks."""
        self.after(100, self.startup_check)
        self.deiconify()

    def _show_context_menu(self, event):
        widget = event.widget
        theme_colors = self.THEMES[self.theme_var.get()]
        menu = CustomContextMenu(self, theme_colors)

        is_entry = isinstance(widget, ctk.CTkEntry)
        is_textbox = isinstance(widget, ctk.CTkTextbox)

        if is_entry and widget.select_present():
            menu.add_command("Cut", lambda: widget.event_generate("<<Cut>>"))
            menu.add_command("Copy", lambda: widget.event_generate("<<Copy>>"))
        elif is_textbox and widget.tag_ranges("sel"):
            menu.add_command("Copy", lambda: widget.event_generate("<<Copy>>"))

        try:
            if self.clipboard_get():
                menu.add_command("Paste", lambda: widget.event_generate("<<Paste>>"))
        except tk.TclError:
            pass 

        if is_entry and widget.get():
             menu.add_command("Select All", widget.select_range(0, 'end'))
        elif is_textbox:
            menu.add_command("Select All", lambda: widget.tag_add("sel", "1.0", "end"))
        
        menu.popup(event.x_root, event.y_root)

    def _set_theme(self):
        theme_name = self.theme_var.get()
        colors = self.THEMES[theme_name]
        is_gradient = theme_name == "purple_gradient"
        
        menu_config = {"background": colors["menu_bg"], "foreground": colors["menu_fg"], "activebackground": colors["menu_active"], "activeforeground": colors["menu_fg"]}
        self.menu_bar.configure(**menu_config); self.settings_menu.configure(**menu_config); self.theme_submenu.configure(**menu_config)
        
        if self.background_label: self.background_label.place_forget(); self.background_label = None
        self.configure(fg_color=colors["bg_color"] if not isinstance(colors["bg_color"], list) else colors["bg_color"][0])

        self.main_frame.configure(
            fg_color=colors["fg_color"],
            corner_radius=colors.get("main_frame_corner_radius", 0),
            border_width=colors.get("main_frame_border_width", 0),
            border_color=colors.get("main_frame_border_color")
        )

        for label in [self.l_master_key, self.img_name, self.l_input, self.l_output, self.status]: label.configure(text_color=colors["text_color"])
        for box in [self.pw_box, self.in_box, self.out_box]: box.configure(fg_color=colors["entry_fg"], text_color=colors["text_color"], border_color=colors.get("button_hover", colors["entry_fg"]))
        self.show_pw_checkbox.configure(text_color=colors["text_color"], bg_color="transparent")

        if is_gradient:
            self.after(50, self._apply_gradient_background)
        self.update_chip()

    def show_message(self, title, text, buttons=["OK"]):
        theme_colors = self.THEMES.get(self.theme_var.get(), self.THEMES["dark"])
        msg_box = CustomMessagebox(master=self, title=title, text=text, theme_colors=theme_colors, buttons=buttons)
        return msg_box.wait()

    def ask_string(self, title, text, show=""):
        theme_colors = self.THEMES.get(self.theme_var.get(), self.THEMES["dark"])
        ask_dialog = CustomAskString(master=self, title=title, text=text, theme_colors=theme_colors, show=show)
        return ask_dialog.wait()

    def show_status(self, msg: str, color="green"):
        theme_name = self.theme_var.get()
        if color == "green": text_color = "#39FF14" if theme_name in ["dark", "purple_gradient"] else "green"
        elif color == "orange": text_color = "#FFA500" if theme_name in ["dark", "purple_gradient"] else "#E65100"
        else: text_color = color
        
        self.status.configure(text=msg, text_color=text_color)
        # Clear the text after 4 seconds
        self.after(4000, lambda: self.status.configure(text=""))
    
    def on_theme_change(self):
        self._set_theme()
        try:
            with open(CONFIG_FILE, "w") as f: json.dump({"theme": self.theme_var.get()}, f)
        except IOError:
            print(f"Warning: Could not save theme to {CONFIG_FILE}")

    def _create_gradient_image(self, width, height, color1, color2):
        base, top, mask = Image.new("RGB", (width, height), color1), Image.new("RGB", (width, height), color2), Image.new("L", (width, height))
        mask_data = [int(255 * (y / height)) for y in range(height) for _ in range(width)]
        mask.putdata(mask_data)
        base.paste(top, (0, 0), mask)
        return base
    
    def _debounce_resize(self, event=None):
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._apply_gradient_background)

    def _apply_gradient_background(self):
        if self.theme_var.get() != "purple_gradient": return
        colors = self.THEMES["purple_gradient"]
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1 or height <= 1: return
        
        bg_img = self._create_gradient_image(width, height, colors["bg_color"][0], colors["bg_color"][1])
        bg_ctk_img = CTkImage(light_image=bg_img, size=(width, height))
        
        if self.background_label is None:
            self.background_label = ctk.CTkLabel(self, text="", image=bg_ctk_img)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.background_label.lower()
        else:
            self.background_label.configure(image=bg_ctk_img)
            self.background_label.image = bg_ctk_img

    def _toggle_rainbow_mode(self):
        if self.rainbow_mode_var.get(): self._animate_rainbow()
        else: self._set_theme()

    def _animate_rainbow(self):
        if not self.rainbow_mode_var.get(): return

        hover_color = self.THEMES[self.theme_var.get()].get("button_hover")
        r, g, b = colorsys.hsv_to_rgb(self.rainbow_hue, 1.0, 1.0)
        border_hex = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        
        buttons_to_animate = [b for b in self.all_buttons if b.cget("state") == "normal"]
        
        for btn in buttons_to_animate:
            btn.configure(fg_color="transparent", hover_color=hover_color, border_width=2, border_color=border_hex)
            
        self.rainbow_hue = (self.rainbow_hue + 0.005) % 1.0
        self.after(20, self._animate_rainbow)

    def toggle_pw(self): self.pw_box.configure(show="" if self.show_pw.get() else "*")

    def get_pass(self, title, is_new=False):
        if not is_new and pass_cache["passphrase"]: return pass_cache["passphrase"]
        pw = self.ask_string(title, "Enter passphrase:", show="*")
        if not pw: return None
        if is_new:
            pw2 = self.ask_string("Confirm Passphrase", "Re-enter passphrase:", show="*")
            if pw != pw2:
                self.show_message("Error", "Passphrases do not match", ["OK"])
                return None
            if is_weak(pw):
                if self.show_message("Warning", "Weak passphrase detected. Continue anyway?", ["Yes", "No"]) != "yes": return None
        pass_cache["passphrase"] = pw
        return pw
    
    def startup_check(self):
        if os.path.exists(key_path) and self.show_message("Load Key", f"Found {key_path}. Would you like to load it?", ["Yes", "No"]) == "yes":
            self.load_key_file(use_dialog=False)
    
    def rand_key(self):
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+"
        self.pw_box.delete(0, tk.END)
        self.pw_box.insert(0, "".join(secrets.choice(chars) for _ in range(48)))
        self.show_status("Random key generated successfully.")

    def save_key_file(self):
        key = self.pw_box.get().strip()
        if not key: return self.show_message("Warning", "The key is empty. Please enter or generate a key first.")
        phrase = self.get_pass("Set Encryption Passphrase", is_new=True)
        if not phrase: return self.show_status("Save operation cancelled.", color="orange")
        path = filedialog.asksaveasfilename(defaultextension=lic_ext, initialfile=default_lic, filetypes=[("RecluseKey Files", f"*{lic_ext}"), ("All files", "*.*")], title="Save Encrypted Key File")
        if path:
            try: save_key(key, phrase, path); self.show_status(f"Key successfully saved to {os.path.basename(path)}")
            except Exception as e: self.show_message("Error", f"Failed to save the key: {e}")
            
    def load_key_file(self, use_dialog=True):
        path = key_path
        if use_dialog or not os.path.exists(path):
            path = filedialog.askopenfilename(filetypes=[("RecluseKey Files", f"*{lic_ext}"), ("All files", "*.*")], title="Open Encrypted Key File")
        if not path:
            return self.show_status("Load operation cancelled.", color="orange")
        
        for attempt in range(3):
            phrase = self.get_pass("Enter Passphrase to Load Key")
            if not phrase:
                return self.show_status("Load operation cancelled.", color="orange")
            try:
                key = load_key(phrase, path)
                self.pw_box.delete(0, tk.END); self.pw_box.insert(0, key)
                self.show_status(f"Key successfully loaded from {os.path.basename(path)}")
                return
            except Exception:
                pass_cache["passphrase"] = None
                if attempt < 2:
                    if self.show_message("Error", "Failed to load key.\nThis usually means an incorrect passphrase was entered.", ["Retry", "Cancel"]) != "retry":
                        break
                else:
                    self.show_message("Error", "Failed to load key after multiple attempts.")
        
        self.show_status("Load operation failed.", color="red")

    def rekey(self):
        old_pw = self.pw_box.get().strip()
        if not old_pw: return self.show_message("Warning", "Please enter the old key before re-keying.")
        new_key = "".join(secrets.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+") for _ in range(48))
        pt = f"new password: {new_key}"
        try:
            token = b32_e(encrypt_with_pass(pt, old_pw))
            self.clipboard_clear(); self.clipboard_append(token)
            self.pw_box.delete(0, tk.END); self.pw_box.insert(0, new_key)
            self.show_status("New key generated and encrypted old key copied to clipboard.")
        except Exception as e: self.show_message("Error", str(e))

    def pick_file(self):
        path = filedialog.askopenfilename()
        if path: self.file, self.folder, self.pasted = path, None, None; self.update_chip()
    def pick_folder(self):
        path = filedialog.askdirectory()
        if path: self.folder, self.file, self.pasted = path, None, None; self.update_chip()
    def get_pasted_img(self):
        try:
            img = paste_img()
            if isinstance(img, Image.Image): self.pasted, self.file, self.folder = img.convert("RGB"), None, None; self.update_chip()
            else: self.show_message("Info", "No image data found on the clipboard.")
        except Exception as e: self.show_message("Error", f"Failed to paste image: {e}")
    def clear_img(self):
        self.file, self.folder, self.pasted = None, None, None
        self.update_chip()
        self.show_status("Image selection has been cleared.", color="orange")
    
    def update_chip(self):
        theme_name = self.theme_var.get()
        colors = self.THEMES[theme_name]
        is_loaded = any([self.file, self.folder, self.pasted])
        
        if self.file: filename = os.path.basename(self.file)
        elif self.folder: filename = f"Folder: {os.path.basename(self.folder)}"
        elif self.pasted: filename = "Image from Clipboard"
        else: filename = "No Image Selected"
        
        self.dot.configure(fg_color="#27c93f" if is_loaded else "gray")
        self.img_name.configure(text=filename)
        
        if self.rainbow_mode_var.get(): return
        
        for btn in self.all_buttons:
            if btn.cget('state') == 'normal':
                if theme_name == "purple_gradient":
                    btn.configure(fg_color=colors["button_color"], hover_color=colors["button_hover"], border_color=colors["button_border"], text_color=colors["text_color"], border_width=2)
                else:
                    btn.configure(fg_color=colors["button_color"], hover_color=colors["button_hover"], text_color='white', border_width=0)
        
        if is_loaded and self.b_clear_img.cget('state') == 'normal':
            if theme_name == "purple_gradient": self.b_clear_img.configure(border_color="#D32F2F", hover_color="#B71C1C", fg_color="transparent")
            else: self.b_clear_img.configure(fg_color="#D32F2F", hover_color="#B71C1C")

    def do_encrypt(self):
        pw, inp = self.pw_box.get().strip(), self.in_box.get("1.0", tk.END).strip()
        if not (pw and inp): return self.show_message("Warning", "A master key and an input message are both required.")
        try:
            data = encrypt_with_pass(inp, pw)
            use_stego = any([self.pasted, self.file, self.folder])
            if use_stego:
                if self.pasted: img, name = self.pasted, "clipboard_image"
                elif self.file: img, name = Image.open(self.file).convert("RGB"), os.path.basename(self.file)
                else:
                    imgs = [f for f in os.listdir(self.folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
                    if not imgs: raise ValueError("No suitable image files were found in the selected folder.")
                    name = random.choice(imgs)
                    img = Image.open(os.path.join(self.folder, name)).convert("RGB")
                
                new_img = hide_data_dct(img, data)
                copy_img(new_img)
                self.show_status(f"Encrypted message hidden in {name} and copied to clipboard.")
            else:
                token = b32_e(data)
                self.clipboard_clear(); self.clipboard_append(token)
                self.out_box.delete("1.0", tk.END); self.out_box.insert("1.0", token)
                self.show_status("Encrypted text copied to clipboard.")
        except Exception as e: self.show_message("Encryption Error", f"An error occurred: {e}")
            
    def do_decrypt(self):
        pw = self.pw_box.get().strip()
        if not pw: return self.show_message("Warning", "Please enter the master key to decrypt.")
        
        # try to decrypt from an image on the clipboard
        try:
            img = paste_img()
            pt = decrypt_with_pass(find_data_dct(img), pw)
            self.out_box.delete("1.0", tk.END); self.out_box.insert("1.0", pt)
            self.show_status("Successfully decrypted from clipboard image.")
            return
        except (ValueError, TypeError, struct.error) as e:
            # if there's no image or it fails, we don't show an error yet.
            # just proceed to check for text. We only show an error if it's
            # a real image processing error, not just a lack of image.
            if "No image data on clipboard" not in str(e) and "not a valid image" not in str(e):
                return self.show_message("Decryption Error", f"Failed to process clipboard image:\n\n{e}")

        # if image decryption fails or there is no image try text
        try: token = self.clipboard_get().strip()
        except tk.TclError: token = self.in_box.get("1.0", tk.END).strip()

        if not token: return self.show_message("Warning", "Could not find anything to decrypt.\n\nPaste an image or encrypted text.")
        try:
            pt = decrypt_with_pass(b32_d(token), pw)
            self.out_box.delete("1.0", tk.END); self.out_box.insert("1.0", pt)
            self.show_status("Successfully decrypted from text.")
        except Exception as e: self.show_message("Decryption Error", f"Could not find anything to decrypt. It might be corrupted, the wrong key, or not encrypted text.\n\nDetails: {e}")


def main():
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    
    instance = App()
    instance.after(10, instance.run_startup_sequence)
    instance.mainloop()

if __name__ == "__main__":

    main()


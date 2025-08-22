# Recluse Encryptor v15

Kruelty Encryptor is a desktop application for Windows that provides strong, multi-layered encryption for text. It features a modern graphical user interface and an advanced steganography capability, allowing users to hide encrypted data securely within images.

![App Screenshot](https://cdn.discordapp.com/attachments/1276974120663388345/1408578235360481473/HeqwiYv.png?ex=68aa4035&is=68a8eeb5&hm=3cbd1fabeb42bbab348f659fae2cdc77243e14a29d9e39b63b08e322d60269c6&)  <!-- **Action:** Replace this with a URL to a screenshot of your app -->

## Features

-   **Multi-Layer Encryption:** Combines ChaCha20-Poly1305 and AES-GCM ciphers for robust security.
-   **Strong Key Derivation:** Uses `scrypt` and `HKDF` to derive encryption keys from a user-provided passphrase, protecting against brute-force attacks.
-   **Steganography:** Encrypt and hide your text data directly within image files (PNG, JPG, BMP) using a DCT (Discrete Cosine Transform) method.
-   **Clipboard Integration:** Easily paste images from the clipboard to hide data in, or paste encrypted text/images to decrypt them. The encrypted result is automatically copied to your clipboard.
-   **Secure Key Storage:** Save and load your master keys to an encrypted `.kruelkey` file, protected by a separate passphrase.
-   **Customizable UI:** Switch between a modern dark theme and a purple gradient theme.

## Installation

This application is designed for Windows. You will need Python 3 installed.

**1. Clone the repository:**
```bash
git clone https://github.com/warcampsurvivor/Kruelty-Encryptor.git
cd your-repo-name
```

**2. Install Dependencies:**
Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## How to Use

**1. Run the application:**
```bash
python denver.py
```

**2. Master Key Management:**
-   **Generate:** Click `🔑 Generate Random` to create a strong, secure master key.
-   **Save:** Click `💾 Save Key to File` to encrypt and save the current master key. You will be prompted for a passphrase to protect this file.
-   **Load:** Click `📂 Load Key from File` to decrypt and load a previously saved `.kruelkey` file.

**3. Encrypting Text:**
1.  Enter your Master Key in the top field.
2.  Type or paste the text you want to encrypt into the "Input" box.
3.  Click `🔐 Encrypt`.
4.  The encrypted text (Base32 encoded) will appear in the "Output" box and will be automatically copied to your clipboard.

**4. Encrypting and Hiding in an Image (Steganography):**
1.  Enter your Master Key.
2.  Load an image using one of the three methods:
    -   `📂 Select Image File`
    -   `📁 Select Image Folder` (a random image will be chosen)
    -   `📋 Paste Image from Clipboard`
3.  Type the secret message into the "Input" box.
4.  Click `🔐 Encrypt`.
5.  A new version of the image containing the hidden encrypted data is copied to your clipboard. You can now paste it into an image editor or messaging app and save it.

**5. Decrypting Data:**
1.  Enter the correct Master Key used for encryption.
2.  **For Text:** Paste the encrypted text into the "Input" box or have it on your clipboard.
3.  **For Images:** Copy the image containing the hidden data to your clipboard.
4.  Click `🔓 Paste & Decrypt`.

5.  The application will automatically detect if the clipboard contains an image with data or encrypted text. The decrypted message will appear in the "Output" box.




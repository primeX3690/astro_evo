# 13_realtime_telemetry/encrypted_hil_connector.py
import base64
import json
from cryptography.fernet import Fernet

class EncryptedHILConnector:
    def __init__(self, secret_key: bytes = None):
        # Generates or loads key for Hardware-in-the-Loop encryption
        self.key = secret_key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def get_exportable_key(self) -> str:
        return self.key.decode('utf-8')

    def encrypt_telemetry(self, telemetry_data: dict) -> str:
        """Encrypts flight computer telemetry payload for ground station."""
        json_data = json.dumps(telemetry_data).encode('utf-8')
        encrypted_token = self.cipher.encrypt(json_data)
        return base64.b64encode(encrypted_token).decode('utf-8')

    def decrypt_telemetry(self, encrypted_payload: str) -> dict:
        """Decrypts payload received at the ground station end."""
        decoded_bytes = base64.b64decode(encrypted_payload.encode('utf-8'))
        decrypted_data = self.cipher.decrypt(decoded_bytes)
        return json.loads(decrypted_data.decode('utf-8'))
import hashlib
import secrets

def sign(message, private_key):
    signature = b""
    if len(message) * 64 <= len(private_key):
        for i, mc in enumerate(message):
            for j, part in enumerate(divmod(mc, 16)):
                signature_part = private_key[(i*2+j)*32:(i*2+j+1)*32]
                for _ in range(part):
                    signature_part = hashlib.sha256(signature_part).digest()
                signature = signature + signature_part
        return signature
    else:
        return None

def verify(message, signature, public_key):
    if len(message) * 64 == len(signature) and len(message) * 64 <= len(public_key):
        for i, mc in enumerate(message):
            for j, message_part in enumerate(divmod(mc, 16)):
                signature_part = signature[(i*2+j)*32:(i*2+j+1)*32]
                for _ in range(15 - message_part):
                    signature_part = hashlib.sha256(signature_part).digest()
                if (signature_part != public_key[(i*2+j)*32:(i*2+j+1)*32]):
                    return False
        return True
    else:
        return False

def generate_private_key(message_length):
    private_key = bytearray(secrets.randbits(8) for _ in range(64*message_length))
    return private_key

def calculate_public_key(private_key):
    public_key = b""
    for i in range(len(private_key) // 32):
        part_key = private_key[i*32:(i+1)*32]
        for _ in range(15):
            part_key = hashlib.sha256(part_key).digest()
        public_key = public_key + part_key
    return public_key

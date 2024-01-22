#Lamport

import hashlib
import os

def generate_key_pair():
    # 8 Paare an zufallszahlen - urandom(32) -> kryptographisch sicher und 256 Bits lang
    private_key = [[os.urandom(32), os.urandom(32)] for _ in range(8)]
    # public_key als liste von hashes der private_key teile
    public_key = [[hashlib.sha256(key_part).digest() for key_part in pair] for pair in private_key]
    return private_key, public_key

def sign(message, private_key):
    signature = []
    for i in range(8):
        bit = (message >> i) & 1
        signature.append(private_key[i][bit])
    return signature

def verify(message, signature, public_key):
    for i in range(8):
        bit = (message >> i) & 1
        if hashlib.sha256(signature[i]).digest() != public_key[i][bit]:
            return False
    return True

def run_tests():
    private_key, public_key = generate_key_pair()
    
    message = 0x5A
    signature = sign(message, private_key)

    # correct message
    if verify(message, signature, public_key):
        print("\033[92m" + "✓ Test 1 Passed: Signature verified successfully." + "\033[0m")
    else:
        print("\033[91m" + "✗ Test 1 Failed: Signature verification failed." + "\033[0m")

    # incorrect message
    wrong_message = 0x3C
    if not verify(wrong_message, signature, public_key):
        print("\033[92m" + "✓ Test 2 Passed: Incorrect message failed verification as expected." + "\033[0m")
    else:
        print("\033[91m" + "✗ Test 2 Failed: Incorrect message should not verify." + "\033[0m")

run_tests()



from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

GREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'

# ChatGPT information
curve_overview = {
    'brainpoolP256r1': 'Weierstrass curve, part of the Brainpool set of curves for ECC.',
    'brainpoolP384r1': 'Weierstrass curve, part of the Brainpool set of curves for ECC.',
    'brainpoolP512r1': 'Weierstrass curve, part of the Brainpool set of curves for ECC.',
    'secp192r1': 'Weierstrass curve, also known as P-192, used in various cryptographic applications.',
    'secp224r1': 'Weierstrass curve, also known as P-224, used in various cryptographic applications.',
    'secp256k1': 'Weierstrass curve, known for its use in Bitcoin.',
    'secp256r1': 'Weierstrass curve, also known as P-256 or prime256v1, widely used in SSL/TLS and other protocols.',
    'secp384r1': 'Weierstrass curve, also known as P-384, widely used for stronger security in SSL/TLS and other protocols.', # DIESE
    'secp521r1': 'Weierstrass curve, also known as P-521, offers even stronger security.',
    'sect163k1': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect163r2': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect233k1': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect233r1': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect283k1': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect283r1': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect409k1': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect409r1': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect571k1': 'Binary curve, part of the SECT set of curves for ECC.',
    'sect571r1': 'Binary curve, part of the SECT set of curves for ECC.'
}

curve_attributes = [getattr(ec, attr) for attr in dir(ec) if isinstance(getattr(ec, attr), type)]

elliptic_curves = [curve for curve in curve_attributes if issubclass(curve, ec.EllipticCurve)]

for curve in elliptic_curves:
    curve_name = curve.name
    description = curve_overview.get(curve_name, "No description available.")
    print(f"{curve_name}: {description}")

input("Diffie Hellman test with SECP384R1")
private_key1 = ec.generate_private_key(ec.SECP384R1())
private_key2 = ec.generate_private_key(ec.SECP384R1())

public_key1 = private_key1.public_key()
public_key2 = private_key2.public_key()

shared_secret1 = private_key1.exchange(ec.ECDH(), public_key2)
shared_secret2 = private_key2.exchange(ec.ECDH(), public_key1)

try:
    assert shared_secret1 == shared_secret2
    print(GREEN + 'Shared secrets match. Test passed!' + ENDC)
except AssertionError:
    print(RED + 'Shared secrets do not match. Test failed!' + ENDC)


input("Signature test with SECP384R1")
private_key = ec.generate_private_key(ec.SECP384R1())

public_key = private_key.public_key()

data = b"Sollem ipsum oder so"

signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))

try:
    public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
    print(GREEN + 'Signature is valid. Verification passed!' + ENDC)
except InvalidSignature:
    print(RED + 'Signature is invalid. Verification failed!' + ENDC)

from cryptography.hazmat.primitives.asymmetric import ec

# Define a dictionary with curve names and their descriptions
curve_overview = {
    'brainpoolP256r1': 'Weierstrass curve, part of the Brainpool set of curves for ECC.',
    'brainpoolP384r1': 'Weierstrass curve, part of the Brainpool set of curves for ECC.',
    'brainpoolP512r1': 'Weierstrass curve, part of the Brainpool set of curves for ECC.',
    'secp192r1': 'Weierstrass curve, also known as P-192, used in various cryptographic applications.',
    'secp224r1': 'Weierstrass curve, also known as P-224, used in various cryptographic applications.',
    'secp256k1': 'Weierstrass curve, known for its use in Bitcoin.',
    'secp256r1': 'Weierstrass curve, also known as P-256 or prime256v1, widely used in SSL/TLS and other protocols.',
    'secp384r1': 'Weierstrass curve, also known as P-384, widely used for stronger security in SSL/TLS and other protocols.',
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

# Getting all the attributes from the 'ec' module that represent elliptic curves
curve_attributes = [getattr(ec, attr) for attr in dir(ec) if isinstance(getattr(ec, attr), type)]

# Filtering out those which are elliptic curves
elliptic_curves = [curve for curve in curve_attributes if issubclass(curve, ec.EllipticCurve)]

# Printing the overview of each curve
for curve in elliptic_curves:
    curve_name = curve.name
    description = curve_overview.get(curve_name, "No description available.")
    print(f"{curve_name}: {description}")


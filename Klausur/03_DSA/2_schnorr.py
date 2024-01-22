import hashlib

# Schnorr-Parameter
p = 6277
g = 2004
omega = 523
alpha = 213

# a) pub key
beta = pow(g, alpha, p)

# b) Signatur für "Hello World"
k = 123  # "random" number
r = pow(g, k, p)
message = "Hello World"
c = int(hashlib.sha256((message + str(r)).encode()).hexdigest(), 16) % omega
s = (k + c * alpha) % omega

# c) Überprüfung der Signatur
r = pow(g, s, p) * pow(beta, -c, p) % p
h_prime = int(hashlib.sha256((message + str(r)).encode()).hexdigest(), 16) % omega

print("Öffentlicher Schlüssel (beta):", beta)
print("Signatur (c, s):", (c, s))
print("Ist die Signatur gültig:", h_prime == c)

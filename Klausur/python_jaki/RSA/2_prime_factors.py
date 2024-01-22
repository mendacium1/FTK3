import si

# Gegeben z.B.: 
# Z^*_9788741
# phi(n) = 9782412
#
# Berechne primfaktoren!
# Händisch
# euler_phi( p * q ) = p * q - (p + q - 1)
# Zwei Funktionen -> Gleichungssystem
# Wolfram alpha (oder händisch):
#  9782412=p * q - (p + q - 1); 9788741= p * q
#  p = 2687, q = 3643

n = 9788731

primes = si.prime_factors(9788741)

print(f"p,q = {primes}")

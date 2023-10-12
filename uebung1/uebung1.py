import si
# 8.) DecryptClassic and DecryptCRS
p = 1913
q = 1297
d = 1723265

def DecryptClassic(p, q, d, c):
    d_p = d % p
    d_q = d % q

    m_p = c**d_p % p
    m_q = c**d_q % q

    # Y = ? -> erweiterter Euklid
    _, _, y = si.extended_gcd(p, q)
    # print(y)

    h = (m_p- m_q) * y % p
    m = m_q + q*h % p*q

    return m




def DecryptCRS(p, q, d, c):
    d_p = d % p
    d_q = d % q # hier wird noch kein square-and-multiply benötigt.

    d_p_bin = bin(d_p)[2:]
    m_p_multipliers = [c % p]
    for i in range(1, len(d_p_bin)):
        m_p_multipliers.append(m_p_multipliers[i-1]**2 % p)
    d_p_multiplied = 1
    for i,j in enumerate(list(d_p_bin)[::-1]):
        if j == "1":
            d_p_multiplied *= m_p_multipliers[i]
    m_p = d_p_multiplied % p

    d_q_bin = bin(d_q)[2:]
    m_q_multipliers = [c % q]
    for i in range(1, len(d_q_bin)):
        m_q_multipliers.append(m_q_multipliers[i-1]**2 % q)
    d_q_multiplied = 1
    for i,j in enumerate(list(d_q_bin)[::-1]):
        if j == "1":
            d_q_multiplied *= m_q_multipliers[i]
    m_q = d_q_multiplied % q

    # Y = ? -> erweiterter Euklid
    _, _, y = si.extended_gcd(p, q)
    # print(y)

    h = (m_p- m_q) * y % p
    m = m_q + q*h % p*q

    return m




import time

# Assuming you have the functions and necessary variables defined somewhere
# Start timing the first function
start_time = time.time()
decrypted_m = DecryptClassic(p, q, d, 1983)
end_time = time.time()
print(decrypted_m)
print(f"Time taken for DecryptClassic: {end_time - start_time} seconds")

# Start timing the second function
start_time = time.time()
decrypted_m = DecryptCRS(p, q, d, 1983)
end_time = time.time()
print(decrypted_m)
print(f"Time taken for DecryptCRS: {end_time - start_time} seconds")

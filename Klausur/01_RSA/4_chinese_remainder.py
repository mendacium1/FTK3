import si

# Gegeben Gleichungssystem z.B.:
# x = 45 (mod 79)
# x = 49 (mod 89)
#
# wobei:
# z = z_1 (mod n_1)
# z = z_2 (mod n_2)
#
# dadurch:
# n = n_1 * n_2
# Euklid für:
# n_1 * x_1 + n_2 * x_2 = 1
# Lösung:
# z := z_1 * n_2 * x_2 + z_2 * n_1 * x_1 mod n

n_s = [2,3,5,7,11,13,17,19]
#n_s = [79, 89]
#z_s = [45, 49]
z_s = [1,1,1,1,1,1,1,1]
result = si.chinese_remainder(n_s, z_s)

print(f"Chinese_remainder:\n{result}")

# Note: Altklausur 1b
# Weitere Lösungen: {z + kn | k element von Z}

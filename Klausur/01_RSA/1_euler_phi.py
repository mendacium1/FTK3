import si

# Gegeben z.B.: Z^*_77
#
# Berechne phi(77)
# -> Alle Elemente teilerfremd zu n
# aka: {ggT(x,n) = 1}
n = 77

phi = si.euler_phi(77)

print(phi)

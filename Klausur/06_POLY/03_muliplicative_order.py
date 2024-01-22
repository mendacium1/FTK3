import galois

# Multiplikative Ordnung
GF = galois.GF(2**4, repr="poly")

def find_multiplicative_order(element, field):
    order = 1
    while field(order) != field(1):
        order += 1
    return order

print("Element : Multiplicative Order")
for element in GF.elements[1:]:
    order = find_multiplicative_order(element, GF)
    print(f"{element} : {order}")

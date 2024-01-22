import galois

GF = galois.GF(2**4, repr="poly")

# a)
print(f"Prime Number: {GF.characteristic}")
print(f"Defining Polynomial: {GF.irreducible_poly}")
# b)
print("Elements:")
[print(element) for element in GF.elements]

# c)
print("Table:")
elements = GF.elements
table = [[a * b for b in elements] for a in elements]

def format_element(element):
    return str(element).center(10)

def print_line():
    print("+" + ("-" * 10 + "+") * (len(elements) + 1))

print_line()
header = "|" + " " * 10 + "|" + "|".join(format_element(e) for e in elements) + "|"
print(header)
print_line()

for a in elements:
    row = "|" + format_element(a) + "|" + "|".join(format_element(a * b) for b in elements) + "|"
    print(row)
    print_line()




# input("Aufgabe 6")
# multiplikative Ordnung: a^order = 1

def find_multiplicative_order(element, field):
    order = 1
    while field(order) != field(1):
        order += 1
    return order

print("Element : Multiplicative Order")
for element in GF.elements[1:]:
    order = find_multiplicative_order(element, GF)
    print(f"{element} : {order}")

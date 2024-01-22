import galois

k = galois.GF(3**2, irreducible_poly="x^2+1", repr="poly")

print(f"Count of elements: {len(k.elements)}")
print("Elements:")
[print(element) for element in k.elements]

#------- TABLE -------
print("Table:")
elements = k.elements
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


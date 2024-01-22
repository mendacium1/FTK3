import si

curve = si.EC(3, 6, 11)
points = list(curve.points())


a = points[3]
multiplicator = 10

print(f"A: {a}\ntype(A): {type(a)}")

print(f"A * x = {a} * {multiplicator} = {a.mult(multiplicator)}")

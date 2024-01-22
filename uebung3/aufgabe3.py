base = 6
target = 37
n = 131

x = 0
while True:
    if pow(base, x, n) == target:
        break
    x += 1

print(x)



import math

def calc_solutions_with_maximal_decimal(decimal_places, moduli) -> int:
    maximal_value = int("9"*decimal_places)
    total_mod = 1
    for mod in moduli:
        total_mod *= mod

    return maximal_value//total_mod


def hasse_interval(q):
    lower_bound = q + 1 - 2 * math.sqrt(q)
    upper_bound = q + 1 + 2 * math.sqrt(q)

    return lower_bound, upper_bound

def find_curve_order(point_order, modulo):
    lower, upper = hasse_interval(modulo)

    orders = []
    for i in range(int(lower), int(upper) + 1):
        if i % point_order == 0:
            orders.append(i)
    return orders

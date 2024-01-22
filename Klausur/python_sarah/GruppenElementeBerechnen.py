import math
import pandas
import si


class Gruppe:
    def __init__(self, prime: int, operation: str, kehrwert: str, restklasse: int, star=True):
        self.prime = prime
        self.operation = operation
        self.kehrwert = kehrwert
        self.restklasse = restklasse
        self.star = star

    def set_Elements(self):
        elements = []
        if self.star:
            i = 1
            self.order = si.euler_phi(self.prime)
        else:
            i = 0
            self.order = self.prime
        for number in range(i, self.prime):
            if self.star and (math.gcd(self.prime, number) == 1):
                elements.append(number)
                continue
            elif not self.star:
                elements.append(number)

        self.elements = elements
        self.calculate_order_and_kehrwert()

    def calculate_order_and_kehrwert(self):
        data = []
        for element in self.elements:
            order_found = False
            order = math.inf
            row = []
            for i in range(1, len(self.elements)+1):
                if self.operation == '+':
                    result = pow(element*i, 1, self.prime)
                    if not(order_found) and result == self.restklasse:
                        order = i
                        order_found = True
                    row.append(result)
                    continue
                if self.operation == '*':
                    result = pow(element, i, self.prime)
                    if not(order_found) and result == self.restklasse:
                        order = i
                        order_found = True
                    row.append(result)
            row.append(order)
            if self.kehrwert == '-':
                row.append((-element) % self.prime)
            if self.kehrwert == '-1':
                bla = row[(order-2)]
                row.append(bla)
            data.append(row)

        self.data = data

    def printOut(self):
        print(
            f"Gruppe({self.prime},{self.operation},{self.kehrwert},{self.restklasse}) {self.order=}")
        headers = []
        for i in range(1, len(self.elements)+1):
            headers.append("g^"+str(i))
        headers.append("Ordnung")
        headers.append("ĝ")
        dataframe = pandas.DataFrame(
            self.data, [e for e in self.elements], headers)
        dataframe.to_json(str(self.prime)+".json")
        print(dataframe.to_string(index=False))

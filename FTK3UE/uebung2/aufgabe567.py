# -- aufgabe5 --
def print_table():
    # Define the size of the group and the generator.
    group_size = 10
    
    # Create a header for the table.
    header = ['g^{}'.format(i) for i in range(1, group_size + 1)]
    print("{:<4}".format(' '), end='|')
    for item in header:
        print("{:<4}".format(item), end='|')
    print()  # Newline for the start of the table.

    # Create the body of the table.
    for y in range(group_size):  # For each element in the group (y-axis).
        print("{:<4}".format(y), end='|')  # Print the y-value.
        
        for x in range(1, group_size + 1):  # For each power of g (x-axis).
            value = (y * x) % group_size  # Calculate y * g^x mod 10.
            print("{:<4}".format(value), end='|')  # Print the calculated value.
        
        print()  # Newline for the next row.

input("aufgabe5")
# Run the function to print the table.
print_table()

# -- aufgabe6 --
class MultiplicativeGroup:
    def __init__(self, modulus):
        self.modulus = modulus
        # Generate the elements of the group: numbers from 1 to modulus-1.
        self.elements = [x for x in range(1, modulus)]

    def op(self, a, b):
        """The group operation (in this case, multiplication modulo 'modulus')."""
        return (a * b) % self.modulus

    def inverse(self, a):
        """Find the multiplicative inverse of 'a' modulo 'modulus'."""
        # This implementation assumes 'a' is in fact invertible (i.e., coprime with the modulus).
        for i in range(1, self.modulus):
            if (a * i) % self.modulus == 1:
                return i
        raise ValueError(f"No inverse found for {a} mod {self.modulus}")

    def exp(self, g, n):
        """Simulate 'exponentiation' by repeated application of the group operation."""
        result = 1  # neutral element for multiplication
        for _ in range(n):
            result = self.op(result, g)
        return result

    def create_table(self):
        # Header for the table.
        print(' g |', ' '.join(f'g^{i}|' for i in range(1, len(self.elements) + 1)))

        # Separator line.
        print('-' * (4 * len(self.elements) + 3))

        # Each row corresponds to an element of the group.
        for g in self.elements:  # for each element in the multiplicative group
            # Calculate the 'powers' of g and format them as a row in the table.
            powers = ' | '.join(str(self.exp(g, n)).rjust(2) for n in range(1, len(self.elements) + 1))
            print(f'{str(g).rjust(2)} | {powers}')


input("aufgabe6")
group = MultiplicativeGroup(11)  # for Z_11* under multiplication
group.create_table()

# -- aufgabe7 --
input("aufgabe7")
group = MultiplicativeGroup(15)
group.create_table()


def print_table():
    # Define the size of the group and the generator.
    group_size = 30
    
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



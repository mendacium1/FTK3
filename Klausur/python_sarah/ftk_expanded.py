"""
Extended functions for use in FTK3. Mainly functionality not contained in si.py
Author: Ioan R.
Date: 03.02.2023
"""
import math
import random
import hashlib
import numpy
import sympy
import si
# https://github.com/Robert-Campbell-256/Number-Theory-Python
import finitefield as ff

def find_elements( group: int ) -> set:
    """
    Returns a set of elements.

    Params:
        group - group (= n) with elements result

    Returns:
        Set of all elements
    """
    # case prime: phi(n) = n - 1
    if si.isprime(group) is True:
        return set(range(1, group))
    # else case: all elements except for factors and multiples
    factors = sympy.ntheory.factorint(group)

    non_elems = set(elem for elem in range(1,group) for factor in factors.keys() if elem % factor == 0)

    return set(result for result in range(1,group) if result not in non_elems)

def korollar112( basis: int, exponent: int, group: int ) -> int:
    """
    Solves using Korollar 1.12 and shows the steps.

    Params:
        basis
        exponent
        group

    Returns:
        result - The final value calculated by the 4 steps
    """
    if si.extended_gcd(basis,group)[0] != 1:
        print("gcd of z and n must be 1.")
    euler_order = si.euler_phi(group)
    print(f"Step 1:\n\tphi({group}) = {euler_order}.")
    b = exponent % euler_order
    print(f"Step 2:\n\tExponent ({exponent}) modulo phi({group}) = {b}")
    x = basis % group
    print(f"Step 3:\n\tBasis ({basis}) modulo {group} = {x}")
    result = pow(x, b, group)
    print(f"Step 4:\n\tergebnis_3 ^ ergebnis_2 mod n => {x}^{b} mod {group} = {result}")

    return result

def rsa_decrypt_chinese( cipher: int, privkey_p: int, privkey_q: int, privkey_d: int, verbose: bool ) -> int:
    """
    Params:
        cipher - The encrypted message
        p -
        q -
        d -
        verbose - Option for a description of steps

    Returns:
        m - The decrypted message
    """
    d_p = privkey_d % si.euler_phi(privkey_p)
    d_q = privkey_d % si.euler_phi(privkey_q)

    gcd_pq = si.extended_gcd(privkey_p, privkey_q)
    x = gcd_pq[1]
    y = gcd_pq[2]
    m_p = pow(cipher, d_p, privkey_p)
    m_q = pow(cipher, d_q, privkey_q)
    h = (m_p - m_q) * y % 47
    m = m_q + privkey_q * h % (privkey_p * privkey_q)

    if verbose:
        print(f"Zuerst d mod p und mod q berechnen:\n\td_p = d mod phi(p) = {privkey_d} % {si.euler_phi(privkey_p)} = {d_p}\n\td_q = d mod phi(q) = {privkey_d} % {si.euler_phi(privkey_q)} = {d_q}\n")
        print(f"Dann GGT von p und q berechnen:\n\tggt(p, q) = {gcd_pq}. x = {x} und y = {y}. \nDa man für x einsetzen kann brauchen wir nur y.\n")
        print(f"Next brauchen wir:\n\tc^d_p mod p und c^d_q mod q: m_p = c^d_p mod p = {cipher} ^ {d_p} mod {privkey_p} = {m_p}\n\tm_q = c^d_q mod q = {cipher} ^ {d_q} mod {privkey_q} = {m_q}\n")
        print(f"Berechnung von:\n\th = (m_p - m_q) * y mod p = ({m_p} - {m_q}) * {y} mod {privkey_p} = {h}\n")
        print(f"Abschließend noch:\n\tm = m_q + q * h mod (p * q) = {m_q} + {privkey_q} * {h} mod ({privkey_p} * {privkey_q}) = {m}\n")
    print(f"Decrypted message: {m}")

    return m

def rsa_decrypt_slow( cipher, d, n ) -> int:
    """
    Slow decryption without square and multiply
    c ^ d mod n
    """
    return cipher**d % n

def rsa_decrypt_classic( cipher, d, n ) -> int:
    """
    Faster version with square and multiply.
    pow(cipher, d, n)
    """
    return pow( cipher, d, n )

def check_order( element: int, group: int ) -> None:
    """
    Checks if ord(element aka g) is ord(group aka p)

    Params:
        element - Element
        group - The group aka p
    """
    omega = si.euler_phi(group)
    factors = si.prime_factors(omega)

    step_one = pow(element, omega, group)
    print(f"Factors of omega:\n\t p({omega}) = {factors}")
    print(f"Check one: if g^omega mod group != 1 -> ord(g) != omega\n\t{element}^{omega} mod {group} = {step_one}")
    if step_one != 1:
        print(f"ord(g) != omega ({omega})")
        return
    print("\n")

    print("Check two: if g^(omega / factors) == 1 -> ord(g) != omega")
    for factor in factors:
        step_two = pow(element, omega//factor, group)
        print(f"g^(omega/{factor}) mod group\n\t{element}^({omega}/{factor}) mod {group} = {step_two}")
        if step_two == 1:
            print(f"{element} ^ ({omega} / {factor}) = 1\n\tord(g) != omega ({omega})")
            return

    print(f"\nBoth checks passed -> ord(g) = omega\n\tord({element}) = {omega}")


def baby_giant_step( element: int, basis: int, group: int ) -> int:
    """
    Uses the baby step giant step algo to solve the DL of element to the basis in group.

    Params:
        element - Element to solve DL for;
        basis - Basis of element;
        group - The group aka p;

    Returns:
        DL solution;
    """
    N = int(math.ceil(math.sqrt(group - 1)))
    r = {}

    # Baby step.
    for i in range(N):
        r[pow(element, i, group)] = i

    c = pow(element, N * (group - 2), group)

    for j in range(N):
        q = (basis * pow(c, j, group)) % group
        if q in r:

            return j * N + r[q]

def pohlig_hellman( basis: int, element: int, group: int ) -> None:
    """
    Calculates the DL of element to the basis in group

    Params:
        element - The element to calculate;
        basis - The basis of element;
        group - The group aka p;
    """
    omega = si.euler_phi(group)
    primes = si.prime_factors(omega)

    m = 1
    for idx in range(len(primes) - 1):
        m *= primes[idx]

    n = primes[len(primes) - 1]
    print(m, n) #ggt(m,n) = 1
    d1 = pow(element, omega//m, group)
    d2 = pow(element, omega//n, group)
    d3 = pow(basis, omega//m, group)
    d4 = pow(basis, omega//n, group)

    x1 = baby_giant_step(d1, d3, group)
    x2 = baby_giant_step(d2, d4, group)

    print(f"alpha = {si.chinese_remainder([m, n], [x1, x2])}")

def create_dsa_private_key( omega: int ) -> int:
    """
    Chooses a random number in range of omega for use as private key alpha.

    Params:
        omega - The order;

    Returns:
        private key - alpha;
    """
    return random.randint(1, omega)

def get_dsa_keys( private_key: int, element: int, group: int ) -> tuple:
    """
    Calculates the public key A for any private key alpha.

    Params:
        private_key - Private key (random number = alpha);
        element - Element g of order omega in group;
        group - The group aka p (Number next to mathematical group term);

    Returns:
        (private_key, public_key) - A keypair in tuple form;
    """
    public_key = pow(element, private_key, group)

    return (private_key, public_key)

def create_dsa_sig( private_key: int, element: int, msg_hash: int, group: int ) -> tuple:
    """
    Creates a DSA signature for a message msg with private key private_key.

    Params:
        private_key - Private key (random number = alpha);
        element - Element g of order omega in group;
        msg - A message to create the signature for -> h(m);
        group - The group aka p (Number next to mathematical group term);

    Returns:
        (r, s) - Signature;
    """
    omega = si.euler_phi(group)
    k_found = False

    # Find a k for ggt(k, omega) = 1
    while not k_found:
        k = random.randint(1,omega-1) % omega
        if si.extended_gcd(omega, k)[0] == 1:
            k_found = True
            continue

    # Calculate sig
    r = pow(element, k, group) % omega
    s = si.inverse_mod(k, omega) * (int(msg_hash) + private_key * r) % omega

    return (r, s) #sig


def check_dsa_sig( signature: tuple, element: int, group: int, msg_hash: int, public_key: int ) -> bool:
    """
    Verifies the DSA signature. (properly)

    Params:
        signature - A tuple (r, s);
        element - Element g of order omega in group;
        group - The group aka p (Number next to mathematical group term);
        msg - A message to check the signature for -> h(m);
        public_key - The public key A;

    Returns:
        True - If step 3 is passed;
        False - If step 3 isn't passed;
    """
    r, s = signature
    omega = si.euler_phi(group)

    # Step 1: Check size of r and s
    print("Step 1:\n\tCheck 1 <= r < omega and 1 <= s < omega")
    if (r >= 1 and r < omega) and (s >= 1 and s < omega):
        print("Passed Step 1")
    else:
        print("Failed Step 1")

    # Step 2: Calculate x and y
    print("\nStep 2:\n\tCalc x = s^(-1) * h(msg) mod omega und y = s^(-1) * r mod omega")
    try:
        x = si.inverse_mod(s, omega) * msg_hash % omega
        y = si.inverse_mod(s, omega) * r % omega
    except TypeError:
        print("Error: Inverse mod failed")
        return
    print(f"\tx = {x}\nand\n\ty = {y}\n")

    # Step 3: Verify
    print("Step 3:\n\tCheck r = (g^x * A^y mod group) mod omega")
    if r == (pow( (pow(element, x, group) * pow(public_key, y, group) % group), 1, omega )):
        print("Passed Step 3")
        return True
    else:
        print("Failed Step 3")
        return False

def get_msg_hash( msg ):
    """
    Creates a SHA256 hash of the parameter in integer form.

    Params:
        msg - The message to hash;

    Returns:
        A SHA256 hash;
    """
    return int(hashlib.sha256(msg.encode()).hexdigest(), base=16)

def broken_dsa_sig( private_key: int, element: int, msg_hash: int, group: int, k: int ) -> tuple:
    """
    Similar to the previous signature but this time k is hardcoded
    => recycling of k breaks security and allows for calculation of the private key.

    Params:
        private_key - Private key (random number = alpha);
        element - Element g of order omega in group;
        msg - A message to create the signature for -> h(m);
        group - The group aka p (Number next to mathematical group term);
        k - k for ggt(k, omega) = 1

    Returns:
        (r, s) - Signature;
    """
    omega = si.euler_phi(group)

    r = pow(element, k, group) % omega
    s = si.inverse_mod(k, omega) * (int(msg_hash) + private_key * r) % omega

    return (r, s) #sig

def check_dsa_privatekey( element: int, group: int, private_key: int, public_key: int ) -> None:
    """
    Quick check to ensure A and alpha create a keypair.

    Params:
        element - g of group with ord(g) = omega;
        private_key - The (calculated) private key to check
        public_key - The (known) public key to check against
    """
    if public_key == pow(element, private_key, group):
        print("A matches the calculated value for alpha")

def calc_dsa_privatekey( sig1: tuple, sig2: tuple, hash1: int, hash2: int, omega: int ) -> int:
    """
    Calculates the private key from two sigs using the same k (Don't recycle kids).

    Params:
        sig1 - Any signature with k;
        sig2 - Any signature with the same k as sig1;
        hash1 - Hash of the first sig's message;
        hash2 - Hash of the second sig's message;
        omega - Order of group;

    Returns:
        private_key - Private key alpha;
    """
    k = si.inverse_mod(sig1[1] - sig2[1], omega) * (hash1 - hash2) % omega
    private_key = si.inverse_mod(sig1[0], omega) * (k * sig1[1] - hash1) % omega
    print(f"Alpha: {private_key}")

    return private_key

def fake_dsa_sig( real_hash: int, fake_hash: int, omega: int, signature: tuple, group: int ) -> list[tuple]:
    """
    Creates a fake DSA signature for a new message (fake_hash) which appears to be
    valid if the first step of verification is skipped (size check of r and s)

    Params:
        real_hash - Hash of the real message;
        fake_hash - Hash of the faked message;
        omega - Order of group;
        signature - Signature of the real message | Format (r, s);
        group - aka p;

    Returns:
        List of tuples: [(fake_r, fake_s), (x, y), (fake_x, fake_y)];
    """
    r = signature[0]
    s = signature[1]


    # u = fake_h * h^(-1) mod omega
    u = pow( fake_hash * si.inverse_mod(real_hash, omega), 1, omega )

    # Calc fake signature
    fake_s = s * u % omega
    fake_r = si.chinese_remainder( [omega, group], [(r*u%omega), (r%group)] )
    print(f"Fake Signature:\n\tr = {fake_r}\n\ts = {fake_s}")

    # Calc x and y of both sigs
    x = si.inverse_mod(s, omega) * real_hash % omega
    fake_x = si.inverse_mod(fake_s, omega) * fake_hash % omega
    print(f"x = {x}\nfake x = {fake_x}")
    y = si.inverse_mod(s, omega) * r % omega
    fake_y = si.inverse_mod(fake_s, omega) * fake_r % omega
    print(f"y = {y}\nfake y = {fake_y}")

    return [(fake_r, fake_s), (x, y), (fake_x, fake_y)]

def broken_dsa_verification( element: int, xy: tuple, fake_xy: tuple, public_key: int, omega: int, group:int ) -> bool:
    """
    Skips the first step of verification. (size check of r and s)

    Params:
        element - Element g of group with ord(element) = omega;
        xy - x and y combined into tuple/list | Format: (x, y);
        fake_xy - faked x and y combined into tuple/list | Format: (fake_x, fake_y);
        public_key - Public key for the real/original signature = A;
        omega - order of group;
        group - The group aka p;

    Returns:
        True - If verification is successful;
        False - Something went wrong;
    """
    print("Skipping step 1 of verification (size checks of r and s)")
    real_verification = pow( pow(element, xy[0], group) * pow(public_key, xy[1], group), 1, group ) % omega
    fake_verification = pow( pow(element, fake_xy[0], group) * pow(public_key, fake_xy[1], group), 1, group ) % omega

    if real_verification == fake_verification:
        print("Successful verification")
        return True
    return False

def is_point_on_curve( curve: si.EC, point: si.Point ) -> bool:
    """
    Checks if a point is on a curve without modulo.

    Params:
        curve_ab - A si.EC object representing the curve;
        point_xy - A tuple containing x and y coordinates of the point (x, y);

    Returns:
        True - Point is on curve (both sides of equation match);
        False - Point is not on curve;
    """
    # y^2 == x^3 + ax + b
    return pow(point.y, 2) == pow(point.x, 3) + (curve.a * point.x) + (curve.b)

# def add_multiple_points( *points: si.Point ) -> si.Point:
#         #Fix floats -> fractions instead
#     """
#     Loops through the Points and adds them together.

#     Params:
#         points - A variable amount of si.Point objects;

#     Returns:
#         base_point - The result as si.Point object;
#     """
#     base_point = points[0]
#     for idx in range(1, len(points)):
#         base_point = base_point.add(points[idx])
#     return base_point

def calc_unknown_coord( curve: si.EC, x ):
    """
    Calculates an unknown coordinate of a point based on the curve.

    Params:
        curve - An elliptic curve the point is on (si.EC);
        xy - A tuple containing the coords (leave coord empty if unknown);

    Returns:
        coord - The unknown coordinate;
    """
    coord = math.sqrt( pow(x, 3) + (curve.a * x) + (curve.b) )
    return ( coord, -coord ) # plus/minus

def find_finite_order( point: si.Point ):
    """
    Continues adding point to itself until it fulfills the requirement.

    Params:
        point - The point to find the finite order of;

    Returns:
        point - Calculated point
    """
    original_point = point
    ctr = 2
    found = False
    while not found:
        point = point.add(original_point)
        ctr += 1
        if -original_point.y == point.y:
            found = True
    print(f"Finite order of P is: {ctr}P = ({point.x}, {point.y})")
    return point

def create_ec_signature( point: si.Point, k: int, omega: int, private_key: int, message_hash: int ) -> tuple:
    """
    Creates a signature with elliptic curves.

    Params:
        point - Aka G, a si.Point;
        k - random value;
        omega - Order of point G;
        private_key - The private key to sign with (alpha/beta);
        message_hash - The hash of the message;

    Returns:
        (r, s) - The signature;
    """
    r = point.mult(k).x
    s = pow(k, -1, omega) * (message_hash + private_key * r) % omega

    return (r, s)

def get_ec_pubkey( private_key: int, point: si.Point ) -> si.Point:
    """
    Calculates the public key from a private key.

    Params:
        private_key - Private key to find the pubkey of (alpha/beta);
        point - Point on curve;
    """
    return point.mult(private_key)

def check_ec_signature( signature: tuple, omega: int, message_hash: int, point: si.Point, public_key: si.Point ) -> bool:
    """
    Checks a signature based on elliptic curves.

    Params:
        signature - A tuple containing (r, s);
        omega - The order of point G;
        message_hash - The hash value of the message;
        point - Point G on curve;
        public_key - Point calculated from private key;
    """
    # 1 <= r < omega && 1 <= s < omega
    if (signature[0] in range (1, omega)) and (signature[1] in range(1, omega)):
        print("Passed Step 1")
    else:
        print("Failed first step")
        return

    # calc x and y
    x = si.inverse_mod(signature[1], omega) * message_hash % omega
    y = si.inverse_mod(signature[1], omega) * signature[0] % omega

    # check if pubkey matches sig
    if signature[0] == (point.mult(x) + public_key.mult(y)).x % omega:
        print("Passed entire verification")

def galois_field( prime: int, poly: list ):
    poly.reverse()
    p = ff.FiniteField(prime, poly)
    print(p.verbstr())
    tmp = list()
    try:
        for e in p:
            print(e)
            tmp.append(e)
    except RuntimeError:
        pass

    line = "(x)".center(15, " ")
    for e in tmp:
        line += f"{e}".center(15, " ")
    print(line)
    line = ""

    for e in tmp:
        line += f"{e}".center(15, " ")
        for a in tmp:
            line += f"{e*a}".center(15, " ")
    print(line)
    line = ""

class Polynomial:
    def __init__( self, poly: list, p ):
        if isinstance(p, Polynomial):
            self.mod_mode = "poly_mod"
        elif isinstance(p, int):
            self.mod_mode = "int_mod"
        elif p is None: # p == 0 default
            self.mod_mode = "no_mod"
        self.mod = p
        poly.reverse() # allow for intuative param and easy programming
        self.polynomial = poly

    def __str__( self ) -> str:
        poly = self.polynomial
        poly.reverse() # turn it around again
        exponent = len(poly) - 1 # all but last number are x
        output = "("
        for idx,num in enumerate(poly):
            if num == 0:
                exponent -= 1 # 1 less with every x
                continue # skip empty values

            if idx == 0:
                output += f"{num}x^{exponent} " # dont write symbols for first part
            elif idx > len(poly) - 2: # no x in last part
                if num > 0:
                    output += f"+ {num}"
                else:
                    output += f"- {-(num)}"
            elif num > 0:
                output += f"+ {num}x^{exponent} " # plus symbol
            elif num < 0:
                output += f"- {-(num)}x^{exponent} " # minus symbol and invert negative number
            exponent -= 1 # 1 less with every x
        if self.mod_mode != "no_mod":
            return output + f") mod {self.mod}"
        return output + ")"

    def add( self, p ):
        """
        Add two polynomials with the same modulo.

        Params:
            p - A polinomial to add

        Returns:
            The resulting polynomial
        """
        if not isinstance(p, Polynomial):
            print("Not a polynomial")
        if self.mod != p.mod:
            print("These polynomials don't share the same modulo.")
        poly1 = self.polynomial
        poly2 = p.polynomial

        min_val = min(len(poly1), len(poly2)) # see which list is shorter for index
        addition = [poly1[i] + poly2[i] for i in range(min_val)] # the longer list isn't fully added yet
        # check if values still need to be added
        if len(poly1) == len(poly2):
            addition.reverse()
            return Polynomial( addition, self.mod )
        if len(poly1) > len(poly2):
            for val in poly1[min_val:len(poly1):]:
                addition.append(val)
        elif len(poly1) < len(poly2):
            for val in poly2[min_val:len(poly2):]:
                addition.append(val)
        addition.reverse()
        return Polynomial( addition, self.mod )

    def __add__( self, p ):
        """
        Override + operator
        """
        return self.add( p )

    def sub( self, p ):
        """
        Subtracts two polynomials with the same modulo.

        Params:
            p - A polynomial to subtract

        Returns:
            The resulting polynomial
        """
        if not isinstance(p, Polynomial):
            print("Not a polynomial")
        if self.mod != p.mod:
            print("These polynomials don't share the same modulo.")
        poly1 = self.polynomial
        poly2 = p.polynomial

        min_val = min(len(poly1), len(poly2))
        subtraction = [poly1[i] - poly2[i] for i in range(min_val)]

        if len(poly1) == len(poly2):
            subtraction.reverse()
            return Polynomial( subtraction, self.mod )
        if len(poly1) > len(poly2):
            for val in poly1[min_val:len(poly1):]:
                subtraction.append(val)
        elif len(poly1) < len(poly2):
            for val in poly2[min_val:len(poly2):]:
                subtraction.append(-val)
        subtraction.reverse()
        return Polynomial( subtraction, self.mod )

    def __sub__( self, p ):
        """
        Override - operator
        """
        return self.sub( p )

    def mul( self, p ):
        """
        Multiply two polynomials

        Returns:
            Result as Polynomial
        """
        if not isinstance(p, Polynomial):
            print("Not a polynomial")
        if self.mod != p.mod:
            print("These polynomials don't share the same modulo.")
        poly1 = self.polynomial
        poly2 = p.polynomial

        # Reverse because numpy takes the list front to back
        poly1.reverse()
        poly2.reverse()
        poly1 = numpy.array(poly1)
        poly2 = numpy.array(poly2)
        multiplication = numpy.polymul( poly1, poly2).tolist()

        return Polynomial( multiplication, self.mod )

    def __mul__( self, p ):
        """
        Override *
        """
        return self.mul( p )

    def div( self, p ):
        """
        Divide polynomials
        """
        if not isinstance(p, Polynomial):
            print("Not a polynomial")
        if self.mod != p.mod and self.mod_mode != "poly_mod":
            print("These polynomials don't share the same modulo.")
        poly1 = self.polynomial
        poly2 = p.polynomial

        poly1.reverse()
        poly2.reverse()

        poly1 = numpy.array(poly1)
        poly2 = numpy.array(poly2)
        quotient, remainder = numpy.polydiv(poly1, poly2)

        quotient = quotient.tolist()
        remainder = remainder.tolist()
        return Polynomial( quotient, self.mod ), Polynomial( remainder, p.mod )

    def check_poly_mod( self ) -> None:
        """
        Checks if modulo needs to be done and calculates it.
        """
        if self.mod_mode == "no_mod":
            print("Polynomial has no modulo specified - no changes made.")
        elif self.mod_mode == "int_mod":
            for idx, num in enumerate(self.polynomial):
                self.polynomial[idx] = num % self.mod
        elif self.mod_mode == "poly_mod":
            if (len(self.polynomial) > len(self.mod.polynomial)) or min(self.polynomial, self.mod.polynomial) != self.mod.polynomial:
                quotient, remainder = self.div( self.mod ) # quotient not used
                self.polynomial = remainder.polynomial
                self.mod.polynomial.reverse()
            else:
                print("Polynomial is smaller than mod - no changes made.")
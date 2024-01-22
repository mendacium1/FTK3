"""
Useful functions for DML1, GDK2, and FTK3
Author: Jürgen Fuß
Date: 2022-06-03

list_primes( a, b ):
    return a list of all primes p, st. a <= p < b
prime_factors( n, unique=False ): 
    compute all prime factors of n (with multiplicity)
    if unique==True: compute all prime factors of n (without multiplicity)
euler_phi( n ):
    computes the value of the Euler totient funtion at n
equalmod(a, b, n):
    returns true, iff a = b mod n 
extended_gcd(a, b, verbose=0): 
    computes the extended GCD, i.e. gcd, x, y, st. gcd = ax + by
    shows all steps, when verbose>0
inverse_mod(a, m, verbose=0):
    computes the modular inverse of a mod m or None, if it does not exist
continued_fraction( a, b ):
    computes a continued fraction for a/b.
    returns a list
cf_approx( a ):
    returns a list of approximations for a continued fraction a
    approximations are pairs representing fractions
intsqrt( n ):
    returns the largest nonnegative number whose square is
    smaller than or equal to n
chinese_remainder( n, a ):    
    compute the result of the Chinese Remainder Theorem
    for a list of moduli n and a list of remainders a
    returns an integer
miller_rabin( n, max_rounds = 40 ):
    use the Miller-Rabin primality test to check, if n is a pseudoprime
isprime(n):
    checks, if n is prime with a Miller Rabin test over 64 rounds
next_prime(n):
    Returns the smallest prime greater than n.
multiplicative_order( a, p ):
    returns the smallest positive number x, st. pow(a,x,p)=1.
generator_mod( p ):
    returns a generating element mod p.
hasse_bounds( p ):
    returns upper and lower bounds for the order of an elliptic curve  mod p
order2params( p ):
    returns a dictionary with a list of parameter pairs for each possible order of an elliptic curve mod p
EC(a,b,p):
    construct an elliptic curve y^2=x^3+ax+b mod p
    methods and attributes:
        order(): number of points on the curve (computes all points!)
        points(): list of all elements of the curve
        zero(): returns the neutral element of the group
Point(c,(x,y)):
    construct point (x,y) on the elliptic curve c
    tests, whether (x,y) is on c, but only warns, if it is not
    methods and attributes:
        p == q: checks, if two points are equal
        p.inverse() or -p: compute additive inverse of point p
        p.add(q) or p+q: add the points p and q (checks, if curves are identical)
        p.double() or 2*p: double the point p
        p.mult(n) or n*p: compute n*p for any integer n
        p.order(): compute the order of the point p
gramschmidt( b ):
    transform the basis b into a orthonormal basis

See docstrings for functions and classes.
Run the file to see examples.
"""

import math, random
import sympy
from numpy import array, copy, zeros, eye, concatenate, ndarray, asarray

def _printv( v, threshold, string, end="\n" ):
    """
    Print depending on verbose level.

        Parameters:
            v (int): an integer
            threshold (int): an integer
            string (string): a string
            end="\n" (string): ending string for print() function

        Returns:
            Prints string, if v>th.
            Indentation decreases with verbose level v
    """
    if v>threshold:
        print( "'"+3*(4-v)*' ' + string, end=end )

### primes and factorisation

def list_primes( a, b ):
    """
    Returns a list of all primes in an interval.
    
        Parameters:
            a (int): an integer
            b (int): an integer

        Returns:
            list_primes( a, b ): The list of all primes p, st. a <= p < b
    """
    return list( sympy.sieve.primerange( a, b ) )

def prime_factors( n, unique=False ):
    """
    Returns all prime factors of n (with multiplicity).
    
        Parameters:
            n (int): a positive integer

        Returns:
            prime_factors( n ): A list of prime numbers whose product is n.
    """
    if unique:
        return list( sympy.ntheory.factorint(n).keys() )
    else:
        return sympy.ntheory.factorint( n, multiple=True )

def euler_phi( n ):
    """
    Returns the value of the Euler totient funtion at n.
    
        Parameters:
            n (int): a positive integer

        Returns:
            euler_phi( n ): the value of the Euler totient funtion at n.
    """
    factors = sympy.ntheory.factorint( n )
    return math.prod( (p-1) * p**(e-1) for (p,e) in factors.items() )

def equalmod( a, b, n, tol=1e-8 ):
    """
    Returns true, iff a = b mod n.

        Parameters:
            a (int): an integer
            b (int): an integer
            n (int): a positive integer
            tol (float): a positive float

        Returns:
            equalmod( a, b, n ): True, if n|a-b, False, otherwise.
                If n==0, then approximate a==b (tolerance tol) is returned.
    """
    if n==0:
        return abs(a-b) < tol
    else:    
        return (a-b)%n == 0

def extended_gcd( a, b, verbose=0 ):
    """ 
    Computes the extended GCD, i.e. gcd, x, y, st. gcd = ax + by.

        Parameters:
            a (int): an integer
            b (int): an integer
            verbose (int): a non-negative integer

        Returns:
            extended_gcd( a, b, verbose ): A triple (d,a,b), st. d=gcd(a,b) and d=ax+by.
                If verbose>0, the extended Euclidean algorithm is shown step-by-step. 
    """
    if a<b:
        (g,x,y) = extended_gcd( b, a, verbose=verbose )
        return (g,y,x)
    width = 3 + max( len(str(a)), len(str(b)) )
    width_plus = 2*width - 5
    x,y, u,v = 0,1, 1,0
    _printv( verbose, 0, f"{a:{width+width_plus}d}{b:{width_plus}d}" )
    while a != 0:
        q, r = b//a, b%a
        m, n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
        if q==0:
            _printv( verbose, 0, f"{b:{width}d}{x:{width_plus}d}{y:{width_plus}d}" )
        else:
            _printv( verbose, 0, f"{b:{width}d}{x:{width_plus}d}{y:{width_plus}d}{q:{width}d}" )
    gcd = b
    return gcd, x, y
    
def inverse_mod( a, m, verbose=0 ):
    """
    Computes the modular inverse of a mod m, or None, if it does not exist.
    
        Parameters:
            a (int): an integer
            m (int): a positive integer
            verbose (flag): a non-negative integer

        Returns:
            inverse_mod( a, m ): An integer b, st. ab=1 mod m.
                If gcd(a,m)>1, the result is None.
    """
    a = a % m
    gcd, x, y = extended_gcd( m, a, verbose )
    if gcd != 1:
        return None  # modular inverse does not exist
    else:
        return y % m

def continued_fraction( a, b ):
    """
    Computes a continued fraction for a/b.

        Parameters:
            a (int): a positive integer
            b (int): a positive integer

        Returns:
            continued_fraction( a, b ): A list which represents the continued fraction of a/b.
    """
    cf = []
    while b!=0:
       cf.append( a//b ) 
       a, b = b, a%b
    return cf

def cf_approx( a ):
    """
    Returns a list of approximations for a continued fraction a
       approximations are pairs representing fractions

        Parameters:
            a ([int]): a list representing a continued fraction

        Returns:
            cf_approx( a ): a list of pairs (x,y) st. the i-th x/y is the i-th approximation if a.
    """
    p = [ a[0], a[0]*a[1] + 1 ]
    q = [ 1, a[1] ]
    for n in range( 2, len(a) ):
        p.append( a[n]*p[n-1] + p[n-2] )
        q.append( a[n]*q[n-1] + q[n-2] )
    return list( zip( p, q ) )

def intsqrt( n ):
    """
    Returns a non-negative integer approximation of the square root of n
    
        Parameters:
            n (int): a positive integer
            
        Returns:
            intsqrt( n ): the largest integer s st. s*s <= n.
    """
    x = n
    y = (x+1) // 2
    while y < x:
        x = y
        y = (x + n//x) // 2
    return x

def chinese_remainder( ns, xs ):
    """
    Compute the result of the Chinese Remainder Theorem
        for a list of moduli ns and a list of remainders xs

        Parameters:
            ns ([int]): a list of positive integers
            xs ([int]): a list of integers

        Returns:
            chinese_remainder( ns, xs ): An integer x, st. x = xs[i] mod ns[i] for all i.
                A ZeroDivisionError is thrown, if the moduli are not pairswise relative prime.
    """
    prod = 1
    for ni in ns:
        if math.gcd( ni, prod ) > 1:
            raise ZeroDivisionError("Moduli have a common factor.")
        prod *= ni        
    res = 0
    for ni, xi in zip( ns, xs ):
        p = prod // ni
        res += xi * inverse_mod( p, ni ) * p
    return res % prod

def miller_rabin( n, max_rounds = 40, verbose=0, candidates=[] ):
    """
    Use the Miller-Rabin primality test to check if n is a pseudoprime.
    
        Parameters:
            n (int): a positive integer
            max_rounds (int): a positive integer
            verbose=0 (flag): show more information
            candidates ([int]): a list of bases to test
                If candidates=[], bases are randomly selected.

        Returns:
            miller_rabin( n, max_rounds ): False, if a witness against primality is found
                in the first max_rounds test rounds, True otherwise   
    """
    if n%2 == 0 and n!=2: # Miler-Rabin is not made for even numbers
        return False
    if candidates == []:
        if max_rounds > n-3:
            _printv( verbose, 0, f"There are only {n-3} bases to test. Testing all bases." )
            candidates = range( 2, n-2 )
        else:
            candidates = [ random.randint( 2, n-2 ) for _ in range( max_rounds ) ]

    # factor n-1 into 2**s * k with k odd
    k = n-1
    s = 0
    while( k%2 == 0 ):
        k //= 2
        s += 1
    # test for at most max_rounds rounds
    for a in candidates:
        witness = True
        # choose random basis a
        _printv( verbose, 0, f"Testing with basis a={a}", end=' ' )
        a = pow( a, k, n )
        _printv( verbose, 1, f"a**{k} == {a}" ,end='')
        if a == 1:
            # a**k == 1
            _printv( verbose, 0, '... success!' )
            continue
        for r in range( s ):
            if a == n-1:
                # a**(2**r)*k == -1
                # not a witness
                witness = False
                _printv( verbose, 0, '... success!' )
                break
            a = pow( a, 2, n )
            if r>0:
                _printv( verbose, 1, f"a**{2**r*k} == {a}", end='' )
        if witness:
            # witness found
            _printv( verbose, 0, 'Witness found!' )
            return False
    # max_rounds rounds survived
    return True

def isprime( n ):
    """
    Calls miller_rabin(n,max_rounds=64).
    I believe that the Miller Rabin test with 64 rounds is always correct.
    """
    # trial division
    for p in list_primes( 2, min(n,1_000_000) ):
        if n%p == 0:
            return False

    return miller_rabin( n, max_rounds=64 )

def multiplicative_order( a, n ):
    """
    Return the smallest positive number x, st. pow(a,x,n)==1.

        Parameters:
            a (int): an integer
            n (int): a positive integer

        Returns:
            multiplicative_order(a,p): the smallest positive number x, st. pow(a,x,n)=1.
    """
    assert math.gcd( a, n )==1, "a and n must be coprime."

    order = euler_phi( n )
    order_found = False
    while not(order_found):
        pf = prime_factors( order, unique=True )
        order_found = True
        for q in pf:
            if pow( a, order//q, n ) == 1:
                order = order//q
                order_found = False
                break
    return order

def generator_mod( p ):
    """
    Returns an element of multiplicative oder p-1.

        Parameters:
            p (int): a prime number

        Returns:
            generator_mod( p ): an element of multiplicative oder p-1.
    """
    assert isprime( p ), "p must be prime"
    while True:
        g = random.randint( 2, p-2 )
        if multiplicative_order( g, p ) == p-1:
            return g

def next_prime( n ):
    """
    Returns the smallest prime greater than n.

        Parameters:
            n (int): a positive integer

        Returns:
            next_prime(n): the smallest prime greater than n.
    """
    return sympy.ntheory.nextprime( n )

### elliptic curves

def hasse_bounds( p ):
    """
    Return a pair (a,b) of positive integers
    such that the order of a curve mod p
    is between a and b (including a and b).
    """
    a = math.ceil( p + 1 - 2*math.sqrt(p) )
    b = math.floor( p + 1 + 2*math.sqrt(p) )
    return ( a, b )


class EC:
    """
    A class to represent an eliptic curve with equation y^2==x^3+a*x+b mod p.
    Choose p==0 (default) for a real curve y^2==x^3+a*x+b.

    Attributes:
        a: int
            Parameter a in the curve equation
        b: int
            Parameter b in the curve equation
        prime: int
            Parameter p in the curve equation
    
    Methods:
        zero():
            Returns the neutral element of the group
        points():
            Returns a list with all points on the curve as Point's
        order():
            Returns the number of points on the curve (including the neutral element)
    """
    def __init__( self, a, b, prime=0 ):
        """
        Construct a curve object.

            Parameters:
                a (int): an integer
                b (int): an integer
                prime/0 (int): a prime number

            Returns:
                An object representing the curve y^2=x^3+ax+b mod prime.
                If prime==0, the real curve y**2=x**3+a*x+b is considered
        """
        assert prime==0 or isprime( prime ), "argument prime must be 0 or a prime number"
        self.a = a
        self.b = b
        self.prime = prime

    def __str__( self ):
        """
        A string with the curve equation.
        """
        real = self.prime==0
        if real:
            return f"y**2 = x**3 + {self.a}*x + {self.b}"
        else:
            return f"y**2 = x**3 + {self.a}*x + {self.b} mod {self.prime}"
       
    def __repr__( self ):
        """
        A string generating the curve.
        """
        return f"EC( {self.a}, {self.b}, {self.prime} )"
    
    def zero( self ):
        """
        Return the neutral element of the group as an Point.
        """
        return Point( self, None )
            
    def points( self ):
        """
        Compute a list of all points on the curve.
        
            Returns:
                A list with all points on the curve as Point's.
        """
        prime = self.prime
        real = prime==0

        if real:
            raise NotImplementedError('Cannot compute the points on a real curve.')
        coords = [ (x,y) for x in range(prime) for y in range(prime) if equalmod( y**2, x**3+self.a*x+self.b, prime ) ]
        return [ Point( self, coord ) for coord in coords ] + [ Point(self,None) ]

    def order( self ):
        """
        Compute the order of the group.
        """
        real = self.prime==0

        if real:
            raise NotImplementedError('Cannot compute the order of a real curve.')

        return len( self.points() )

class Point:
    """
    A class to represent points on an elliptic curve.

    Attributes:
        x: int/None
            x coordinate of the point, None for the neutral element
        y: int/None
            y coordinate of the point, None for the neutral element
        iszero: bool
            True iff the point is the neutral element
        ec: EC
            The curve on which the point lies.
    
    Methods:
        add(q,verbose=0):
            Add the point to the point q, return the result as a Point.
            If verbose>0 more information is printed.
        inverse():
            Return the additive inverse of the point as a Point.
        double(verbose=0):
            Double the point, return the result as a Point.
            If verbose>0 more information is printed.
        mult(k,verbose=0):
            Compute the k fold multiple of the point, return the result as a Point.
            If verbose>0 more information is printed.
        order():
            Compute the order (smallest multiple equal to the neutral element) of the point.

        Infix operators for group operations are available:
            pt1+pt2 = pt1.add(pt2)
            -pt     = pt.inverse()
            pt1-pt2 = pt1.add(pt2.inverse())
            k*pt    = pt.mult(k)
        Equality can be checked with the '==' operator.
    """
    def __init__( self, ec, coord ):
        """
        Construct an object representing a point on an elliptic curve.

            Parameters:
                ec (EC): An elliptic curve
                coord ((int,int)/None): A pair of integers representing the coordinates of the point,
                    None, if it is the neutral element.
            
            Returns:
                an object representing the point coord on the elliptic curve ec.
                If the point does not satisfy the curve equation, a warning is printed.
        """
        real = ec.prime==0

        if coord == None:
            self.iszero = True
            self.x = None
            self.y = None
        else: 
            self.iszero = False
            if real:
                self.x, self.y = coord
            else:
                self.x = coord[0] % ec.prime
                self.y = coord[1] % ec.prime
            
            if not( equalmod( self.y**2, self.x**3+ec.a*self.x+ec.b, ec.prime ) ):
                print( f"Warning: Point {coord} is not on the curve. Continuing ..." )
        self.ec = ec

    def __str__( self ):
        """
        A string for the point as a coordinate pair.
        """
        if self.iszero:
            return "∞"
        else: 
            return f"( {self.x}, {self.y} )"
    
    def __repr__( self ):
        """
        A string generating the point.
        """
        if self.iszero:
            return f"Point( {self.ec!r}, None )"
        else:
            return f"Point( {self.ec!r}, ( {self.x}, {self.y} ) )"

    def inverse( self ):
        """
        Compute the inverse of a point
        
            Returns:
                the additive inverse of the point as a Point.
        """
        if self.iszero:
            return self
        
        real = self.ec.prime == 0
        if real:
            return Point( self.ec, ( self.x, -self.y ) )
        else:
            return Point( self.ec, ( self.x, (-self.y)%self.ec.prime ) )
            
    def __neg__( self ):
        """
        Overload unary '-' operator
        """
        return self.inverse()
            
    def add( self, q, verbose=0 ):
        """
        Add two arbitrary points (may be the same point twice).

            Parameters:
                q (Point): a point on the same elliptic curve
                verbose=0 (int): If verbose>0 more information is printed.

            Returns:
                The sum of self and q as an Point.
        """
        width = 5 + len( f"adding {self} and {q}:" )
        _printv( verbose, 0, width*'_' )
        _printv( verbose, 0, f"adding {self} and {q}:" )
        if self.ec != q.ec:
            raise ("Cannot add points of two different elliptic curves")
        if self.iszero:
            _printv( verbose, 0, f"{self} + {q} = {q}" )
            _printv( verbose, 0, width*'=' )
            return q
        if q.iszero:
            _printv( verbose, 0, f"{self} + {q} = {self}" )
            _printv( verbose, 0, width*'=')
            return self
        p = self.ec.prime
        real = p==0
        x1, y1 = self.x, self.y
        x2, y2 = q.x, q.y
        if equalmod( x1, x2, p ):
            if equalmod( y1, y2, p ):
                return self.double( verbose )
            else: 
                _printv( verbose, 0, f"{self.ec.zero()}" )
                _printv( verbose, 0, width*'=' )
                return self.ec.zero()
        else: 
            if equalmod( x1, x2, p ):
                raise("(modular) division by zero")
            if real:
                k = (y1-y2) / (x1-x2)
            else:
                k = ( (y1-y2) * inverse_mod ( x1-x2, p, verbose-2 ) ) % p

            x3 = k**2 - x1 - x2
            y3 = -y1 + k*(x1-x3)
            if not real:
                x3 %= p
                y3 %= p
            _printv( verbose, 1, f"     k = ({y1}-{y2}) / ({x1}-{x2})" )
            if not real:
                _printv( verbose, 1, f"       = {(y1-y2)%p} / {(x1-x2)%p}" )
                _printv( verbose, 1, f"       = {(y1-y2)%p} * {inverse_mod(x1-x2,p)}" )
            _printv( verbose, 1, f"       = {k}" )
            _printv( verbose, 1, f"    x3 = {k}**2 - {x1} - {x2}" )
            _printv( verbose, 1, f"       = {x3}" )
            _printv( verbose, 1, f"    y3 = -{y1} + {k}*({x1}-{x3})" )
            if not real:
                _printv( verbose, 1, f"       = -{y1}  + {k}*{(x1-x3)%p}" )
            _printv( verbose, 1, f"       = {y3}" )
            _printv( verbose, 0, f"{self} + {q} = {Point( self.ec, (x3, y3) )}" )
            _printv( verbose, 0, width*'=' )
            return Point( self.ec, (x3, y3) )

    def __add__( self, q ):
        """
        Overload '+' operator
        """
        return self.add( q )

    def __sub__( self, q ):
        """
        Overload binary '-' operator
        """
        return self.add( q.inverse() )

    def double( self, verbose=0 ):
        """
        Double a point
        
            Parameters:
                verbose=0 (int): If verbose>0 more information is printed.

            Returns:
                The sum of self and self as an Point.
        """
        width = 10+len( f"doubling {self}:" )
        _printv( verbose, 0, width*'_' )
        _printv( verbose, 0, f"doubling {self}:" )
        if self.iszero:
            _printv( verbose, 0, f"2 * {self} = {self}" )
            _printv( verbose, 0, width*'=' )
            return self
        p = self.ec.prime
        real = p==0
        a = self.ec.a
        x1, y1 = self.x, self.y
        if equalmod(y1,0,p):
            _printv( verbose, 0, f"{self.ec.zero()}" )
            _printv( verbose, 0, width*'=' )
            return self.ec.zero()
        if real:
            k = ( 3* x1**2 + a ) / (2*y1)
        else:
            k = ( ( 3* x1**2 + a ) * inverse_mod( 2*y1, p, verbose-2 ) ) % p
        x3 = k**2 - 2*x1
        y3 = -y1 + k*(x1-x3)
        if not real:
            x3 %= p
            y3 %= p
        _printv( verbose, 1, f"     k = (3*{x1}**2 + {a}) / (2*{y1})" )
        if not real:
            _printv( verbose, 1, f"       = {(3* x1**2 + a)%p} / {(2*y1)%p}" )
            _printv( verbose, 1, f"       = {(3* x1**2 + a)%p} * {inverse_mod( 2*y1, p )}" )
        _printv( verbose, 1, f"       = {k}" )
        _printv( verbose, 1, f"    x3 = {k}**2 - 2*{x1}" )
        _printv( verbose, 1, f"       = {x3}" )
        _printv( verbose, 1, f"    y3 = -{y1} + {k}*({x1}-{x3})" )
        if not real:
            _printv( verbose, 1, f"       = -{y1} + {k}*{(x1-x3)%p}" )
        _printv( verbose, 1, f"       = {y3}" )
        _printv( verbose, 0, f"2 * {self} = {Point( self.ec, (x3, y3) )}" )
        _printv( verbose, 0, width*'=' )
        return Point( self.ec, (x3, y3) )
        
    def mult( self, k, verbose=0 ):
        """
        Computes the result of adding the point k times with itself using the binary method.
        
            Parameters:
                k (int): an integer
                verbose=0 (int): If verbose>0 more information is printed.

            Returns:
                The point k*self as an Point.
        """
        _printv( verbose, 0, f"computing {k} * {self}" )
        if k<0:
            _printv( verbose, 0, f"computing {-k} * (- {self})" )
            return self.inverse().mult( -k, verbose )
        s = self.ec.zero()
        base = self.add( s, verbose=0 )
        while k!=0:
            if k%2==1:
                _printv( verbose, 0, "adding ... " )
                s = s.add( base, verbose-1 )
            if k>1:
                _printv( verbose, 0, "doubling ... " )
                base = base.double( verbose-1 )
            k //= 2
        _printv( verbose, 1, '' )
        return s

    def __rmul__( self, k ):
        """
        Overload the '*' operator.
        """
        return self.mult( k )

    def _recursive_order( self, o ):
        """
        Internal function used to determine the order of a point in a recursive fashion.
        """            
        plist = prime_factors( o, unique=True )
        for p in plist:
            opt = self.mult( o//p )
            if opt.iszero:
                return self._recursive_order( o//p )
        return o 

    def order( self ):
        """
        Compute the order of the point.
        
            Returns:
                The smallest (positive) k, st. k*self is equal to the neutral element.
        """
        return self._recursive_order( self.ec.order() )
        
    def __eq__( self, other ):
        """
        Overload the '==' operator.
        """
        return (self.x, self.y) == (other.x, other.y)        

def order2params( p ):
    """
    Return a dictionary of all orders
    and lists of corresponding parameter pairs (a,b)
    (all curves mod p)"""
    assert p<=10000, "p is too large (>10000). This would take too long ..."
    assert isprime(p), "p must be a prime number"
    order_param = {}
    (l,u) = hasse_bounds(p)
    for o in range( l, u+1 ):
        order_param[o] = []
        
    for a in range( p ):
        for b in range( p ):
            if (4*a**3 + 27*b**2) % p == 0: # singular curve
                continue
            o = EC( a, b, p ).order()
            order_param[o] += [ (a,b) ]
    return order_param

### vector spaces

def gramschmidt( v ):
    """
    Return an orthonormal basis for the linear hull of v
    
        Parameters:
            v([np.array]): a list of vectors
                or a matrix with basis vectors as columns
            
        Returns:
            gramschmidt( v ): a list of orthonormal vectors that form a basis
                for the linear hull of the vectors in v
                (in the same format as the argument)
    """

    if type(v) == type( array( [[0,0],[0,0]] ) ):
        # argument is a matrix -> parse as a list of column vectors
        matrixtype = True
        v = [ v.T[i] for i in range( len(v.T) ) ]
    else:
        matrixtype = False

    u = len(v)*[0]
    for i in range( len(v) ):
        u[i] = v[i]
        for j in range( i ):
            u[i] = u[i] - ( v[i]@u[j] ) * u[j]  # project
        u[i] = 1/math.sqrt( u[i]@u[i] ) * u[i]  # normalize

    if matrixtype:
        # argument was a matrix -> return a matrix
        # argument is a list of vectors -> return a list of vectors
        return array( u ).T
    else:
        return u

### binary matrices

class BinaryMatrix(ndarray):
    """
    takes a numpy matrix, checks if all entries are either 0 or 1
    Methods:
        binary_rref(): compute reduced row echelon form modulo 2
        binary_nullspace(): compute a basis of the nullspace of the matrix
    """

    def __new__(cls, input_array):
        assert BinaryMatrix.isbinary(input_array), 'Matrix is not a binary matrix!'
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    @classmethod
    def isbinary( self, mat ):
        """ a matrix is binary, if every entry is either 0 or 1 """
        return all( [ mat[i,j] == 0 or mat[i,j]==1 for i in range(mat.shape[0]) for j in range(mat.shape[1]) ] )
    
    def binary_rref( self ):
        """ compute the row echelon form of a binary matrix over GF2 """
        mat = copy(self)
        m,n = mat.shape
        piv = []
        # subdiagonal
        row = 0
        for col in range( n ):
            if row<m:
                if mat[row,col]==0:
                    i = row+1
                    while i<m and mat[i,col]==0:
                        i += 1
                    if i<m:
                        mat[[row,i]] = mat[[i,row]]
                if mat[row,col]==1:
                    piv.append(col)
                    for i in range(row+1,m):
                        if mat[i,col]==1:
                            mat[i] = (mat[i]+mat[row])%2
                    row += 1
        # superdiagonal
        for col in range( n-1, -1, -1 ):
            if col in piv:
                row = piv.index(col)
                for i in range(row):
                    if mat[i,col]==1:
                        mat[i] = (mat[i]+mat[row])%2
        return mat, piv
    
    def binary_nullspace( self ):
        """ compute a basis for the null space of a binary matrix over GF2 """
        n = self.shape[1]
        rr, piv = self.binary_rref()
        nonpiv = [i for i in range(n) if not( i in piv ) ]
        basis = []
        for i in range(n):
            if i in piv:
                basis.append( list(rr[:,nonpiv][piv.index(i),:]) )
            else:
                basis.append( list(eye(len(nonpiv))[nonpiv.index(i),:]) )
        return BinaryMatrix(array(basis))

    def __matmul__( self, v ):
        """ inherited matrix multiplication is not mod 2 """
        return ndarray.__matmul__( self, v )%2

_EXCTR=0
def _ex( string ):
    global _EXCTR
    print( ' '+60*'_','\n EX', _EXCTR:=_EXCTR+1, string,'\n', 60*'_' )

if __name__ == "__main__":
    # ==============
    # == Examples ==
    # ==============
    
    # number theory
    _ex('primes')
    print( '> list_primes(7,19)' )
    print( list_primes(7,19) )
    print( '> next_prime(2**50+2**30)' )
    print( next_prime(2**50+2**30) )

    _ex('prime_factors')
    print( '> prime_factors( 57480192000 )' )
    print( prime_factors( 57480192000 ) )
    print( '> prime_factors( 57480192000, unique=True )' )
    print( prime_factors( 57480192000, unique=True ) )

    _ex('euler_phi')
    print( '> euler_phi( 3*25*343 )' )
    print( euler_phi( 3*25*343 ) )

    _ex('equalmod')
    print( '> equalmod(199,99,10)' )
    print( equalmod(199,99,10) )
    print( '> equalmod(0,1e-10,0)' )
    print( equalmod(0,1e-10,0) )

    _ex('extended_gcd')
    print( '> extended_gcd(1035,336)' )
    print( extended_gcd(1035,336) )
    print( '> extended_gcd(1035,336, verbose=1 )' )
    print( extended_gcd(1035,336, verbose=1 ) )

    _ex('inverse_mod')
    print( inverse_mod(17,33,verbose=1) )
    
    _ex('continued fractions')
    print( '> continued_fraction( 123456789, 654321 )' )
    print( continued_fraction( 123456789, 654321 ) )
    print( '> cf_approx( [188, 1, 2, 8, 1, 1, 66, 1, 14, 4] )' )
    print( cf_approx( [188, 1, 2, 8, 1, 1, 66, 1, 14, 4] ) )

    _ex('chinese remainder')
    print( '> chinese_remainder( [3,4,5], [1,2,3] )' )
    print( chinese_remainder( [3,4,5], [1,2,3] ) )

    _ex('miller_rabin')
    print( '> isprime( 119 ) ')
    print( isprime( 119 ) )
    print( '> miller_rabin( 113, max_rounds=4, verbose=2 )' )
    print( miller_rabin( 113, max_rounds=4, verbose=2 ) )
    print( '> miller_rabin( 119, candidates=[17,2], verbose=1 )' )
    print( miller_rabin( 119, candidates=[17,2], verbose=1 ) )

    _ex('order')
    print( '> multiplicative_order( 19, 48 )' )
    print( multiplicative_order( 19, 48 ) )
    print( '> generator_mod( 37 )' )
    print( generator_mod( 37 ) )

    _ex('real EC')
    print( '> c = EC(1/2,0)' )
    c = EC(1/2,0)
    print( '> g = Point( c, (2,3) )' )
    g = Point( c, (2,3) )
    print( '> g' )
    print(repr(g))
    print( '> g+g' )
    print(g+g)
    print( '> -12*g' )
    print(repr(-12*g))

    _ex('EC mod p')
    print( '> hasse_bounds(5)' )
    print( hasse_bounds(5) )

    print( '> order2params(5)' )
    print( order2params(5) )

    print( '> EC( 1, 4, 5 ).order()' )
    print( EC( 1, 4, 5 ).order() )

    #define the curve y**2 = x**3 + 2x + 3 mod 541                    
    print( '> curve = EC(2,3,541)' )
    curve = EC(2,3,541)
    print( '> Point(c,(1,1))' )
    print( repr(Point(c,(1,1))) )

    # compute all curve points
    print( '> cp = curve.points()' )
    cp = curve.points()

    print( '> cp[100]' )
    print( repr(cp[100]) )

    # compute the order of the curve
    print ( '> curve.order()' )
    print ( curve.order() )
    
    # define points (3,6) and (7,3) on the curve
    print( '> q = Point(curve,(4,285))' )
    q = Point(curve,(4,285))
    print( '> r = Point(curve, (93,151))' )
    r = Point(curve, (93,151))
    
    print( '> -q' )
    print( repr(-q) )

    # use the neutral element of the curve
    print( '> curve.zero()' )
    print( repr(curve.zero() ) )
    # z = Point(curve,None) is equivalent
    
    _ex( 'EC point addition' )
    # double q
    print( '> q.double()' )
    print( repr(q.double()) )
    print( '> q+q' )
    print( repr(q+q) )

    # verbose output
    print( '> q.double(1)' )
    print( repr(q.double(1)) )
    # more verbose output
    print( '> q.double(2)' )
    print( repr(q.double(2)) )
    # even more verbose output
    print( '> q.double(3)' )
    print( repr(q.double(3)) )
 
     # add q and r
    print( '> q+r' )
    print( repr(q+r) )
    # verbose
    print( '> q.add(r,2)' )
    print( repr(q.add(r,2)) )

    _ex( 'EC point multiplication' )
    # compute multiples of points
    # 9*r
    print( '> 9*r' )
    print( repr(9*r) )
    print( '> r.mult(9,1)' )
    print( repr(r.mult(9,1)) )
    print( '> q.mult(3,2)' )
    print( repr(q.mult(3,3)) )
    
    _ex('EC orders')
    # compute orders of points
    print( '> for pt in cp[18:24]:' )
    print( '>    pt, " has order ",pt.order()' )
    for pt in cp[18:24]:
        print (repr(pt), " has order ",pt.order())

    _ex('Gram-Schmidt for a list of vectors')
    # compute an orthonormal basis
    print( '> v1=array([2,4,4,0])' )
    print( '> v1=array([2,4,4,0])' )
    print( '> v2=array([4,3,4,2])' )
    print( '> v3=array([-1,-3,-1,4])' )
    v1=array([2,4,4,0])
    v1=array([2,4,4,0])
    v2=array([4,3,4,2])
    v3=array([-1,-3,-1,4])
    print( '> gramschmidt([v1,v2,v3])' )
    print( gramschmidt([v1,v2,v3]) )

    _ex('Gram-Schmidt for a matrix')
    print( '> b = array([v1,v2,v3]).T' )
    print( '> gramschmidt(b)' )
    b = array([v1,v2,v3]).T
    print( gramschmidt(b) )

    _ex('Null space of a binary matrix')
    print( '> m = array( [[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]] )' )
    m = array( [[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]] )
    print( '> b = BinaryMatrix(m)' )
    b = BinaryMatrix(m)
    print( '> b.binary_nullspace()' )
    print( b.binary_nullspace() )
    print( '> b.binary_rref()' )
    print( b.binary_rref() )


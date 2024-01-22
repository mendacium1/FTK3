#import si
#from si import EC, Point
import finitefield as ff
#from sympy.abc import x

"""
GF = ff.FiniteField(2,[1,1,0]) # 1 + x +x^3
print("Field:", GF.verbstr())
last = ff.FiniteFieldElt(GF,[1,1,1])
print("last =", last.verbstr())
inv_last = last.inv()
print("last inv =",inv_last.verbstr())

middle = ff.FiniteFieldElt(GF,[1,0,1,0,1])
print("middle =", middle.verbstr())
first = ff.FiniteFieldElt(GF,[1,1])
print("first =", first.verbstr())

first4 = (first**4)
print("first**4 =", first4.verbstr())

addi = middle.add(first4)
print("addition First**4 + middle =", addi.verbstr())
div = middle.div(first)
print("div =", div.verbstr())
mul = addi.mult(inv_last)
print("Result =", mul.verbstr())

"""

"""
GF = ff.FiniteField(2,[1,1,0]) # 1 + x +x^3
print("Field:", GF.verbstr())

first = ff.FiniteFieldElt(GF,[0,1,1]) #x^2+x
print("first =", first.verbstr())

first4 = (first**4)
print("first4 =", first4.verbstr())

second = ff.FiniteFieldElt(GF,[1,1,1]) # x^2+x+1
print("second=",second.verbstr())

result = first4*second
print("result=",result.verbstr())
"""
# 
GF = ff.FiniteField(2,[1,0,1])
print("Field:", GF.verbstr())

#a)
"""
first = ff.FiniteFieldElt(GF,[1,0,1])
print("first =", first.verbstr())
second = ff.FiniteFieldElt(GF,[1,1])
print("second=",second.verbstr())
res = first.mult(second)
print("Result =", res.verbstr())
"""
#b)
"""
first = ff.FiniteFieldElt(GF,[1,0,1])
print("first =", first.verbstr())
second = ff.FiniteFieldElt(GF,[1,1])
print("last =", second.verbstr())
second_inv = second.inv()
print("second_inv =", second_inv.verbstr())
res = first.mult(second_inv)
print("Result =", res.verbstr())
"""

#c)
first = ff.FiniteFieldElt(GF,[0,1])
print("first =", first.verbstr())
first1001 = first**1001
print("first1001 =", first1001.verbstr())
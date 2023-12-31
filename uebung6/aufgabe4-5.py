#Winternitz 

# 4)
# Da auf die Prüfbits verzichtet wurde gibt es weniger Redundanz.
# Beginnt eine Nachricht gleich, so beginnt auch ihre Signatur gleich:
# Sind die ersten 4 Bit gleich, so ist auch deren Signatur gleich.
# Für die Nachricht "Ich hasse dich" ohne Prüfbits könnte man theoretisch eine Signatur für "Ich küsse dich" erzeugen, wenn man den entsprechenden Block m0m0​ findet, der nur um eins erhöht werden muss, um von "hasse" zu "küsse" zu wechseln. Man müsste dann den gehashten Wert dieses Blocks m0m0​-mal hashen, um den neuen Wert zu erhalten.

# Falsch:
# Da zb. von "Ich hasse dich" auf "Ich küsse dich" nur zwei Buchstaben unterschiedlich sind und es keine Prüfbits gibt, erleichter dies einen Bruteforceangriff.
# Der Schlüsselraum ist nicht mehr "67 Werte zwischen 0 und 15" groß

# 5) 
#Für die Nachricht "Ich liebe dich" wäre es komplizierter, da der Unterschied in den Blöcken größer ist als eine einfache Inkrementierung um eins. Daher ist es unwahrscheinlich, dass eine gültige Signatur für diese Nachricht aus der Signatur von "Ich hasse dich" ohne Kenntnis des Private Keys abgeleitet werden kann.


"""
Veränderungen:

hasse auf küsse
h = 68 = 0110 1000
k = 6B = 0110 1011
-> 2 Unterschiede

a = 61 = 0110 0001
ü = FC = 1111 1100
-> 4 Unterschiede


hasse auf liebe
h = 68 = 0110 1000
l = 6C = 0110 1100
-> 1 Unterschied

a = 61 = 0110 0001
i = 69 = 0110 1001
-> 1 Unterschied

s = 73 = 0111 0011
e = 65 = 0110 0101
-> 3 Unterschiede

s = 73 = 0111 0011
b = 62 = 0110 0010
-> 1 Unterschied


"""

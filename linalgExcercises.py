import numpy as np

#A = [[1, 4, 5, 12],
#    [-5, 8, 9, 0],
#    [-6, 7, 11, 19]]

#print("A =", A)
#print("A[1] =", A[1])      # 2nd row
#print("A[1][2] =", A[1][2])   # 3rd element of 2nd row
#print("A[0][-1] =", A[0][-1])   # Last element of 1st Row

#column = [];        # empty list
#for row in A:
# column.append(row[2])

#print("3rd column =", column)

print("Riki Karjalainen, 769875, Excercise sheet 1 ")
print("")
print("")

x = np.array([2,3,4]).transpose()
y = np.array([1,0,2]).transpose()
z = np.array([0,1,0]).transpose()
z = np.array([0,1,0]).transpose()
w = np.array([0,0,3]).transpose()
v = np.array([0,2,2]).transpose()

A = np.array([x,y,z])
B = np.array([x,y,w])
C = np.array([x,z,v])

print("1. x=[2 3 4]T, y=[1 0 2]T, z=[0 1 0]T") #######1 alkaa
print("")
print("a) Vectors are linearly dependent, since 2y + 3z = x.")
print("Determinant of 3x3 matrix that has x, y, and z: ", np.linalg.det(A))
print("")

print("b) w can for example be [0 0 3]. Now none of x, y, and w can be expressed as a linear combination of the other two.")
print("Determinant of 3x3 matrix that has x, y, and w: ", np.linalg.det(B))
print("")

print("c) v can for example be [0 2 2]. Now none of x, z, and c can be expressed as a linear combination of the other two.")
print("Determinant of 3x3 matrix that has x, z, and v: ", np.linalg.det(C))


print("")
print("")


#########2 alkaa

print("2. x=[1 2 3]T")
print("")
print("a) x = e_1 + 2e_2 + 3e_3")
print("")
print("b) q1=[1 1 0]T, q2=[1 0 1]T, q3=[1 1 1]T")

E = np.array([[1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])

abc = np.array([1, 2, 3])
bvast = np.linalg.inv(E).dot(abc)
print("Coefficients for q1, q2, and q3 to represent x: ", bvast)
print("")

print("c) v1=[-1 1 -1]T, v2=[1 2 2]T, v3=[1 -2 1]T")

Ec = np.array([[-1, 1, 1],
              [1, 2, -2],
              [-1, 2, 1]])

bvast = np.linalg.inv(Ec).dot(abc)
print("Coefficients for v1, v2, and v3 to represent x: ", bvast)



print("")
print("")


A3 = np.array([[1, -1, 0],
              [1, 0, -1],
              [0, -2, 2]])
b = np.array([1,-3,8]).transpose()


print("3. A= ")
print(A3, ", b=[1 -3 8]T")
print("")
print("First finding the null space of A. Using gaussian elimination I got the matrix")
print("1 0 -1|0")
print("0 1 -1|0")
print("0 0 0 |0")
print("from there it's pretty simple to get x1=x3, x2=x3, x3=x3, so the null space basis N(A) is [s s s]T")
print("")

print("Then let's solve Ax=b. Again with gaussian elimination by hand we get:")
print("1 0 -1|-3")
print("0 1 -1|-4")
print("0 0 0 |0")
print("Therefore x = [s-3 s-4 s]T")

print("")
print("")
################ 4 alkaaaaaaaaaaaaaaaaaaaaa

print("Excercise 4 is on different file")





print("")
print("")
print("")


import numpy as np
import matplotlib.pyplot as plt

#monomial (단항식)

##데이터 생성
X = np.random.rand(100)
pure_y = 6.12 * X+7.05
noise = np.random.uniform(-0.5, 0.5, 100)
Y = pure_y + noise
print(Y)

plt.scatter(X, Y)
plt.plot(X, pure_y)
plt.show()

##gradient descent in polynomial

#STEP 1
#set starting point as 0

a = 0
b = 0

#number of observations : n
n = float(len(X))

L = 0.05  #learning rate
I = 1500  #number of iterations

#STEP 2
errors=[ ] 
for i in range(I):

    Y_pred = a*X + b

    deriv_a = (-2/n)*sum(X*(Y-Y_pred))
    deriv_b = (-2/n)*sum(Y-Y_pred)

    a = a - L*deriv_a
    b = b - L*deriv_b

    Q= ((Y_pred - Y)**2).mean()     #((Y-Y_pred)**2).mean()   #(1/n)*sum((Y-Y_pred)**2)
    errors.append(Q)

print("prediction for parameters: a:{0}, b:{1}, Q:{2}".format(a, b, Q))


print(errors) #to show it converges 
 


#STEP 4
Y_pred = a * X + b


plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color="red")
plt.show()
'''

'''
#오류가 어떠한 수로 수렴함을 보여줌
plt.figure(figsize=(10, 10))
plt.plot(errors)
plt.show()


##polynomial (다항식)
X_1 = np.random.rand(100)
X_2 = np.random.rand(100)
X_3 = np.random.rand(100)
noise = np.random.uniform(-0.5, 0.5, 100)

pure_y = 3.5*X_1 + 5.9*X_2 + 7*X_3 + 3.7
Y = pure_y + noise
print(Y)

##gradient descent in polynomial
#step1

w1=0
w2=0
w3=0
b=0

L = 0.05
I = 5000
n = 100
#STEP 2
errors=[ ]
for i in range(I):

    Y_pred = w1*X_1 + w2*X_2 + w3*X_3 + b

    deriv_w1 = (-2/n)*sum(X_1*(Y-Y_pred))
    deriv_w2 = (-2/n)*sum(X_2*(Y-Y_pred))
    deriv_w3 = (-2/n)*sum(X_3*(Y-Y_pred))

    deriv_b = (-2/n)*sum(Y-Y_pred)

    w1 = w1 - L * deriv_w1
    w2 = w2 - L * deriv_w2
    w3 = w3 - L * deriv_w3
    b = b - L*deriv_b

    Q= ((Y_pred - Y)**2).mean()     #((Y-Y_pred)**2).mean()   #(1/n)*sum((Y-Y_pred)**2)
    errors.append(Q)

print("prediction for parameters: w1:{0}, w2:{1}, w3:{2}, b:{3}, Q:{4}".format(w1, w2, w3, b, Q))

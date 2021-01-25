import numpy as np

#gradient(differentiation)
def gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

#gradient descent
def gradient_descent(f, init_x, L = 0.01, I = 20):
    x = init_x

    for i in range(I):
        grad = gradient(f,x)
        x -= L *grad

    return x


#Process
def gradient_descent_process(f, init_x, L = 0.01, I = 100):
    x = init_x
    process = []

    for i in range(I):
        process.append(x.copy())
        grad = gradient(f,x)
        x -= L*grad

    return x, np.array(process)


#Test
def function_1(x):
    return x[0]**2 +x[1]**2

x, x_process = gradient_descent_process(function_1, np.array([-3.0, 4.0]), L = 0.1, I = 50 )
print(x_process)
print(x)

from matplotlib.pylab import plt

#graph 1 (2 dimensional graph)
plt.plot( [-5, 5], [0,0], '-b')    #color = blue
plt.plot( [0,0], [-5, 5], '-b')    #color = blue

plt.plot(x_process[:,0], x_process[:,1], '.-') #수렴과정
plt.plot(0, 0, '+' , color = "red") #극소점

plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

#graph 2 (contour graph)
x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)

X, Y = np.meshgrid(x, y)
Z = function_1(np.array([X, Y]))

idx = 1
plt.subplot(2, 2, idx)
idx += 1
plt.plot( x_process[:,0], x_process[:,1], '.-', color="blue") #수렴과정
plt.contour(X, Y, Z)

plt.ylim(-5, 5)
plt.xlim(-5, 5)

plt.plot(0, 0, '+', color = "red")  #극소점
plt.show()


#graph 3 (3 dimensional graph)
x, x_process = gradient_descent_process(function_1, np.array([-3.0, 4.0]), L = 0.1, I = 50 )

x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)
x, y = np.meshgrid(x, y)
z = function_1(np.array([X, Y]))


fig = plt.figure()
ax = fig.gca(projection ="3d")
surf = ax.plot_wireframe(x, y, z , color='grey', alpha = 0.2)    #수렴과정 (불투명도 = 0.2) 
ax.plot3D(x_process[:,0], x_process[:,1],np.array([function_1(x_process[i,:]) for i in range(len(x_process))]) , color = "blue", alpha = 0.7) #수렴과정 (불투명도 = 0.7)
ax.plot3D(0, 0, '+', color = "red") #극소점 
plt.show()


#gradient descent to find the parameters
X_1 = np.random.rand(100)
X_2 = np.random.rand(100)
X_3 = np.random.rand(100)
noise = np.random.uniform(-0.5, 0.5, 100)


def function_1(x):
    pure_y = 3.5 * X_1 + 5.9 * X_2 + 7 * X_3 + 3.7
    Y = pure_y + noise

    Y_pred = x[0] * X_1 + x[1] * X_2 + x[2] * X_3 + x[3]
    return (((Y_pred - Y) ** 2).mean())


print(gradient_descent(function_1, np.array([0.0, 0.0, 0.0, 0.0]), L=0.05, I=1000))

x, x_process = gradient_descent_process(function_1, np.array([0.0, 0.0, 0.0, 0.0]), L = 0.05, I = 1000 )
print(x_process)
print(x)

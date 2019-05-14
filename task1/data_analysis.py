from dataset import MNIST
import matplotlib.pyplot as plt

mnist = MNIST(shuffle=True)
num = 0
for image in mnist.X:
    zeros = 0
    one = 0
    num += 1
    num_list = []
    for number in image:
        if number == 0:
            zeros += 1
        elif number == 1:
            one += 1
        else:
            num_list.append(number)
    plt.figure(num)
    image = image.reshape((28, 28))
    plt.imshow(image)
    plt.show()
    print(num_list)

    print('num ', num, 'one:', one, 'zero:', zeros)

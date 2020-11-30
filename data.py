import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = []
    for i in range(12, 172):
        filename = './results/' + str(i) + '.txt'
        with open(filename, 'r') as in_file:
            result = []
            for line in in_file:
                temp = line.strip('\n')
                if temp != '':
                    result.append(float(temp))
            in_file.close()
        data.append(result[0])
    iteration = []

    for i in range(0, 8):
        iteration.append([i]*20)
    plt.xlabel('iteration')
    plt.ylabel('cost(-peak density / 3e+8)')
    plt.title('Compress MOT Optimization')
    plt.plot([0, 9], [-0.733, -0.733], linewidth='0.5')
    plt.scatter(iteration, data)
    plt.show()

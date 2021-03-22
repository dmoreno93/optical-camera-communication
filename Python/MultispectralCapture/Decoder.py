import numpy as np
from matplotlib import pyplot as plt


def create_header(bits, samples_per_bit):
    return np.concatenate((np.ones((bits-1)*samples_per_bit), np.zeros(samples_per_bit)))


def create_bit_waveform(bit, samples_per_bit, duty):
    on_samples = int(samples_per_bit*duty)
    off_samples = samples_per_bit - on_samples
    on_period = np.ones(on_samples)
    off_period = np.zeros(off_samples)

    return np.concatenate((on_period, off_period)) if bit == 1 else np.concatenate((off_period, on_period))


def create_signals(bits, samples_per_bit, header_length, duty):
    combinations = 2**bits
    header = create_header(header_length, samples_per_bit)

    matrix = np.zeros((combinations, samples_per_bit*(header_length + bits)))
    for i in range(matrix.shape[0]):
        bit_string = [(i >> k) & 0x01 for k in range(bits)]
        bit_string = reversed(bit_string)
        placeholder = np.empty(0)
        for bit in bit_string:
            placeholder = np.concatenate((placeholder, create_bit_waveform(bit, samples_per_bit, duty)))

        matrix[i, :] = np.concatenate((header, placeholder))
        matrix[i, :] -= np.mean(matrix[i, :])
        matrix[i, :] /= np.std(matrix[i, :])

    return matrix


def perform_correlation(input_data, matrix):
    window = matrix.shape[1]
    correlation_result = np.zeros((matrix.shape[0], len(input_data) - window + 1))
    for displacement in range(correlation_result.shape[1]):
        aux = input_data[displacement:displacement+window]
        correlation_result[:, displacement] = np.matmul(matrix, (aux - np.mean(aux))/np.std(aux))

    return correlation_result/window


def arreglar_desastre(input_data):
    output_data = np.zeros((len(input_data), input_data[0].shape[0]))
    for index, firma in enumerate(input_data):
        output_data[index, :] = firma
    return output_data


def execute(band):
    list_filename = 'captures/list.npy'
    data = np.load(list_filename, allow_pickle=True)

    detector_matrix = create_signals(bits=8, samples_per_bit=10, header_length=5, duty=0.8)
    bits_err = 0
    for i in range(len(data)):
        captures = np.load("captures/" + str(i) + ".npy", allow_pickle=True)
        captures = arreglar_desastre(captures)
        correlations = perform_correlation(captures[:, band], detector_matrix)
        max_corr = np.max(np.max(correlations))
        result = np.argwhere(correlations == max_corr)
        # err = ord(data[i]) - result[0, 0]  # if char are sent
        #print(format(data[i], '08b'))
        #print(format(result[0, 0], '08b'))
        data_bin = format(data[i], '08b')
        #print(int(data_bin[0]))
        result_bin = format(result[0, 0], '08b')
        #print(result_bin[1])
        for j in range(8):
            if int(data_bin[j], 2) ^ int(result_bin[j], 2):
                bits_err += 1

        #print(bits_err)
        #data_bin ^ result_bin
        #err = data[i] - result[0, 0]  # if int are sent
        #print(err)
    print(bits_err/len(data)/8)


def visualize(num):
    for i in range(num):
        captures = np.load("captures/" + str(i) + ".npy", allow_pickle=True)
        captures = arreglar_desastre(captures)
        plt.figure(1)
        plt.plot(captures)
        plt.show()


def view_waveforms():
    for k in range(500):
        captures = np.load("captures/" + str(k) + ".npy", allow_pickle=True)
        captures = arreglar_desastre(captures)
        plt.figure(1)
        pos = 1
        for i in range(3):
            for j in range(3):
                plt.subplot(3, 3, pos)
                plt.plot(captures[:, pos-1])
                pos += 1

        plt.show()


if __name__ == "__main__":
    view_waveforms()
    #for k in range(9):
    #    execute(k)

        #visualize(10)

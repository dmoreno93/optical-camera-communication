import numpy as np
from matplotlib import pyplot as plt


def create_header(bits, samples_per_bit):
    """
    Create the header of the bit stream. The last bit of the header is 0.
    :param bits: number of header bits
    :param samples_per_bit: samples per bit (S) that depend on the bit period (Tb) and the frame rate (fps): S = Tb/(1/fps)
    :return: header considering the samples per bit
    """
    return np.concatenate((np.ones((bits - 1) * samples_per_bit), np.zeros(samples_per_bit)))


def create_bit_waveform(bit, samples_per_bit, duty):
    """
    Create the waveform for a Manchester encoded bit stream.
    :param bit: number of data bits
    :param samples_per_bit: samples per bit (S) that depend on the bit period (Tb) and the frame rate (fps): S = Tb/(1/fps)
    :param duty: PWM duty cycle
    :return: waveform
    """
    on_samples = int(samples_per_bit * duty)
    off_samples = samples_per_bit - on_samples
    on_period = np.ones(on_samples)
    off_period = np.zeros(off_samples)

    return np.concatenate((on_period, off_period)) if bit == 1 else np.concatenate((off_period, on_period))


def create_signals(bits, samples_per_bit, header_length, duty):
    """
    Create all available (2^bits) signals (header + data) from the passed parameters to use them in the correlation
    process later.
    :param bits: number of data bits
    :param samples_per_bit: samples per bit (S) that depend on the bit period (Tb) and the frame rate (fps): S = Tb/(1/fps)
    :param header_length: number of header bits
    :param duty: PWM duty cycle
    :return: matrix with all combinations. Matrix dimension: 2^bits x ( (header_length + bits) * samples_per_bit)
    """
    combinations = 2 ** bits
    header = create_header(header_length, samples_per_bit)
    matrix = np.zeros((combinations, samples_per_bit * (header_length + bits)))
    for i in range(matrix.shape[0]):
        bit_string = [(i >> k) & 0x01 for k in range(bits)]
        bit_string = reversed(bit_string)
        placeholder = np.empty(0)
        for bit in bit_string:
            placeholder = np.concatenate((placeholder, create_bit_waveform(bit, samples_per_bit, duty)))

        matrix[i, :] = np.concatenate((header, placeholder))
        # Z-score:
        matrix[i, :] -= np.mean(matrix[i, :])
        matrix[i, :] /= np.std(matrix[i, :])

    return matrix


def perform_correlation(input_data, matrix):
    """
    Correlate signals.
    :param input_data: input data
    :param matrix: matrix used to detect sequence
    :return: correlation coefficients
    """
    window = matrix.shape[1]
    correlation_result = np.zeros((matrix.shape[0], len(input_data) - window + 1))
    for displacement in range(correlation_result.shape[1]):
        aux = input_data[displacement:displacement + window]
        correlation_result[:, displacement] = np.matmul(matrix, (aux - np.mean(aux)) / np.std(aux))

    return correlation_result / window


def tuple_to_array(input_data):
    """
    Convert data from tuple to array.
    :param input_data: data as a tuple
    :return: data as a numpy array
    """
    output_data = np.zeros((len(input_data), input_data[0].shape[0]))
    for index, signature in enumerate(input_data):
        output_data[index, :] = signature

    return output_data


def execute(band):
    """
    Perform correlation and calculate BER.
    :param band: perform correlation for each band.
    :return: BER
    """
    # Load data sent
    list_filename = 'captures/list.npy'
    data = np.load(list_filename, allow_pickle=True)

    # Parameters: 50 fps , Tb = 200 ms
    detector_matrix = create_signals(bits=8, samples_per_bit=10, header_length=5, duty=0.8)
    bits_err = 0
    for i in range(len(data)):
        captures = np.load("captures/" + str(i) + ".npy", allow_pickle=True)
        captures = tuple_to_array(captures)

        correlations = perform_correlation(captures[:, band], detector_matrix)
        max_corr = np.max(np.max(correlations))
        result = np.argwhere(correlations == max_corr)
        data_bin = format(data[i], '08b')
        # print("Data sent:", int(data_bin))
        result_bin = format(result[0, 0], '08b')
        # print("Data received:", int(result_bin))
        for j in range(8):
            if int(data_bin[j], 2) ^ int(result_bin[j], 2):  # XOR to find errors
                bits_err += 1

        # err = ord(data[i]) - result[0, 0]  # if char are sent
        # print(format(data[i], '08b'))
        # print(format(result[0, 0], '08b'))

        # print(bits_err)
        # data_bin ^ result_bin
        # err = data[i] - result[0, 0]  # if int are sent
        # print(err)
    print("BER:", bits_err / len(data) / 8)


def execute_2_txs(channel_matrix):
    """
    Perform correlation and calculate BER.
    :param channel_matrix: matrix comprising the spectral signatures involved in the transmission.
    :return: BER
    """
    # Load data sent
    list_filename_1 = 'captures/list_1.npy'
    list_filename_2 = 'captures/list_2.npy'
    data_1 = np.load(list_filename_1, allow_pickle=True)
    data_2 = np.load(list_filename_2, allow_pickle=True)

    # Parameters: 50 fps , Tb = 200 ms
    detector_matrix_1 = create_signals(bits=8, samples_per_bit=10, header_length=5, duty=0.25)
    detector_matrix_2 = create_signals(bits=8, samples_per_bit=10, header_length=5, duty=0.8)
    bits_err_1 = 0
    bits_err_2 = 0

    # channel_matrix = np.zeros((2, 9))

    for i in range(len(data_1)):
        captures = np.load("captures/" + str(i) + ".npy", allow_pickle=True)
        captures = tuple_to_array(captures)

        ####### PLOT #######
        # ax1 = plt.subplot(412)
        # ax1.plot(captures[50, :])
        # ax1.set_title('Combined signature')
        # ax2 = plt.subplot(421)
        # ax2.plot(channel_matrix[0, :])
        # ax2.set_title('Signature 1')
        # ax3 = plt.subplot(422)
        # ax3.plot(channel_matrix[1, :])
        # ax3.set_title('Signature 2')
        # ax4 = plt.subplot(413)
        # ax4.plot(captures[:, :])
        # ax4.set_title('Signals')
        # ax5 = plt.subplot(414)
        # ax5.plot(np.matmul(captures[:, :], np.linalg.pinv(channel_matrix)[:, :]))
        # ax5.set_title('Signals after compensation')
        # plt.show()
        ############################

        ######## Dynamic channel matrix  ########
        # Load signatures
        signatures_1 = np.load("captures/signatures1_" + str(i) + ".npy", allow_pickle=True)
        signatures_1 = tuple_to_array(signatures_1)
        signatures_2 = np.load("captures/signatures2_" + str(i) + ".npy", allow_pickle=True)
        signatures_2 = tuple_to_array(signatures_2)
        # Select a sample where the LED is ON
        id_1 = np.argwhere(signatures_1 == np.amax(signatures_1))
        id_2 = np.argwhere(signatures_2 == np.amax(signatures_2))
        # Add normalized signature to the channel matrix
        channel_matrix[0, :] = signatures_1[id_1[0][0], :] / np.max(signatures_1[id_1[0][0], :])
        channel_matrix[1, :] = signatures_2[id_2[0][0], :] / np.max(signatures_2[id_2[0][0], :])
        ########################################

        compensation = np.matmul(captures[:, :], np.linalg.pinv(channel_matrix)[:, :])  # ZF equalization

        correlations_1 = perform_correlation(compensation[:, 0], detector_matrix_1)
        correlations_2 = perform_correlation(compensation[:, 1], detector_matrix_2)
        max_corr_1 = np.max(np.max(correlations_1))
        result_1 = np.argwhere(correlations_1 == max_corr_1)
        max_corr_2 = np.max(np.max(correlations_2))
        result_2 = np.argwhere(correlations_2 == max_corr_2)
        data_bin_1 = format(data_1[i], '08b')
        data_bin_2 = format(data_2[i], '08b')
        # print("Data sent LED 1:", int(data_bin_1))
        # print("Data sent LED 2:", int(data_bin_2))
        result_bin_1 = format(result_1[0, 0], '08b')
        result_bin_2 = format(result_2[0, 0], '08b')
        # print("Data received LED 1:", int(result_bin_1))
        # print("Data received LED 2:", int(result_bin_2))
        for j in range(8):
            if int(data_bin_1[j], 2) ^ int(result_bin_1[j], 2):  # XOR to find errors
                bits_err_1 += 1
            if int(data_bin_2[j], 2) ^ int(result_bin_2[j], 2):  # XOR to find errors
                bits_err_2 += 1
    print("BER LED 1:", bits_err_1 / len(data_1) / 8)
    print("BER LED 2:", bits_err_2 / len(data_2) / 8)


def visualize(num):
    """
    Plot the waveform of received signals.
    :param num: number of characters to be plotted
    :return: None
    """
    for i in range(num):
        captures = np.load("captures/" + str(i) + ".npy", allow_pickle=True)
        captures = tuple_to_array(captures)
        plt.figure(1)
        plt.plot(captures)
        plt.show()


def view_waveforms(num):
    """
    Plot the waveform of each band of the received signals.
    :param num: number of characters to be plotted
    :return: None
    """
    for k in range(num):
        captures = np.load("captures/" + str(k) + ".npy", allow_pickle=True)
        captures = tuple_to_array(captures)
        plt.figure(1)
        pos = 1
        for i in range(3):
            for j in range(3):
                plt.subplot(3, 3, pos)
                plt.plot(captures[:, pos - 1])
                pos += 1

        plt.show()


if __name__ == "__main__":
    # TRANSMISSION USING 1 LED
    # for b in range(9):
    #     execute(b)
    # view_waveforms()
    # visualize(10)

    # TRANSMISSION USING 2 LED
    # signature_1 = np.load("captures/green_signature.npy", allow_pickle=True)
    # signature_2 = np.load("captures/blue_signature.npy", allow_pickle=True)
    signature_1 = np.load("captures/signature1.npy", allow_pickle=True)
    signature_2 = np.load("captures/signature2.npy", allow_pickle=True)
    signature_1 = tuple_to_array(signature_1)
    signature_2 = tuple_to_array(signature_2)

    # Find index of maximum value from the signature array to select a sample where the LED is ON
    max_value_id_1 = np.where(signature_1 == np.amax(signature_1))
    max_value_id_2 = np.where(signature_2 == np.amax(signature_2))

    # Zip the 2 arrays to get the exact coordinates
    list_of_coordinates_1 = list(zip(max_value_id_1[0], max_value_id_1[1]))
    list_of_coordinates_2 = list(zip(max_value_id_2[0], max_value_id_2[1]))

    # Select the sample where the maximum value was found
    signature_1 = signature_1[list_of_coordinates_1[0][0], :]
    # signature_1 = signature_1 / np.max(signature_1)  # normalization
    signature_2 = signature_2[list_of_coordinates_2[0][0], :]
    # signature_2 = signature_2 / np.max(signature_2)  # normalization
    h = np.zeros((2, 9))
    h[0, :] = signature_1
    h[1, :] = signature_2

    # plt.plot(signature_1)
    plt.plot(h[0, :])
    plt.show()
    # plt.plot(signature_2)
    plt.plot(h[1, :])
    plt.show()
    execute_2_txs(channel_matrix=h)

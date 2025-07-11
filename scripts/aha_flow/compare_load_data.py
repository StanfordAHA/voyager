import numpy as np


def bfbin2float(bfstr):
    sign = bfstr[0]
    exp = bfstr[1:9]
    lfrac = bfstr[9:16]
    if sign == "0" and exp == "11111111" and lfrac != "0000000":
        return float('nan')
    elif sign == "1" and exp == "11111111" and lfrac != "0000000":
        return -float('nan')
    elif sign == "0" and exp == "11111111" and lfrac == "0000000":
        return float('inf')
    elif sign == "1" and exp == "11111111" and lfrac == "0000000":
        return -float('inf')
    elif sign == "0" and exp == "00000000" and lfrac == "0000000":
        return float(0)
    elif sign == "1" and exp == "00000000" and lfrac == "0000000":
        return -float(0)
    else:
        mult = 1
        if sign == "1":
            mult = -1
        nexp = int(exp, 2) - 127
        if exp != 0:
            lfrac = "1" + lfrac
        else:
            lfrac = "0" + lfrac
        nfrac = int(lfrac, 2)
        return mult * nfrac * (2 ** (nexp - 7))

def read_bytes_from_waveform(file_path):
    all_bytes = []

    with open(file_path, "r") as file:
        for line in file:
            if "=" in line:
                hex_data = line.split("=")[-1].strip().replace(" ", "")
                # Break into bytes (2 hex chars), LSB first
                bytes_list = [hex_data[i:i+2] for i in range(0, len(hex_data), 2)]
                bytes_list.reverse()  # LSB first
                all_bytes.extend(bytes_list)

    return all_bytes


def read_bytes_from_systemC(file_path):
    all_bytes = []

    with open(file_path, "r") as file:
        for line in file:
            cleaned = line.strip().replace(" ", "")
            if cleaned:  # skip empty lines
                all_bytes.append(cleaned)

    return all_bytes

def read_bytes_from_hw_output_txt(file_path):
    float_array = []

    with open(file_path, "r") as file:
        hex_array = []

        for line in file:
            if line.strip():  # Check if the line is not empty
                values = [int(value, 16) for value in line.split()]
                hex_array.extend(values)

    # float_array = np.array(all_bytes, dtype=np.float32)
    float_array = np.array([bfbin2float(bin(x)[2:].zfill(16)) for x in hex_array], dtype=np.float32)
    return float_array


# Compare the two lists
def compare_data(data1, data2, name):
    if len(data1) != len(data2):
        print(f"Length mismatch in {name}: {len(data1)} vs {len(data2)}")
        # return False
    for i in range(len(data1)):
        if data1[i] != data2[i]:
            print(f"Mismatch in {name} at index {i}: {data1[i]} vs {data2[i]}")
            # return False
    print(f"{name} match!")
    # return True



def read_SA_output_from_systemC(input_txt_path):
    float_list = []

    with open(input_txt_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines just in case
                floats = [np.float32(x) for x in line.strip().split()]
                float_list.extend(floats)

    float_array = np.array(float_list, dtype=np.float32)

    return float_array


def compare_floating_point_data(data1, data2, name):
    if len(data1) != len(data2):
        print(f"Length mismatch in {name}: {len(data1)} vs {len(data2)}")

    for i in range(len(data1)):
        if not np.isclose(data1[i], data2[i], rtol=1e-05, atol=1e-08):
            print(f"Mismatch in {name} at index {i}: {data1[i]} vs {data2[i]}")

    print(f"{name} match!")



if __name__ == "__main__":
    # input_data_mu_in = read_bytes_from_waveform("/aha/garnet/tests/test_app/input_data_mu_in.txt")
    # weight_data_mu_in = read_bytes_from_waveform("/aha/garnet/tests/test_app/weight_data_mu_in.txt")
    # bias_data_mu_in = read_bytes_from_waveform("/aha/garnet/tests/test_app/bias_data_mu_in.txt")
    # inputScale_data_mu_in = read_bytes_from_waveform("/aha/garnet/tests/test_app/inputScale_data_mu_in.txt")
    # weightScale_data_mu_in = read_bytes_from_waveform("/aha/garnet/tests/test_app/weightScale_data_mu_in.txt")

    # # systolic_array_data_out = read_bytes_from_waveform("/aha/garnet/tests/test_app/systolic_array_output.txt")
    # # print(len(systolic_array_data_out))


    # input_data_systemC = read_bytes_from_systemC("/aha/voyager/compiled_collateral/resnet18-submodule_6/compare/input_data_systemC.txt")
    # weight_data_systemC = read_bytes_from_systemC("/aha/voyager/compiled_collateral/resnet18-submodule_6/compare/weight_data_systemC.txt")
    # bias_data_systemC = read_bytes_from_systemC("/aha/voyager/compiled_collateral/resnet18-submodule_6/compare/bias_data_systemC.txt")
    # inputScale_data_systemC = read_bytes_from_systemC("/aha/voyager/compiled_collateral/resnet18-submodule_6/compare/inputScale_data_systemC.txt")
    # weightScale_data_systemC = read_bytes_from_systemC("/aha/voyager/compiled_collateral/resnet18-submodule_6/compare/weightScale_data_systemC.txt")

    SA_output_systemC = read_SA_output_from_systemC("/aha/SA_output_systemC.txt")



    glb_hw_output = read_bytes_from_hw_output_txt("/aha/garnet/tests/test_app/hw_output.txt")

    # Reshape it to (Y1, Y0, X1, X0, K2, K1, K0=32)
    glb_hw_output = glb_hw_output.reshape((1, 14, 1, 14, 8, 1, 32))
    glb_hw_output = glb_hw_output.transpose(0, 2, 4, 5, 1, 3, 6)  # (Y1, X1, K2, K1, Y0, X0, K0=32)
    glb_hw_output = glb_hw_output.flatten()  # Flatten to 1D array


    # compare_data(input_data_mu_in, input_data_systemC, "input_data")
    # compare_data(weight_data_mu_in, weight_data_systemC, "weight_data")
    # compare_data(bias_data_mu_in, bias_data_systemC, "bias_data")
    # compare_data(inputScale_data_mu_in, inputScale_data_systemC, "inputScale_data")
    # compare_data(weightScale_data_mu_in, weightScale_data_systemC, "weightScale_data")


    compare_floating_point_data(SA_output_systemC, glb_hw_output, "systolic_array_output_float")






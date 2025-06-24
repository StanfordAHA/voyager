def read_format1(filename):
    with open(filename, 'r') as f:
        # Skip the first two lines, then process the rest
        lines = f.readlines()[2:]
        return [line.strip().lower() for line in lines if line.strip()]

def read_format2(filename):
    with open(filename, 'r') as f:
        # Read lines, strip whitespace, and flatten to a list of bytes
        byte_lines = [line.strip().lower() for line in f if line.strip()]
        if len(byte_lines) % 2 != 0:
            raise ValueError("Format 2 file has an odd number of bytes.")
        # Combine every two bytes into a word in little-endian order
        words = []
        prev_word = ""
        for i in range(0, len(byte_lines), 2):
            lsb = byte_lines[i]
            msb = byte_lines[i + 1]
            word = msb + lsb  # little endian: MSB comes after LSB
            # if prev_word == "3e99" and word == "3f62":
            #      breakpoint()
            prev_word = word
            words.append(word)
        return words

def compare_formats(format1_file, format2_file):
    words1 = read_format1(format1_file)
    words2 = read_format2(format2_file)

    if words1 == words2:
        print("Files match perfectly!")
    else:
        print("Files do not match. Differences:")
        for i, (w1, w2) in enumerate(zip(words1, words2)):
            if w1 != w2:
                print(f"Line {i + 1}: Format1 = {w1}, Format2 = {w2}")
        if len(words1) != len(words2):
            print(f"Length mismatch: Format1 has {len(words1)} words, Format2 has {len(words2)} words")


# Example usage:
if __name__ == "__main__":
    compare_formats('compiled_collateral/resnet18-submodule_2/tensor_files/residual_hex.txt', 'compiled_collateral/resnet18-submodule_2/compare/vectorFetch1_data_systemC.txt')

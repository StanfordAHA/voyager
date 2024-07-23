import struct
import numpy as np


''' @brief: Writes data of form torch.tensor dtype=float64 to binary data. '''
def write_fp64(filename, data):
    data = data.astype(np.float64)
    with open(filename, 'wb') as f:
        floatlist = []
        
        for i in np.nditer(data):
            floatlist.append(i)

        buf = struct.pack('%sd' % len(floatlist), *floatlist)
        f.write(buf)

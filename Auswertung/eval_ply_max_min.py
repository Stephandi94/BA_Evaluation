import os
import numpy as np
from plyfile import PlyData

def main():
    src = os.path.join(r"c:/Users/steph/Downloads/comp.ply")
    ply = PlyData.read(src)
    x_data = np.asarray(ply.elements[0].data['x'])
    y_data = np.asarray(ply.elements[0].data['y'])
    z_data = np.asarray(ply.elements[0].data['z'])
    i_data = np.asarray(ply.elements[0].data['I']) * 100

    numpy_pc = np.ndarray((x_data.size, 4), dtype=float)

    numpy_pc[:, 0] = x_data
    numpy_pc[:, 1] = y_data
    numpy_pc[:, 2] = z_data
    numpy_pc[:, 3] = i_data

    print(np.min(i_data))
    print(np.max(i_data))

if __name__=='__main__':
    main()
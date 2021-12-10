import pickle
import numpy as np
import glob

with open('lane_polynomials.obj', 'rb') as file:
    d = pickle.load(file)
    print(d)
    for i in d:
        y = d[i]['y']
        c = d[i]['coefficients']
        x = [c[0] * a ** 3 + c[1] * a ** 2 + c[2] * a + c[3] for a in y]
        p = np.array([x, y])
        filename = 'lane_polynomials_exact/poly_points_' + i[-8:-4] + '.csv'
        np.savetxt(filename, p, delimiter=",")
    file.close()

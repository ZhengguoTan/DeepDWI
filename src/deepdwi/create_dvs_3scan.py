import argparse
import h5py

import numpy as np

# %%
parser = argparse.ArgumentParser(description='create dvs h5 file for 3-scan trace.')

parser.add_argument('--bvals', nargs='+', type=int,
                    default=[50, 100, 200, 300, 400, 500])

parser.add_argument('--bavgs', nargs='+', type=int,
                    default=[2, 2, 2, 2, 2, 2])

args = parser.parse_args()

# %%
trace_array = [[1, 1, -0.5],
               [1, -0.5, 1],
               [-0.5, 1, 1]]

# trace_array = [[1, 0, 0],
#                [0, 1, 0],
#                [0, 0, 1]]

b0_array = np.array([0, 0, 0]).reshape((1, 3))


list_bvals = args.bvals  # [0, 100, 800, 1600]
list_bavgs = args.bavgs  # [1, 2, 4, 8]
list_bvecs = [trace_array, trace_array, trace_array]

bvals_repet = [0] * len(list_bvals)

skip_flag = [False] * len(list_bvals)

o_bvals = np.array([]).reshape((0, 1))
o_bvecs = np.array([]).reshape((0, 3))

break_flag = True

while break_flag:

    for n in range(len(list_bvals)):

        skip_flag[n] = True if bvals_repet[n] >= list_bavgs[n] else False

        if skip_flag[n] is False:

            bval = list_bvals[n]
            bvec = b0_array if bval == 0 else trace_array

            o_bvals = np.append(o_bvals, np.array([bval] * len(bvec)).reshape(len(bvec), 1), axis=0)
            o_bvecs = np.append(o_bvecs, bvec, axis=0)

            bvals_repet[n] += 1

        if all(skip_flag):
            break_flag = False
            break


print('> bvals shape: ', o_bvals.shape)
print('  ', o_bvals)
print('> bvecs shape: ', o_bvecs.shape)
print('  ', o_bvecs)

str_bvals = 'bval'
str_bavgs = 'bavg'
for n in range(len(list_bvals)):
    str_bvals += '-' + str(list_bvals[n]).zfill(3)
    str_bavgs += '-' + str(list_bavgs[n]).zfill(2)
print(str_bvals)
print(str_bavgs)

with h5py.File('dvs_3scan_' + str_bvals + '_' + str_bavgs + '.h5', 'w') as f:
    f.create_dataset('bvals', data=o_bvals)
    f.create_dataset('bvecs', data=o_bvecs)

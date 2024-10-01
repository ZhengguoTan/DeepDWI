import argparse
import os

CURR_DIR = os.getcwd()
DATA_DIR = DIR = os.path.dirname(os.path.realpath(__file__))

# %%
parser = argparse.ArgumentParser(description='load NAViEPI data from Zenodo.')

parser.add_argument('--records', default='10474402', type=str,
                    help='doi records number on zenodo')

# here multiple files should be separated with space
parser.add_argument('--file', default=None, type=str,
                    help='name of the file to be downloaded. Note: \
                        multiple files should be separated with space. \
                            Default is None, i.e. download all files.')

args = parser.parse_args()

# %% download data
if bool(args.file and not args.file.isspace()):  # not blank
    print('> user provided file: ', args.file)
    files_list = args.file.split()
else:
    print('> download all data (slow!)')

# download
for f in files_list:

    if os.path.exists(DATA_DIR + '/' + f):
        print(f'The file {f} exists.')
    else:
        os.system('wget -P ' + DATA_DIR + ' -q https://zenodo.org/records/' + args.records + '/files/' + f)

# check
os.chdir(DATA_DIR)

for f in files_list:
    os.system('cat md5sum.txt | grep ' + f + ' | md5sum -c --ignore-missing')

os.chdir(CURR_DIR)
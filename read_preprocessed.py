import numpy as np

def display_head_npy(npyfile):

    arr = np.load(npyfile)

    np.set_printoptions(edgeitems=10,linewidth=10000, precision=3)
    print(arr)

def display_head_txt(txt_file):
    arr = np.loadtxt(txt_file, dtype=np.int32)

    np.set_printoptions(edgeitems=10,linewidth=10000, precision=3)
    print(arr)


def main():
    npy_file = "/home/wcherlet/InstSeg/Mask3D/data/processed/s3dis/Area_1/conferenceRoom_1.npy"

    txt_file = "/home/wcherlet/InstSeg/Mask3D/data/processed/trees/instance_gt/train/Tile_1.txt"

    # display_head_npy(npy_file)

    display_head_txt(txt_file)

if __name__ == "__main__":
    main()
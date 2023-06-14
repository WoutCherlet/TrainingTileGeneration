import numpy as np

def display_head_npy(npyfile):

    arr = np.load(npyfile)

    np.set_printoptions(edgeitems=10,linewidth=10000, precision=3)
    print(arr)



def main():
    npy_file = "/home/wcherlet/InstSeg/Mask3D/data/processed/s3dis/Area_1/conferenceRoom_1.npy"

    display_head_npy(npy_file)

if __name__ == "__main__":
    main()
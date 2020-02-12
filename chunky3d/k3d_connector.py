import k3d
import numpy as np
from time import sleep


def get_k3d_obj(sparse):
    return k3d.voxels_group(np.array(sparse.shape, dtype=np.uint32)[::-1],
                            voxels_group=sparse.get_k3d_voxels_group_dict(),
                            bounds=sparse.get_voxels_bounds(), outlines=True,
                            color_map=k3d.nice_colors[0:-2] * 50)


class k3d_connector:
    k3d_voxels_chunk = None
    sparse = None

    def __init__(self, sparse):
        self.sparse = sparse
        self.k3d_voxels_chunk = dict()

        for coord in sparse.kchunks_initialized:
            self.create_k3d_chunk(coord)

    def get_ends(self, coord):
        global_coord = self.sparse._chunk_to_global_coord(coord, (0, 0, 0))
        end = np.array(self.sparse.shape) - np.array(global_coord)
        end = np.min([end, np.array(self.sparse._chunk_shape)], axis=0)

        return end

    def create_k3d_chunk(self, coord):
        end = self.get_ends(coord)
        chunk = k3d.voxel_chunk(
            voxels=self.sparse.get_chunk(coord)[:end[0], :end[1], :end[2]],
            coord=(np.array(coord, dtype=np.uint32) * np.array(self.sparse._chunk_shape, dtype=np.uint32))[::-1],
            multiple=1)

        self.k3d_voxels_chunk[coord] = chunk

    def get_ids(self):
        return [g.id for g in self.k3d_voxels_chunk.values()]

    def sync(self, k3d_object):
        k3d_object._hold_remeshing = True
        # to compare
        for coord in self.sparse.kchunks_initialized.intersection(self.k3d_voxels_chunk.keys()):
            end = self.get_ends(coord)
            data = np.array(self.sparse.get_chunk(coord)[:end[0], :end[1], :end[2]])

            if np.any(self.k3d_voxels_chunk[coord].voxels != data):
                self.k3d_voxels_chunk[coord].voxels = data

        # new to add
        for coord in self.sparse.kchunks_initialized.difference(self.k3d_voxels_chunk.keys()):
            self.create_k3d_chunk(coord)

        # new to delete
        for coord in set(self.k3d_voxels_chunk.keys()).difference(self.sparse.kchunks_initialized):
            del self.k3d_voxels_chunk[coord]

        k3d_object.chunks_ids = self.get_ids()
        sleep(2) # TODO: fixme race condition
        k3d_object._hold_remeshing = False

import sys
import numpy as np
import trimesh
from sdf import SDF

teapot = trimesh.load(sys.argv[1], use_embree=(sys.argv[2] == 'True'))
teapot_sdf = SDF(np.array(teapot.vertices), np.array(teapot.faces))
print('SDF:', teapot_sdf)
print('Trimesh:', teapot)
print('Trimesh ray:', teapot.ray)
print('SFD threads:', SDF.num_threads)

rand_points = np.random.randn(10000, 3)
import time
start_time = time.time()
cont = teapot_sdf.contains(rand_points)
end_time = time.time()
print('SDF:', end_time - start_time)

start_time = time.time()
cont_tm = teapot.contains(rand_points)
end_time = time.time()
print('Trimesh:', end_time - start_time)

print(len((cont != cont_tm).nonzero()[0]), 'disagreements')

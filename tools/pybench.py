import sys
import numpy as np
import trimesh
from pysdf import SDF

teapot = trimesh.load(sys.argv[1], use_embree=(sys.argv[2] == 'True'))
teapot_sdf = SDF(np.array(teapot.vertices), np.array(teapot.faces))
print('SDF:', teapot_sdf)
print('Trimesh:', teapot)
print('Trimesh ray:', teapot.ray)
print('SFD threads:', SDF.num_threads)

NUM_POINTS = 1000
rand_points = np.random.randn(NUM_POINTS, 3)
import time
start_time = time.time()
cont = teapot_sdf.contains(rand_points)
end_time = time.time()
print('SDF.contains:', end_time - start_time)

start_time = time.time()
cont_tm = teapot.contains(rand_points)
end_time = time.time()
print('Trimesh.contains:', end_time - start_time)

start_time = time.time()
sdf = teapot_sdf(rand_points)
end_time = time.time()
print('SDF:', end_time - start_time)

start_time = time.time()
sdf_tm = trimesh.proximity.signed_distance(teapot, rand_points)
end_time = time.time()
print('Trimesh SDF:', end_time - start_time)

print(sdf[:10])
print(sdf_tm[:10])

print(len((cont != cont_tm).nonzero()[0]), 'contain disagreements')
print(np.sum(np.abs(sdf - sdf_tm)) / NUM_POINTS, 'average SDF difference')

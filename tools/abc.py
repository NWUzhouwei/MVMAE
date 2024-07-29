import open3d as o3d

# 加载点云
pcd = o3d.io.read_point_cloud("/home/remote/DataSet/ShapeNet55/shapenet_pc/02691156-1a04e3eab45ca15dd86060f189eb133.npy")
# Alpha shapes 
alpha = 0.1
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
# 可视化重建结果
o3d.visualization.draw_geometries([mesh], window_name="点云重建",
                                  width=800, 
                                  height=600,
                                  mesh_show_back_face=True)  
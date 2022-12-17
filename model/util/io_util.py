import numpy as np

def save_volume(volume, fname,  dim_h=192, dim_w=128, voxel_size=1/192):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = volume.shape[0], volume.shape[1], volume.shape[2]
    with open(fname, 'w') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if volume[xx, yy, zz] > 0:
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))
def save_v_volume(v_volume, fname, dim_h=192, dim_w=128, voxel_size=1/192):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = v_volume.shape[0], v_volume.shape[1], v_volume.shape[2]
    with open(fname, 'w') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if (v_volume[xx, yy, zz, :] != np.zeros((3,), dtype=np.float32)).any():
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f %f %f %f\n' %
                                 (pt[0], pt[1], pt[2], v_volume[xx, yy, zz, 0], v_volume[xx, yy, zz, 1], v_volume[xx, yy, zz, 2]))
def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.

    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def save_samples_rgb(fname, points, rgb):
    '''
    Save the visualization of samplings' RGB-colors to a ply file.

    :param fname: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    '''
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )
def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    """
    input
        mesh_path: XXX.obj
        verts    : (N, 3) in the mesh-coords.
        faces    : (N, 3), order not switched yet
        colors   : (N, 3), RGB, float 0 ~ 1
    """

    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1])) # switch the order, so that the computed normals later can face outwards from the mesh
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()
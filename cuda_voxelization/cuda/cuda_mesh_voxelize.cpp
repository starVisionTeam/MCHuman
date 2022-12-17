#include <torch/extension.h>
#include <vector>
std::vector<at::Tensor> forward_voxelization_cuda(
    at::Tensor vertices_faces,
    at::Tensor occ_volume,float H_NORMALIZE);
std::vector<at::Tensor> forward_semantic_voxelization_cuda(
    at::Tensor smpl_vertices,
    at::Tensor smpl_vertex_code,
    at::Tensor smpl_faces,
    at::Tensor occ_volume,
    at::Tensor semantic_volume,
    at::Tensor weight_sum_volume,
    float sigma,float H_NORMALIZE);
// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor forward_voxelization(
    at::Tensor vertices_faces,at::Tensor occ_volume,float H_NORMALIZE) {

    CHECK_INPUT(vertices_faces);
    CHECK_INPUT(occ_volume);
    CHECK_INPUT(vertices_faces);
    return forward_voxelization_cuda(
        vertices_faces, occ_volume,H_NORMALIZE)[0];
}
std::vector<at::Tensor> forward_semantic_voxelization(
    at::Tensor smpl_vertices,
    at::Tensor smpl_vertex_code,
    at::Tensor smpl_faces,
    at::Tensor occ_volume,
    at::Tensor semantic_volume,
    at::Tensor weight_sum_volume,
    float sigma,float H_NORMALIZE) {

    CHECK_INPUT(smpl_vertices);
    CHECK_INPUT(smpl_vertex_code);
    CHECK_INPUT(smpl_faces);
    CHECK_INPUT(occ_volume);
    CHECK_INPUT(semantic_volume);
    CHECK_INPUT(weight_sum_volume);
    return forward_semantic_voxelization_cuda(
        smpl_vertices, smpl_vertex_code, smpl_faces,
       occ_volume, semantic_volume, weight_sum_volume, sigma, H_NORMALIZE);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_semantic_voxelization", &forward_semantic_voxelization, "forward_semantic_voxelization (CUDA)");
    m.def("forward_voxelization", &forward_voxelization, "forward_voxelization (CUDA)");
}
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("forward_voxelization", &forward_voxelization, "forward_voxelization (CUDA)");
//}

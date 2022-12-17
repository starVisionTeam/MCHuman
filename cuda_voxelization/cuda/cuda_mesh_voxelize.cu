#include <iostream>
#include <ATen/ATen.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif
#define X 0
#define Y 1
#define Z 2
#define EPSILON_ABS_ZERO 1e-10
#define EPSILON_DIV_ZERO 1e-4
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2];

#define MAX(x, y) x > y? x:y
#define MIN(x, y) x < y? x:y
#define FINDMINMAX(x0,x1,x2,min,max) \
  min = max = x0;   \
  if(x1<min) min=x1;\
  if(x1>max) max=x1;\
  if(x2<min) min=x2;\
  if(x2>max) max=x2;
/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)			   \
	p0 = a*v0[Y] - b*v0[Z];			       	   \
	p2 = a*v2[Y] - b*v2[Z];			       	   \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_X2(a, b, fa, fb)			   \
	p0 = a*v0[Y] - b*v0[Z];			           \
	p1 = a*v1[Y] - b*v1[Z];			       	   \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[Y] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)			   \
	p0 = -a*v0[X] + b*v0[Z];		      	   \
	p2 = -a*v2[X] + b*v2[Z];	       	       	   \
        if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Y1(a, b, fa, fb)			   \
	p0 = -a*v0[X] + b*v0[Z];		      	   \
	p1 = -a*v1[X] + b*v1[Z];	     	       	   \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Z];   \
	if(min>rad || max<-rad) return 0;

/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)			   \
	p1 = a*v1[X] - b*v1[Y];			           \
	p2 = a*v2[X] - b*v2[Y];			       	   \
        if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(min>rad || max<-rad) return 0;

#define AXISTEST_Z0(a, b, fa, fb)			   \
	p0 = a*v0[X] - b*v0[Y];				   \
	p1 = a*v1[X] - b*v1[Y];			           \
        if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;} \
	rad = fa * boxhalfsize[X] + fb * boxhalfsize[Y];   \
	if(min>rad || max<-rad) return 0;
// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

namespace{

/*
    Above-triangle test
    Method: consider the tetrahedron constructed from the triangle and the query point and
    check whether the signed volume of the tetrahedron is positive
*/
template<typename scalar_t>
__device__ bool above_triangle_test(
    const scalar_t *v0, const scalar_t *v1, const scalar_t *v2, const scalar_t *p) {

    const scalar_t x1 = v1[0] - v0[0], y1 = v1[1] - v0[1], z1 = v1[2] - v0[2];
    const scalar_t x2 = v2[0] - v0[0], y2 = v2[1] - v0[1], z2 = v2[2] - v0[2];
    const scalar_t x3 =  p[0] - v0[0], y3 =  p[1] - v0[1], z3 =  p[2] - v0[2];
    return (x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1) >= 0;
}

/*
    In-tetrahedron test
    Method: check whether the query point is "above" the four triangle of the tetrahedron
*/
template<typename scalar_t>
__device__ bool in_tetrahedron_test(const scalar_t *tet, const scalar_t* p) {
    bool flags[4];
    const int tris[3*4] {
        /* root, edge1, edge2 */
        0, 2, 1,
        0, 3, 2,
        0, 1, 3,
        1, 2, 3
    };
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const scalar_t* v0 = tet + 3 * tris[3*k+0];
        const scalar_t* v1 = tet + 3 * tris[3*k+1];
        const scalar_t* v2 = tet + 3 * tris[3*k+2];
        flags[k] = above_triangle_test(v0, v1, v2, p);
    }
    return flags[0] == flags[1] && flags[0] == flags[2] && flags[0] == flags[3];
}
__device__ __forceinline__ int planeBoxOverlap(float normal[3], float vert[3], const float maxbox[3])	// -NJMP-
{
    int q;
    float vmin[3], vmax[3], v;
    for (q = X; q <= Z; q++)
    {
        v = vert[q];					// -NJMP-
        if (normal[q] > 0.0f)
        {
            vmin[q] = -maxbox[q] - v;	// -NJMP-
            vmax[q] = maxbox[q] - v;	// -NJMP-
        }
        else
        {
            vmin[q] = maxbox[q] - v;	// -NJMP-
            vmax[q] = -maxbox[q] - v;	// -NJMP-
        }
    }
    if (DOT(normal, vmin) > 0.0f) return 0;	// -NJMP-
    if (DOT(normal, vmax) >= 0.0f) return 1;	// -NJMP-

    return 0;
}
template<typename scalar_t>
__device__ bool triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3],const scalar_t *triverts) {

    float v0[3], v1[3], v2[3];
    float min, max, p0, p1, p2, rad, fex, fey, fez;		// -NJMP- "d" local variable removed
    float normal[3], e0[3], e1[3], e2[3];
    for(int i=0;i<3;i++)
    {
        v0[i]=triverts[3*0+i]-boxcenter[i];
        v1[i]=triverts[3*1+i]-boxcenter[i];
        v2[i]=triverts[3*2+i]-boxcenter[i];
    }
    SUB(e0, v1, v0);      /* tri edge 0 */
    SUB(e1, v2, v1);      /* tri edge 1 */
    SUB(e2, v0, v2);      /* tri edge 2 */

    /* Bullet 3:  */
    /*  test the 9 tests first (this was faster) */
    fex = fabsf(e0[X]);
    fey = fabsf(e0[Y]);
    fez = fabsf(e0[Z]);
    AXISTEST_X01(e0[Z], e0[Y], fez, fey);
    AXISTEST_Y02(e0[Z], e0[X], fez, fex);
    AXISTEST_Z12(e0[Y], e0[X], fey, fex);

    fex = fabsf(e1[X]);
    fey = fabsf(e1[Y]);
    fez = fabsf(e1[Z]);
    AXISTEST_X01(e1[Z], e1[Y], fez, fey);
    AXISTEST_Y02(e1[Z], e1[X], fez, fex);
    AXISTEST_Z0(e1[Y], e1[X], fey, fex);

    fex = fabsf(e2[X]);
    fey = fabsf(e2[Y]);
    fez = fabsf(e2[Z]);
    AXISTEST_X2(e2[Z], e2[Y], fez, fey);
    AXISTEST_Y1(e2[Z], e2[X], fez, fex);
    AXISTEST_Z12(e2[Y], e2[X], fey, fex);
    FINDMINMAX(v0[X], v1[X], v2[X], min, max);
    if (min > boxhalfsize[X] || max < -boxhalfsize[X]) return 0;

    /* test in Y-direction */
    FINDMINMAX(v0[Y], v1[Y], v2[Y], min, max);
    if (min > boxhalfsize[Y] || max < -boxhalfsize[Y]) return 0;

    /* test in Z-direction */
    FINDMINMAX(v0[Z], v1[Z], v2[Z], min, max);
    if (min > boxhalfsize[Z] || max < -boxhalfsize[Z]) return 0;

    /* Bullet 2: */
    /*  test if the box intersects the plane of the triangle */
    /*  compute plane equation of triangle: normal*x+d=0 */
    CROSS(normal, e0, e1);
    // -NJMP- (line removed here)
    if (!planeBoxOverlap(normal, v0, boxhalfsize)) return 0;	// -NJMP-

    return 1;   /* box and triangle overlaps */
}

/*
    Voxel labeling
*/
__device__ void label_occupied_voxel(float *voxel) {
    atomicMax((int*)voxel, __float_as_int(1.0f));
}

__device__ void label_occupied_voxel(double *voxel) {
    atomicCAS((unsigned long long*)voxel, __double_as_longlong(0), __double_as_longlong(1.0));
}

/*
    Distance Calculator
*/
__device__ float calc_squared_dist(const float *p1, const float *p2) {
    const float x = p1[0] - p2[0];
    const float y = p1[1] - p2[1];
    const float z = p1[2] - p2[2];
    return x*x + y*y + z*z;
}

__device__ double calc_squared_dist(const double *p1, const double *p2) {
    const double x = p1[0] - p2[0];
    const double y = p1[1] - p2[1];
    const double z = p1[2] - p2[2];
    return x*x + y*y + z*z;
}
// __device__ float calc_dist(const float *p1, const float *p2) {
//     return __fsqrt_rn(calc_squared_dist(p1, p2));
// }

// __device__ double calc_dist(const double *p1, const double *p2) {
//     return __dsqrt_rn(calc_squared_dist(p1, p2));
// }

template<typename scalar_t>
__global__ void forward_voxelize_cuda_kernel(
    const scalar_t* __restrict__ vertices_faces,
    scalar_t* __restrict__ out_volume,
    int num_faces,
    float H_NORMALIZE,int volume_res_x,int volume_res_y,int volume_res_z) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) {
        return;
    }
    const scalar_t* tet = &vertices_faces[i * 9];
    scalar_t xmin = tet[0], ymin = tet[1], zmin = tet[2];
    scalar_t xmax = tet[0], ymax = tet[1], zmax = tet[2];
    #pragma unroll
    for (int k = 1; k < 3; k++) {
        xmin = fminf(xmin, tet[3*k + 0]);
        xmax = fmaxf(xmax, tet[3*k + 0]);
        ymin = fminf(ymin, tet[3*k + 1]);
        ymax = fmaxf(ymax, tet[3*k + 1]);
        zmin = fminf(zmin, tet[3*k + 2]);
        zmax = fmaxf(zmax, tet[3*k + 2]);
    }
    float3 _vol_min_corner = make_float3(-H_NORMALIZE/2/volume_res_y*volume_res_x, -H_NORMALIZE/2, -H_NORMALIZE/2/volume_res_y*volume_res_z);
    float3 _vol_max_corner = make_float3(H_NORMALIZE/2/volume_res_y*volume_res_x, H_NORMALIZE/2, H_NORMALIZE/2/volume_res_y*volume_res_z);
    float3 step;
    step.x = (_vol_max_corner.x - _vol_min_corner.x) / volume_res_x;
    step.y = (_vol_max_corner.y - _vol_min_corner.y) / volume_res_y;
    step.z = (_vol_max_corner.z - _vol_min_corner.z) / volume_res_z;
    float boxhalfsize[3] = { step.x / 2.f, step.y / 2.f, step.z / 2.f };
    int3 bb_min_corner, bb_max_corner;
    bb_min_corner.x = MAX(int(floor((xmin - _vol_min_corner.x) / step.x)), 0);
    bb_min_corner.y = MAX(int(floor((ymin - _vol_min_corner.y) / step.y)), 0);
    bb_min_corner.z = MAX(int(floor((zmin - _vol_min_corner.z) / step.z)), 0);
    bb_max_corner.x = MIN(int(ceil((xmax - _vol_min_corner.x) / step.x)), volume_res_x);
    bb_max_corner.y = MIN(int(ceil((ymax - _vol_min_corner.y) / step.y)), volume_res_y);
    bb_max_corner.z = MIN(int(ceil((zmax - _vol_min_corner.z) / step.z)), volume_res_z);
    for (int xx = bb_min_corner.x; xx < bb_max_corner.x; xx++)
        {
            for (int yy = bb_min_corner.y; yy < bb_max_corner.y; yy++)
            {
                for (int zz = bb_min_corner.z; zz < bb_max_corner.z; zz++)
                {
                    float boxcenter_x = xx*step.x + boxhalfsize[0]+ _vol_min_corner.x;
                    float boxcenter_y = yy*step.y + boxhalfsize[1]+ _vol_min_corner.y;
                    float boxcenter_z = zz*step.z + boxhalfsize[2]+ _vol_min_corner.z;
                    float boxcenter[3] = { boxcenter_x, boxcenter_y, boxcenter_z };
                    if (triBoxOverlap(boxcenter, boxhalfsize, tet))
                    {
                        const int i_ = zz*volume_res_y*volume_res_x + yy*volume_res_x + xx;
                        label_occupied_voxel(&(out_volume[i_]));
                    }
                }
            }
        }
}


template<typename scalar_t>
__global__ void our_forward_calc_semantic_volume_cuda_kernel(
    const scalar_t* __restrict__ occ_volume,
    const scalar_t* __restrict__ smpl_vertices,
    const scalar_t* __restrict__ smpl_vertex_code,
    scalar_t* __restrict__ semantic_volume,
    scalar_t* __restrict__ weight_sum_volume,
    float sigma,
    int num_vertex,
    float H_NORMALIZE,int volume_res_x,int volume_res_y,int volume_res_z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= volume_res_x * volume_res_y  * volume_res_z) {
        return;
    }
    if (occ_volume[i] < 1e-3) { // empty voxel
        return;
    }
    const int vn = num_vertex;
    float3 vr_half = make_float3(volume_res_x/2, volume_res_y/2, volume_res_z/2);
    const scalar_t voxel_size = H_NORMALIZE / volume_res_y;
    const int bi = i / (volume_res_x * volume_res_y  * volume_res_z);
    const int vi = i % (volume_res_x * volume_res_y  * volume_res_z);
    const int xv = vi % volume_res_x;
    const int yv = (vi/ volume_res_x) %  volume_res_y;
    const int zv = vi / ( volume_res_x* volume_res_y);
    const scalar_t px = (xv + 0.5 - vr_half.x) * voxel_size;
    const scalar_t py = (yv + 0.5 - vr_half.y) * voxel_size;
    const scalar_t pz = (zv + 0.5 - vr_half.z) * voxel_size;
    const scalar_t pt[3] = {px, py, pz};

    const scalar_t* sv = smpl_vertices + bi * vn * 3;
    const scalar_t* sc = smpl_vertex_code + bi * vn * 3;

    scalar_t weight_sum = 1e-10;
    scalar_t code[3] = {(scalar_t)0};
    for (int k = 0; k < vn; k++) {
        const scalar_t d = calc_squared_dist(pt, sv + k*3);
        const scalar_t w = __expf(-d/(sigma*sigma));
        code[0] += w * sc[k * 3 + 0];
        code[1] += w * sc[k * 3 + 1];
        code[2] += w * sc[k * 3 + 2];
        weight_sum += w;
    }

    semantic_volume[3 * i + 0] = code[0] / weight_sum;
    semantic_volume[3 * i + 1] = code[1] / weight_sum;
    semantic_volume[3 * i + 2] = code[2] / weight_sum;
    weight_sum_volume[i] = weight_sum;
}

}


std::vector<at::Tensor> forward_voxelization_cuda(
    at::Tensor vertices_faces,at::Tensor occ_volume,float H_NORMALIZE) {

    const auto num_faces = vertices_faces.size(0);
    const auto volume_res_x = occ_volume.size(0);
    const auto volume_res_y = occ_volume.size(1);
    const auto volume_res_z = occ_volume.size(2);
    //printf("%d,",volume_res);
    const int threads = 512;
    const dim3 blocks_1 ((num_faces - 1) / threads +1);
    AT_DISPATCH_FLOATING_TYPES(vertices_faces.scalar_type(), "forward_voxelize_cuda_kernel", ([&] {
        forward_voxelize_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
            vertices_faces.data_ptr<scalar_t>(),
            occ_volume.data_ptr<scalar_t>(),
            num_faces,
            H_NORMALIZE,volume_res_x,volume_res_y,volume_res_z);
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in forward_voxelize_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {occ_volume};
}



std::vector<at::Tensor> forward_semantic_voxelization_cuda(
    at::Tensor smpl_vertices,
    at::Tensor smpl_vertex_code,
    at::Tensor smpl_faces,
    at::Tensor occ_volume,
    at::Tensor semantic_volume,
    at::Tensor weight_sum_volume,
    float sigma,float H_NORMALIZE) {

    //const auto batch_size = smpl_vertices.size(0);
    const auto num_vertex = smpl_vertices.size(0);
    const auto num_faces = smpl_faces.size(0);
    const auto volume_res_x = occ_volume.size(0);
    const auto volume_res_y = occ_volume.size(1);
    const auto volume_res_z = occ_volume.size(2);
    //printf("%ld",volume_res_z);
    const int threads = 512;
    const dim3 blocks_2 ((volume_res_x * volume_res_y * volume_res_z - 1) / threads +1);
    AT_DISPATCH_FLOATING_TYPES(smpl_vertices.scalar_type(), "our_forward_calc_semantic_volume_cuda_kernel", ([&] {
        our_forward_calc_semantic_volume_cuda_kernel<scalar_t><<<blocks_2, threads>>>(
            occ_volume.data_ptr<scalar_t>(),
            smpl_vertices.data_ptr<scalar_t>(),
            smpl_vertex_code.data_ptr<scalar_t>(),
            semantic_volume.data_ptr<scalar_t>(),
            weight_sum_volume.data_ptr<scalar_t>(),
            sigma,
            num_vertex,
            H_NORMALIZE,volume_res_x,volume_res_y,volume_res_z);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in forward_calc_semantic_volume_cuda_kernel: %s\n", cudaGetErrorString(err));

    return {semantic_volume, weight_sum_volume};
}

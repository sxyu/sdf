#include <functional>
#include <thread>
#include <random>
#include <Eigen/Core>
#include "sdf/internal/nanoflann.hpp"

namespace sdf {

// Min number of items to allow multithreading
const int MULTITHREAD_MIN_ITEMS = 50;

// Parallel for
void maybe_parallel_for(std::function<void(int&)> loop_content,
                        int loop_max = MULTITHREAD_MIN_ITEMS,
                        int num_threads = std::thread::hardware_concurrency());

// Get a seeded mersenne twister 19937
std::mt19937& get_rng();

}  // namespace sdf

namespace nanoflann {
namespace {
// Static KD-tree adaptor using Eigen::Ref, so we can pass in
// e.g. blocks and Maps without copying
template <class MatrixType, int DIM,
          class Distance = nanoflann::metric_L2_Simple,
          typename IndexType = int>
struct KDTreeEigenRefAdaptor {
    typedef KDTreeEigenRefAdaptor<MatrixType, DIM, Distance> self_t;
    typedef typename MatrixType::Scalar num_t;
    typedef
        typename Distance::template traits<num_t, self_t>::distance_t metric_t;
    typedef KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType> index_t;
    index_t* index = nullptr;
    explicit KDTreeEigenRefAdaptor(Eigen::Ref<const MatrixType> mat,
                                   bool build = true,
                                   const int leaf_max_size = 10)
        : m_data_matrix(mat), leaf_max_size(leaf_max_size) {
        if (build) rebuild();
    }
    ~KDTreeEigenRefAdaptor() {
        if (index) delete index;
    }

    // Rebuild the KD tree from scratch. Call if data updated.
    void rebuild() {
        if (index) delete index;
        index = new index_t(
            DIM, *this,
            nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index->buildIndex();
    }

    Eigen::Ref<const MatrixType> m_data_matrix;
    const int leaf_max_size;
    /// Query for the num_closest closest points to a given point (entered as
    /// query_point[0:dim-1]).
    inline void query(const num_t* query_point, const size_t num_closest,
                      IndexType* out_indices, num_t* out_distances_sq) const {
        nanoflann::KNNResultSet<typename MatrixType::Scalar, IndexType>
            resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }
    /// Query for the closest points to a given point (entered as
    /// query_point[0:dim-1]).
    inline IndexType closest(const num_t* query_point) const {
        IndexType out_indices;
        num_t out_distances_sq;
        query(query_point, 1, &out_indices, &out_distances_sq);
        return out_indices;
    }
    const self_t& derived() const { return *this; }
    self_t& derived() { return *this; }
    inline size_t kdtree_get_point_count() const {
        return m_data_matrix.rows();
    }
    /// Returns the distance between the vector "p1[0:size-1]" and the data
    /// point with index "idx_p2" stored in the class:
    inline num_t kdtree_distance(const num_t* p1, const size_t idx_p2,
                                 size_t size) const {
        num_t s = 0;
        for (size_t i = 0; i < size; i++) {
            const num_t d = p1[i] - m_data_matrix.coeff(idx_p2, i);
            s += d * d;
        }
        return s;
    }
    /// Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, int dim) const {
        return m_data_matrix.coeff(idx, dim);
    }
    /// Optional bounding-box computation: return false to default to a standard
    /// bbox computation loop.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }
};
}  // namespace
}  // namespace nanoflann

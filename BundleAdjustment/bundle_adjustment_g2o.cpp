#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

// 카메라 자세, 내부 파라미터 
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6]; // 초점 거리
        k1 = data_addr[7];    // 외곡 계수 
        k2 = data_addr[8];
    }

    // 현재 구조체의 값을 배열에 넣음
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;

};

/// 카메라 상태 정보를 g2o애 반영할 수 있도록 설정
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; /// Eigen 행렬을 안정하게 쓰기 위한 설정 

    virtual void setToOriginImpl() override { /// vertex를 초기 상태로 리셋하는 함수
        _estimate = PoseAndIntrinsics();
    }

    /// 업데이트된 변화량을 추정값에 반영
    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    /// 3D 점을 현재 카메라 파라미터로 2D 이미지 평면에 투영하는 함수 
    Vector2d project(const Vector3d &point) {
        Vector3d pc = _estimate.rotation * point + _estimate.translation; /// 월드 좌표의 점을 카메라 좌표로 변환
        pc = -pc / pc[2];  /// 정규화 이미지 평면으로 투영
        double r2 = pc.squaredNorm(); /// 왜곡량 계산 
        double distortion = 1.0 + r2 *(_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],   ///최종 2D 좌표 생성 
                        _estimate.focal * distortion * pc[1]);
    }
    
    virtual bool read(istream &in) {}
    
    virtual bool write(ostream &out) const {}
};

/// 3차원 point
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOriginImpl() override{
        _estimate = Vector3d(0, 0 ,0);
    }
    /// 최적화가 계산한 작은 변화량 갱신 
    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

/// 카마라와 3D 점이 실제 이미지에서 어떻게 보이는지 연결하는 제약 
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public :
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0]; // 카메라
        auto v1 = (VertexPoint *) _vertices[1];             // 3D 점
        auto proj = v0 -> project(v1->estimate());          // 카메라로 그 point를 찍었을 때 예측되는 2D 위치 
        _error = proj - _measurement;                       // measurement 실제 관측된 값 
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

void SolveBA(BALProblem &bal_problem);
// argc: 전달된 인자 개수 
// argv: 전달된 인자 문자열 배열 
int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);        // 초기 noise
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

// BAL Data -> g2o 그래프(정점 / 엣지)로 변환 -> 최적화 -> 결과 저장 
void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;  // 카메라 pose 9차원, point 3차원
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    // LM(Levenberg-Marquardt) 알고리즘 사용
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    const double *observations = bal_problem.observations();

    // 저장용 vector 준비
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;

    // 카메라마다 vertex 생성
    // 원본 카메라 데이터를 g2o 카메라 정점으로 변환하는 단계  
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);                  // 그래프에 추가
        vertex_pose_intrinsics.push_back(v);
    }
    // 원본 3D point 데이터를 g2o 정점으로 변환하는 단계 
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }
    // Edge 생성 (관측을 그래프 제약으로 변환)
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);          // 관측에 연결된 카메라      
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);                    // 관측에 연결된 3D 점  
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));   // 실제 관측된 2D 좌표 저장
        edge->setInformation(Matrix2d::Identity());                                         // 가중치 
        edge->setRobustKernel(new g2o::RobustKernelHuber());                                // outlier 영향을 줄임
        optimizer.addEdge(edge);                                                            // 그래프에 추가
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);                   // 최대 40번 반복해서 최적화 수행
    
    // 최적화된 카메라 값을 다시 원본 배열에 저장 
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }

    // 최적화된 3D 점 값 저장 
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];    
    }

}
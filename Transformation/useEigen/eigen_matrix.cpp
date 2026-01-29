#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

#define MATRIX_SIZE 50

int main(int argc, char **argv)
{
    // type, 행, 열. 
    Matrix<float, 2, 3> matrix_23;
    matrix_23 << 1, 2, 3, 4, 5, 6;
    cout << "Matrix 2x3 form 1 to 6: \n" << matrix_23 << "\n";

    cout << "Print Matrix 2x3: " << endl;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cout << matrix_23(i, j) << "\t";
        }
        cout << "\n";
    }
    /* ************************************************ */

    // Vector이지만 실질적으로 double , 3*1 행렬.
    Vector3d v_3d;
    Matrix<float, 3, 1> vd_3d;

    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    // 타입에 맞게 변환 후 연산.
    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    // transpose() 전치 행렬 
    cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

    Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    cout << "[1,2,3;4,5,6]*[4,5,6]=" << result2.transpose() << endl; 
    /* ************************************************ */

    Matrix3d matrix_33 = Matrix3d::Zero();
    matrix_33 = Matrix3d::Random();
    cout << "random matrix: \n" << matrix_33 << "\n";
    cout << "transpose: \n" << matrix_33.transpose() << "\n";  // 전치
    cout << "sum: " << matrix_33.sum() << "\n";                // 각 원소의 합
    cout << "trace: "<< matrix_33.trace() << "\n";             // 대각 원소의 합 
    cout << "times 10: \n" << 10 * matrix_33 << "\n";          // 곱셈 
    cout << "inverse: \n" << matrix_33.inverse() << "\n";      // 역행렬
    cout << "det: " << matrix_33.determinant() << "\n";        // 행렬식 

    // Adjoint (수반 행렬) -> A의 여인수 행렬의 전치행렬을 A의 수반행렬이라 한다.
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    // 얼마나 늘어나는지
    cout << "Eigen values: \n" << eigen_solver.eigenvalues() << "\n";
    // 어느 방향으로 늘어나는지
    cout << "Eigen vectors: \n" << eigen_solver.eigenvectors() << "\n";
    /* ************************************************ */

    // 매트릭스 크기가 확실하지 않은 경우, 동적으로 크기가 조정된 행렬을 사용할 수 있다.
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;
    MatrixXd matrix_x;

    // matrix_NN * x = v_Nd에서 x를 구하는 부분
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose();
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();

    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "Time of normal inverse is " 
         << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << "\n";
    cout << "x = " << x.transpose() << "\n";

    // 행렬 분해 QR 분해 기법 
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "Time of Qr decomposition is " 
         << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << "\n";
    cout << "x = " << x.transpose() << "\n";

    // 정행렬에 대해서 cholesky 분해로 풀 수 있다. 
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "Time of ldlt decomposition is "
         << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << "\n";
    cout << "x = " << x.transpose() << "\n";

    return 0;
}



//
// Created by mi on 23-11-29.
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <mi_utils.h>
int main(){
    using namespace std;
    ifstream fin_txt;
    int i;
    vector<std::vector<double>> x;
    vector<double> min;
    vector<double> max;
    vector<double> psi;
    int K, N;
    int d;
    double mir;
    int dim, edim, tau;
    double s, me;
    double addNoise = -1;  // 噪声幅度；默认1e-8

    /*    if (argc < 7) {
        fprintf(stderr, "\nMutual Infomation (MI) k-nearest neighbours statistics (rectangular)\n\n");
        fprintf(stderr, "Usage:\n%s <filename> <dim> <edim> <tau> <# points> <# neighbours> [addNoise]\n\n", argv[0]);
        fprintf(stderr, "Input:\n\t<filename>\ttext file with <dim> columns and <# points> rows\n");
        fprintf(stderr, "\t<dim>\t\tnumber of columns in file\n");
        fprintf(stderr, "\t<edim>\t\tembedding dimension\n");
        fprintf(stderr, "\t<tau>\t\tembedding delay\n");
        fprintf(stderr, "\t<# points>\tnumber of rows (length of characteristic vector)\n");
        fprintf(stderr, "\t<# neighbours>\tnumber of the nearest neighbours for MI estimator\n");
        fprintf(stderr, "\t[addNoise]\tnoise amplitude; default 1e-8\n");
        fprintf(stderr, "\nOutput:\n");
        fprintf(stderr, "\nMI\n");
        fprintf(stderr, "\nContact: kraskov@its.caltech.edu\n");
        exit(-1);
    }
*/

    dim = 2;   // Nd  列
    N = 128;  //N   行
    edim = 1;  // emb_dim   embxding维度（默认为1，无嵌入）
    tau = 1;  // emb_tau   延时，仅当emb_dim>1时才相关（默认值为1）
    K = 6;  //kneig  MI算法的k最近邻域

    //if (argc == 8) { addNoise = atof(argv[7]); }
    //if (argc >= 9) { fprintf(stderr, "Too many input arguments\n"); exit(-1); }

    x.resize(dim, std::vector<double>(N));
    min.resize(dim, DBL_MAX / 2);
    max.resize(dim, -DBL_MAX / 2);

    //reading of the data
    //打开文件
    fin_txt = ifstream("/home/mi/Code/CPP/covert_to_mex/zwspMIhigh.txt");
    if (fin_txt.is_open()) {
        for (i = 0; i < N; i++) {
            for (d = 0; d < dim; d++) {
                fin_txt >> x[d][i];
            }
        }
    } else {
        cerr << "File " << "zwspMIhigh.txt" << " doesn't exist\n";
        exit(-1);
    }
    fin_txt.close();
    // 添加噪声幅度
    if (addNoise) {
        srand((dim + edim + tau) * N * K * (int)(x[(dim) / 2][N / 10]));
        if (addNoise == -1) {
            for (d = 0; d < dim; d++) {
                for (i = 0; i < N; i++) {
                    x[d][i] += (1.0 * rand() / RAND_MAX) * 1e-8;
                }
            }
        }
        else {
            for (d = 0; d < dim; d++) {
                for (i = 0; i < N; i++) {
                    x[d][i] += (1.0 * rand() / RAND_MAX) * addNoise;
                }
            }
        }
    }

    //规范化
    for (d = 0; d < dim; d++) {
        me = s = 0; for (i = 0; i < N; i++) me += x[d][i];
        me /= N;  for (i = 0; i < N; i++) s += (x[d][i] - me) * (x[d][i] - me);
        s /= (N - 1); s = sqrt(s);
        if (s == 0) {  }
        for (i = 0; i < N; i++) {
            x[d][i] = (x[d][i] - me) / s;
            if (x[d][i] < min[d]) min[d] = x[d][i];
            if (x[d][i] > max[d]) max[d] = x[d][i];
        }
        for (i = 0; i < N; i++) x[d][i] = x[d][i] - min[d];
    }

    psi.resize(N + 1);
    psi[1] = -(double).57721566490153;
    for (i = 1; i < N; i++) psi[i + 1] = psi[i] + 1 / (double)i;

    int rows = static_cast<int>(x.size());
    int cols = static_cast<int>(x[0].size());
    auto** x_ptr = new double* [rows];
    for (i = 0; i < rows; i++) {
        x_ptr[i] = new double[cols];
        for (int j = 0; j < cols; j++) {
            x_ptr[i][j] = x[i][j];
        }
    }
    auto* psi_ptr = new double[psi.size()];
    for (i = 0; i < psi.size(); i++) {
        psi_ptr[i] = psi[i];
    }
    redr_embed(x_ptr, dim, edim, tau, N, K, psi_ptr, &mir);
    for (i = 0; i < rows; i++) {
        delete[] x_ptr[i];
    }
    delete[] x_ptr;
    delete[] psi_ptr;
    fprintf(stdout, "%1.8f\n", mir);

    return 0;
}
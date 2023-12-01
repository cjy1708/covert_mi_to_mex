//
// Created by mi on 23-11-29.
//
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>

#include <mi_utils.h>
#include <mex.hpp>
#include <mexAdapter.hpp>

using matlab::mex::ArgumentList;

class MexFunction : public matlab::mex::Function {
    matlab::data::ArrayFactory factory;
public:
    /**
     * @example MiHigherDim("./zwspMIhigh.txt", 2, 1, 1, 128, 6, 2.0)
     * @param outputs
     * @param inputs 参数顺序：<filename> <dim> <edim> <tau> <# point> <# neighbours> [addNoise]
     */
    void operator()(ArgumentList outputs, ArgumentList inputs) override {
        using namespace std;
        using namespace matlab::data;
        checkArguments(outputs, inputs);
        ifstream fin_txt;

        vector<std::vector<double>> x;
        vector<double> min;
        vector<double> max;
        vector<double> psi;

        string filename;
        int K{6};              // kneig  MI算法的k最近邻域
        int N{128};            // N   行
        double miValue;        // MI值
        int dim{2};            // Nd  列
        int eDim{1};           // emb_dim   embxding维度（默认为1，无嵌入）
        int tau{1};            // emb_tau   延时，仅当emb_dim>1时才相关（默认值为1）
        double s, me;
        double addNoise{1e-8};  // 噪声幅度；默认1e-8

        {
            TypedArray<char16_t> inputStr = inputs[0];
            u16string u16str{inputStr.begin(), inputStr.end()};
            filename = matlab::engine::convertUTF16StringToUTF8String(u16str);
        }
        dim = inputs[1][0];
        eDim = inputs[2][0];
        tau = inputs[3][0];
        if (inputs.size() > 4) {
            N = inputs[4][0];
        }
        if (inputs.size() > 5) {
            K = inputs[5][0];
        }
        if (inputs.size() > 6) {
            addNoise = inputs[6][0];
        }

        x.resize(dim, std::vector<double>(N));
        min.resize(dim, DBL_MAX / 2);
        max.resize(dim, -DBL_MAX / 2);

        //reading of the data
        //打开文件
        fin_txt = ifstream(filename);
        if (fin_txt.is_open()) {
            for (int i = 0; i < N; i++) {
                for (int d = 0; d < dim; d++) {
                    fin_txt >> x[d][i];
                }
            }
        } else {
            cerr << "File " << "zwspMIhigh.txt" << " doesn't exist\n";
            exit(-1);
        }
        fin_txt.close();

        // 添加噪声幅度
        srand((dim + eDim + tau) * N * K * (int) (x[(dim) / 2][N / 10]));
        // 产生随机数
        default_random_engine generator;
        uniform_real_distribution<double> distribution(0.0, 1.0);

        for (int d = 0; d < dim; d++) {
            for (int i = 0; i < N; i++) {
                x[d][i] += distribution(generator) * addNoise;
            }
        }

        //规范化
        for (int d = 0; d < dim; d++) {
            me = s = 0;
            for (int i = 0; i < N; i++) me += x[d][i];
            me /= N;
            for (int i = 0; i < N; i++) s += (x[d][i] - me) * (x[d][i] - me);
            s /= (N - 1);
            s = sqrt(s);
            if (s == 0) {}
            for (int i = 0; i < N; i++) {
                x[d][i] = (x[d][i] - me) / s;
                if (x[d][i] < min[d]) min[d] = x[d][i];
                if (x[d][i] > max[d]) max[d] = x[d][i];
            }
            for (int i = 0; i < N; i++) x[d][i] = x[d][i] - min[d];
        }

        psi.resize(N + 1);
        psi[1] = -(double) .57721566490153;
        for (int i = 1; i < N; i++) psi[i + 1] = psi[i] + 1 / (double) i;

        int rows = static_cast<int>(x.size());
        int cols = static_cast<int>(x[0].size());
        auto **x_ptr = new double *[rows];
        for (int i = 0; i < rows; i++) {
            x_ptr[i] = new double[cols];
            for (int j = 0; j < cols; j++) {
                x_ptr[i][j] = x[i][j];
            }
        }
        auto *psi_ptr = new double[psi.size()];
        for (int i = 0; i < psi.size(); i++) {
            psi_ptr[i] = psi[i];
        }
        redr_embed(x_ptr, dim, eDim, tau, N, K, psi_ptr, &miValue);
        for (int i = 0; i < rows; i++) {
            delete[] x_ptr[i];
        }
        delete[] x_ptr;
        delete[] psi_ptr;

        TypedArray<double> output = factory.createScalar(miValue);
        outputs[0] = output;
    }

private:
    void checkArguments(matlab::mex::ArgumentList &outputs, matlab::mex::ArgumentList &inputs) {
        using namespace matlab::data;
        auto matlabPtr = getEngine();

        if (inputs.size() < 5 || inputs.size() > 7) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>({factory.createScalar(
                                     "Usage: mi_higher_dim <filename> <dim> <edim> <tau> <# point> <# neighbours> [addNoise]")}));
        }

        if (inputs[0].getType() != ArrayType::CHAR) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>(
                                     {factory.createScalar("The first input argument {filename} must be a string.")}));
        }

        if (inputs[1].getType() != ArrayType::INT8 && inputs[1].getType() != ArrayType::INT16
            && inputs[1].getType() != ArrayType::INT32 && inputs[1].getType() != ArrayType::INT64) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>(
                                     {factory.createScalar("The second input argument {dim} must be a Int.")}));
        }

        if (inputs[2].getType() != ArrayType::INT8 && inputs[2].getType() != ArrayType::INT16
            && inputs[2].getType() != ArrayType::INT32 && inputs[2].getType() != ArrayType::INT64) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>(
                                     {factory.createScalar("The third input argument {edim} must be a Int.")}));
        }

        if (inputs[3].getType() != ArrayType::INT8 && inputs[3].getType() != ArrayType::INT16
            && inputs[3].getType() != ArrayType::INT32 && inputs[3].getType() != ArrayType::INT64) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>(
                                     {factory.createScalar("The fourth input argument {tau} must be a Int.")}));
        }

        if (inputs.size() > 4 & inputs[4].getType() != ArrayType::INT8 && inputs[4].getType() != ArrayType::INT16
            && inputs[4].getType() != ArrayType::INT32 && inputs[4].getType() != ArrayType::INT64) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>(
                                     {factory.createScalar("The fifth input argument {point} must be a Int.")}));
        }

        if (inputs.size() > 5 && inputs[5].getType() != ArrayType::INT8 && inputs[5].getType() != ArrayType::INT16
            && inputs[5].getType() != ArrayType::INT32 && inputs[5].getType() != ArrayType::INT64) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>(
                                     {factory.createScalar("The sixth input argument {neighbours} must be a Int.")}));
        }

        if (inputs.size() > 6 && inputs[6].getType() != ArrayType::DOUBLE) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>(
                                     {factory.createScalar("The sixth input argument {addNoise} must be a Double.")}));
        }

        if (outputs.size() > 1) {
            matlabPtr->feval(u"error", 0,
                             std::vector<Array>(
                                     {factory.createScalar("Too many output arguments.")}));
        }
    }
};
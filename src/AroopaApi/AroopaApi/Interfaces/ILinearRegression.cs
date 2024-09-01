using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AroopaApi.Interfaces
{
    public interface ILinearRegression
    {
        void Fit(double[,] X, double[] y);

        double[] Predict(double[,] X);

        double[,] Transpose(double[,] matrix);

        double[,] Multiply(double[,] A, double[,] B);

        double[] Multiply(double[,] A, double[] b);

        double[] SolveLinearSystem(double[,] A, double[] b);
    }
}

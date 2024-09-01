using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AroopaApi.IClassifiers
{
    public interface IlogisticRegression
    {
        void Fit(double[,] X, double[] y);

        int[] Predict(double[,] X);

        double[] PredictProba(double[,] X);

        double Sigmoid(double z);
    }
}

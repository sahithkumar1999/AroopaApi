using AroopaApi;
using AroopaApi.Encoders;
using System.Text;
using GPT;
using static System.Runtime.InteropServices.JavaScript.JSType;
using AroopaApi.Models.Regression;

namespace GPT
{
    public class Program
    {
        
        static void Main(string[] args)
        {
            double[,] X = new double[,]
        {
            {1, 1},
            {1, 2},
            {2, 2},
            {2, 3}
        };

            double[] y = new double[] { 6, 6, 9, 11 };

            LinearRegression lr = new LinearRegression();
            lr.Fit(X, y);

            double[] predictions = lr.Predict(X);
            Console.WriteLine("Predicted values: " + string.Join(", ", predictions));

            // Coefficients and intercept
            Console.WriteLine("Coefficients: " + string.Join(", ", lr.coef_));
            Console.WriteLine("Intercept: " + lr.intercept_);


            basicGPT basicGPT = new basicGPT();
            basicGPT.trainGPT();


        }
    }
}

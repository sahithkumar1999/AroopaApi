using AroopaApi.Interfaces;
using System;

namespace AroopaApi.Models.Regression
{
    /// <summary>
    /// A simple implementation of Ordinary Least Squares Linear Regression.
    /// This class provides methods to fit a linear model and predict new values.
    /// </summary>
    public class LinearRegression : ILinearRegression
    {
        public double[] coef_;   // Coefficients of the model (weights of the features)
        public double intercept_; // Intercept (bias) of the model
        public bool fitIntercept; // Indicates whether to calculate the intercept or assume the data is centered

        // Constructor with an optional parameter to decide whether to fit the intercept
        public LinearRegression(bool fitIntercept = true)
        {
            this.fitIntercept = fitIntercept;
            this.coef_ = new double[0]; // Initialize coefficients as an empty array
            this.intercept_ = 0.0; // Initialize intercept as 0.0
        }

        /// <summary>
        /// Fits the linear regression model using the Ordinary Least Squares method.
        /// This method computes the coefficients (and intercept, if applicable) of the model.
        /// </summary>
        /// <param name="X">2D array of input features (n_samples x n_features)</param>
        /// <param name="y">Array of target values (n_samples)</param>
        public void Fit(double[,] X, double[] y)
        {
            int nSamples = X.GetLength(0); // Number of samples
            int nFeatures = X.GetLength(1); // Number of features

            // If fitIntercept is true, augment X with a column of ones to account for the intercept
            double[,] XAugmented;
            if (fitIntercept)
            {
                XAugmented = new double[nSamples, nFeatures + 1];
                for (int i = 0; i < nSamples; i++)
                {
                    XAugmented[i, 0] = 1.0; // Add a column of ones at the beginning for the intercept
                    for (int j = 0; j < nFeatures; j++)
                    {
                        XAugmented[i, j + 1] = X[i, j]; // Copy the original feature values
                    }
                }
            }
            else
            {
                XAugmented = X; // If no intercept, use X as is
            }

            // Calculate (X^T * X), the product of the transpose of X and X itself
            double[,] XtX = Multiply(Transpose(XAugmented), XAugmented);

            // Calculate (X^T * y), the product of the transpose of X and the target values y
            double[] Xty = Multiply(Transpose(XAugmented), y);

            // Solve the linear system (X^T * X) * coef = (X^T * y) to find the coefficients
            double[] coef = SolveLinearSystem(XtX, Xty);

            // Assign coefficients and intercept based on whether fitIntercept is true
            if (fitIntercept)
            {
                intercept_ = coef[0]; // The first coefficient is the intercept
                coef_ = coef.Skip(1).ToArray(); // The rest are the feature coefficients
            }
            else
            {
                intercept_ = 0.0; // No intercept if fitIntercept is false
                coef_ = coef; // All values are feature coefficients
            }
        }

        /// <summary>
        /// Predicts target values using the fitted linear model for the given input features.
        /// </summary>
        /// <param name="X">2D array of input features (n_samples x n_features)</param>
        /// <returns>Array of predicted values</returns>
        public double[] Predict(double[,] X)
        {
            int nSamples = X.GetLength(0); // Number of samples
            double[] predictions = new double[nSamples]; // Array to store predictions

            for (int i = 0; i < nSamples; i++)
            {
                double prediction = intercept_; // Start with the intercept value
                for (int j = 0; j < coef_.Length; j++)
                {
                    prediction += X[i, j] * coef_[j]; // Add the weighted sum of the feature values
                }
                predictions[i] = prediction; // Store the prediction
            }
            return predictions; // Return the array of predictions
        }

        /// <summary>
        /// Utility function to transpose a matrix (swap rows and columns).
        /// </summary>
        /// <param name="matrix">2D array to transpose</param>
        /// <returns>Transposed 2D array</returns>
        public double[,] Transpose(double[,] matrix)
        {
            int rows = matrix.GetLength(0); // Number of rows in the original matrix
            int cols = matrix.GetLength(1); // Number of columns in the original matrix
            double[,] transposed = new double[cols, rows]; // New matrix with swapped dimensions
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transposed[j, i] = matrix[i, j]; // Swap row and column indices
                }
            }
            return transposed; // Return the transposed matrix
        }

        /// <summary>
        /// Utility function to multiply two matrices.
        /// </summary>
        /// <param name="A">First matrix (rowsA x colsA)</param>
        /// <param name="B">Second matrix (colsA x colsB)</param>
        /// <returns>Product matrix (rowsA x colsB)</returns>
        public double[,] Multiply(double[,] A, double[,] B)
        {
            int rowsA = A.GetLength(0); // Number of rows in the first matrix
            int colsA = A.GetLength(1); // Number of columns in the first matrix (must match rows of B)
            int colsB = B.GetLength(1); // Number of columns in the second matrix
            double[,] result = new double[rowsA, colsB]; // Resultant matrix
            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    result[i, j] = 0.0; // Initialize result cell to 0
                    for (int k = 0; k < colsA; k++)
                    {
                        result[i, j] += A[i, k] * B[k, j]; // Multiply and accumulate
                    }
                }
            }
            return result; // Return the product matrix
        }

        /// <summary>
        /// Utility function to multiply a matrix by a vector.
        /// </summary>
        /// <param name="A">Matrix (rowsA x colsA)</param>
        /// <param name="b">Vector (colsA)</param>
        /// <returns>Product vector (rowsA)</returns>
        public double[] Multiply(double[,] A, double[] b)
        {
            int rowsA = A.GetLength(0); // Number of rows in the matrix
            int colsA = A.GetLength(1); // Number of columns in the matrix (must match length of b)
            double[] result = new double[rowsA]; // Resultant vector
            for (int i = 0; i < rowsA; i++)
            {
                result[i] = 0.0; // Initialize result cell to 0
                for (int j = 0; j < colsA; j++)
                {
                    result[i] += A[i, j] * b[j]; // Multiply and accumulate
                }
            }
            return result; // Return the product vector
        }

        /// <summary>
        /// Solves the linear system A * x = b using Gaussian elimination.
        /// </summary>
        /// <param name="A">Coefficient matrix (n x n)</param>
        /// <param name="b">Right-hand side vector (n)</param>
        /// <returns>Solution vector x (n)</returns>
        public double[] SolveLinearSystem(double[,] A, double[] b)
        {
            int n = A.GetLength(0); // Number of equations (rows of A)
            double[,] augmentedMatrix = new double[n, n + 1]; // Augmented matrix [A|b]

            // Build the augmented matrix [A|b]
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    augmentedMatrix[i, j] = A[i, j]; // Copy A into augmented matrix
                }
                augmentedMatrix[i, n] = b[i]; // Append b as the last column
            }

            // Perform Gaussian elimination to solve the system
            for (int i = 0; i < n; i++)
            {
                // Find the pivot row (row with the maximum value in the current column)
                int maxRow = i;
                for (int k = i + 1; k < n; k++)
                {
                    if (Math.Abs(augmentedMatrix[k, i]) > Math.Abs(augmentedMatrix[maxRow, i]))
                    {
                        maxRow = k; // Update the pivot row index
                    }
                }

                // Swap the pivot row with the current row
                for (int k = i; k < n + 1; k++)
                {
                    double tmp = augmentedMatrix[maxRow, k];
                    augmentedMatrix[maxRow, k] = augmentedMatrix[i, k];
                    augmentedMatrix[i, k] = tmp;
                }

                // Make all rows below this one 0 in the current column
                for (int k = i + 1; k < n; k++)
                {
                    double factor = augmentedMatrix[k, i] / augmentedMatrix[i, i];
                    for (int j = i; j < n + 1; j++)
                    {
                        if (i == j)
                        {
                            augmentedMatrix[k, j] = 0; // Set to 0 explicitly for clarity
                        }
                        else
                        {
                            augmentedMatrix[k, j] -= factor * augmentedMatrix[i, j];
                        }
                    }
                }
            }

            // Solve the equation Ax=b for an upper triangular matrix A
            double[] x = new double[n]; // Solution vector
            for (int i = n - 1; i >= 0; i--)
            {
                x[i] = augmentedMatrix[i, n] / augmentedMatrix[i, i]; // Compute the solution for x[i]
                for (int k = i - 1; k >= 0; k--)
                {
                    augmentedMatrix[k, n] -= augmentedMatrix[k, i] * x[i]; // Update the remaining rows
                }
            }
            return x; // Return the solution vector
        }
    }
}

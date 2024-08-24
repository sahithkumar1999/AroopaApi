using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using AroopaApi.Models.NeuralNetworks;
using System;

namespace UnitTestsProject.Models.NeuralNetworks
{
    [TestClass]
    public class EmbeddingUnitTest
    {
        [TestMethod]
        //[TestCategory("Prod")]
        public void Initialization_WithoutWeight()
        {
            int num_embeddings = 5;
            int embedding_dim = 3;
            var embedding = new Embedding(num_embeddings, embedding_dim);

            Console.WriteLine($"Embedding Weight Shape: {embedding.weight.shape[0]}");
            Assert.AreEqual(embedding.weight.shape[0], 15);
            //Assert.AreEqual(embedding.weight.shape[1], embedding_dim);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void Initialization_WithWeight()
        {
            int num_embeddings = 5;
            int embedding_dim = 3;
            var weight = np.ones(new Shape(num_embeddings, embedding_dim));
            var embedding = new Embedding(num_embeddings, embedding_dim, weight: weight);

            Console.WriteLine($"Embedding Weight Shape: {embedding.weight.shape}");
            Assert.AreEqual(embedding.weight.shape[0], num_embeddings);
            Assert.AreEqual(embedding.weight.shape[1], embedding_dim);
            //Assert.IsTrue(np.allclose(embedding.weight, weight));
        }

        [TestMethod]
        //[TestCategory("Prod")]
        public void ForwardPass()
        {
            int num_embeddings = 5;
            int embedding_dim = 3;
            var embedding = new Embedding(num_embeddings, embedding_dim);
            var input = np.array(new int[] { 1, 3, 0 });

            Console.WriteLine($"Input Array: {input}");
            var result = embedding.Forward(input);

            Console.WriteLine($"Forward Pass Result Shape: {result.shape}");
            Assert.AreEqual(result.shape[0], input.size);
            Assert.AreEqual(result.shape[1], embedding_dim);
        }




    }
}

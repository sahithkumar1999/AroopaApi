using AroopaApi;
using AroopaApi.Encoders;
using System.Text;


namespace GPT
{
    public class Program
    {
        static void Main(string[] args)
        {
            // Create an instance of lambda_Encoder
            lambda_Encoder lambda_Encoder = new lambda_Encoder();
            Console.WriteLine("Hello, World!");
            List<int> encoded = lambda_Encoder.Encode("hii there");
            Console.WriteLine(string.Join(", ", encoded));
        }
    }
}

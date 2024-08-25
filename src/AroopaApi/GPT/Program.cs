using AroopaApi;
using AroopaApi.Encoders;
using System.Text;
using GPT;
using static System.Runtime.InteropServices.JavaScript.JSType;


namespace GPT
{
    public class Program
    {
        
        static void Main(string[] args)
        {
            basicGPT basicGPT = new basicGPT();
            basicGPT.trainGPT();


        }
    }
}

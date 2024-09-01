using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

namespace AroopaApi.Interfaces
{
    public interface IEmbedding
    {

        void FillPaddingIndexWithZero();

        NDArray Forward(NDArray input);
    }
}

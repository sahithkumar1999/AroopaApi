using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

namespace AroopaApi.Interfaces
{
    public interface IBigramLanguageModel
    {
        (NDArray logits, NDArray targets) Forward(NDArray idx, NDArray targets = null);

        NDArray Generate(NDArray idx, int maxNewTokens);

        NDArray Softmax(NDArray logits);
    }
}

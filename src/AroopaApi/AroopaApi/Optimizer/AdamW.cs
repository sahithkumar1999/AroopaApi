using System;
using System.Collections.Generic;

namespace AroopaApi.Optimizer
{
    public class AdamW
    {
        public readonly List<Dictionary<string, object>> paramGroups;

        public AdamW(List<Dictionary<string, object>> parameters, float lr = 1e-3f, Tuple<float, float> betas = null,
                     float eps = 1e-8f, float weightDecay = 1e-2f, bool amsgrad = false, bool maximize = false,
                     bool foreachOpt = false, bool capturable = false, bool differentiable = false, bool fused = false)
        {
            // Input validation
            if (lr < 0.0f)
                throw new ArgumentException($"Invalid learning rate: {lr}");

            if (eps < 0.0f)
                throw new ArgumentException($"Invalid epsilon value: {eps}");

            if (betas == null)
                betas = new Tuple<float, float>(0.9f, 0.999f);

            if (betas.Item1 < 0.0f || betas.Item1 >= 1.0f)
                throw new ArgumentException($"Invalid beta parameter at index 0: {betas.Item1}");

            if (betas.Item2 < 0.0f || betas.Item2 >= 1.0f)
                throw new ArgumentException($"Invalid beta parameter at index 1: {betas.Item2}");

            if (weightDecay < 0.0f)
                throw new ArgumentException($"Invalid weight_decay value: {weightDecay}");

            paramGroups = new List<Dictionary<string, object>>();

            // Initialize parameter groups
            foreach (var parameter in parameters)
            {
                var defaults = new Dictionary<string, object>()
                {
                    { "lr", lr },
                    { "betas", betas },
                    { "eps", eps },
                    { "weightDecay", weightDecay },
                    { "amsgrad", amsgrad },
                    { "maximize", maximize },
                    { "foreachOpt", foreachOpt },
                    { "capturable", capturable },
                    { "differentiable", differentiable },
                    { "fused", fused }
                };

                // Merge with provided parameters
                foreach (var key in parameter.Keys)
                {
                    if (!defaults.ContainsKey(key))
                        defaults[key] = parameter[key];
                }

                paramGroups.Add(defaults);
            }
        }

        public void Step(Action closure = null)
        {
            if (closure != null)
                closure();

            foreach (var group in paramGroups)
            {
                // Initialize lists for parameters, gradients, optimizer states
                var paramsWithGrad = new List<Dictionary<string, object>>();
                var grads = new List<Dictionary<string, object>>();
                var expAvgs = new List<Dictionary<string, float>>();
                var expAvgSqs = new List<Dictionary<string, float>>();
                var maxExpAvgSqs = new List<Dictionary<string, float>>();
                var stateSteps = new List<Dictionary<string, object>>();

                var amsgrad = (bool)group["amsgrad"];
                var betas = (Tuple<float, float>)group["betas"];
                var beta1 = betas.Item1;
                var beta2 = betas.Item2;
                var lr = (float)group["lr"];
                var weightDecay = (float)group["weightDecay"];
                var eps = (float)group["eps"];
                var maximize = (bool)group["maximize"];
                var foreachOpt = (bool)group["foreachOpt"];
                var capturable = (bool)group["capturable"];
                var differentiable = (bool)group["differentiable"];
                var fused = (bool)group["fused"];

                foreach (var param in (List<Dictionary<string, object>>)group["params"])
                {
                    if (!param.ContainsKey("grad"))
                        continue;

                    paramsWithGrad.Add(param);
                    grads.Add(param["grad"] as Dictionary<string, object>);
                    var state = (Dictionary<string, object>)param["state"];

                    if (state.Count == 0)
                    {
                        state["step"] = 0;
                        state["exp_avg"] = new Dictionary<string, float>();
                        state["exp_avg_sq"] = new Dictionary<string, float>();

                        if (amsgrad)
                            state["max_exp_avg_sq"] = new Dictionary<string, float>();
                    }

                    expAvgs.Add(state["exp_avg"] as Dictionary<string, float>);
                    expAvgSqs.Add(state["exp_avg_sq"] as Dictionary<string, float>);

                    if (amsgrad)
                        maxExpAvgSqs.Add(state["max_exp_avg_sq"] as Dictionary<string, float>);

                    stateSteps.Add(state);
                }

                foreach (var p in paramsWithGrad)
                {
                    var state = (Dictionary<string, object>)p["state"];
                    var expAvg = (Dictionary<string, float>)state["exp_avg"];
                    var expAvgSq = (Dictionary<string, float>)state["exp_avg_sq"];
                    var step = (int)state["step"];
                    step++; // Increment 'step'

                    var grad = grads[paramsWithGrad.IndexOf(p)];
                    var exp_avg = expAvgs[paramsWithGrad.IndexOf(p)];
                    var exp_avg_sq = expAvgSqs[paramsWithGrad.IndexOf(p)];
                    var step_t = step;

                    state["step"] = step_t;

                    if (weightDecay != 0)
                    {
                        var paramList = (List<float>)p["params"];
                        for (int i = 0; i < paramList.Count; i++)
                        {
                            paramList[i] *= (1 - lr * weightDecay);
                        }
                    }

                    foreach (var g in grad["grad"] as List<float>)
                    {
                        // Ensure grad["grad"] is correctly cast to List<float> before usage
                        var gradList = grad["grad"] as List<float>;

                        // Get the index of current 'g' in gradList
                        int index = gradList.IndexOf(g);
                        string indexStr = index.ToString();

                        // Update exp_avg_sq
                        float currentExpAvgSq = exp_avg_sq.ContainsKey(indexStr) ? exp_avg_sq[indexStr] : 0.0f;
                        float gradValue = g; // 'g' is the float value of the gradient
                        float newExpAvgSq = currentExpAvgSq * beta2 + (float)Math.Pow(gradValue, 2) * (1 - beta2);
                        exp_avg_sq[indexStr] = newExpAvgSq;

                        // Update exp_avg
                        float currentExpAvg = exp_avg.ContainsKey(indexStr) ? exp_avg[indexStr] : 0.0f;
                        exp_avg[indexStr] = currentExpAvg * (1 - beta1);
                    }

                    // Assuming p["params"] is of type List<List<float>>
                    var paramsList = (List<List<float>>)p["params"];

                    for (int i = 0; i < paramsList.Count; i++)
                    {
                        var paramList = paramsList[i]; // Retrieve the parameter list

                        // Convert 'i' to string for dictionary access
                        var paramListStr = i.ToString();

                        // Retrieve current exp_avg and exp_avg_sq values
                        float currentExpAvg = exp_avg.ContainsKey(paramListStr) ? exp_avg[paramListStr] : 0.0f;
                        float currentExpAvgSq = exp_avg_sq.ContainsKey(paramListStr) ? exp_avg_sq[paramListStr] : 0.0f;

                        // Perform the Adam update formula
                        for (int j = 0; j < paramList.Count; j++)
                        {
                            paramList[j] -= (float)((lr / Math.Sqrt(step_t)) * (currentExpAvg / (Math.Sqrt(currentExpAvgSq) + eps)));
                        }

                        // Update the paramList in the original 'params' list
                        paramsList[i] = paramList;
                    }

                    // Update the state
                    state["step"] = step_t;

                    if (amsgrad)
                    {
                        var max_exp_avg_sq = (Dictionary<string, float>)state["max_exp_avg_sq"]; // Ensure max_exp_avg_sq is of type Dictionary<string, float>
                        var t = (int)state["step"];

                        foreach (var paramList in p["params"] as List<List<float>>)
                        {
                            for (int i = 0; i < paramList.Count; i++)
                            {
                                string paramListStr = i.ToString(); // Convert index to string
                                max_exp_avg_sq[paramListStr] = Math.Max(max_exp_avg_sq[paramListStr], exp_avg_sq[paramListStr]);
                            }
                        }
                    }
                }
            }
        }
    }
}
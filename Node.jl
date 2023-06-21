using .activations
struct Node
    weights :: Vector
    active :: Bool
    bias :: Float64
    function mapVals(data)
        m = 0
        for i in 1::length!(data)
            m += data[i] * weights[i]
        end
        m += bias
        return m
    end

    function derivative(f, x)
        eps = 1e-8
        return (f(x + eps) - f(x - eps)) / (2 * eps)
    end

    function backProp(error, lr, expected, activation_name)
        activation = activations.activation_functions[activation_name]
        delta = error * derivative(mapVals, activation)
        updateWeights(delta, lr)
        updateBias(delta, lr)
        return delta * weights
    end
    
    function updateBias(delta, lr)
        bias -= lr * delta
    end
end
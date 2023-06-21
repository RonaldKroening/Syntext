module activations

export activation_functions

# Activation Functions

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

function relu(x)
    return max(0, x)
end

function tanh(x)
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end

function softmax(x)
    ex = exp.(x)
    return ex / sum(ex)
end

function leaky_relu(x, alpha=0.2)
    return max(alpha*x, x)
end

function elu(x, alpha=1.0)
    if x > 0
        return x
    else
        return alpha * (exp(x) - 1)
    end
end

function swish(x, beta=1.0)
    return x / (1 + exp(-beta*x))
end

function identity(x)
    return x
end

function gaussian(x, mu=0.0, sigma=1.0)
    return exp(-(x - mu)^2 / (2*sigma^2))
end

function softplus(x)
    return log(1 + exp(x))
end

# Dictionary of Activation Functions

activation_functions = Dict(
    "sigmoid" => sigmoid,
    "relu" => relu,
    "tanh" => tanh,
    "softmax" => softmax,
    "leaky_relu" => leaky_relu,
    "elu" => elu,
    "swish" => swish,
    "identity" => identity,
    "gaussian" => gaussian,
    "softplus" => softplus
)

end

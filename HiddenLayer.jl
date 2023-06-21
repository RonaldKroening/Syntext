include("Node.jl")
using Random
using .activations

struct HiddenLayer
    Nodes :: Vector{Node}
    connectedTo :: HiddenLayer
    bias :: Real
    act_func :: String
    thold :: Real
    activeNodes :: Vector{Float64}

    function initialize(numNodes)
        for i in 1:numNodes
            if connectedTo != nothing
                s = length(connectedTo.Nodes)
                w = rand(Float64, s)
                bi = rand(Float64)
                n = Node(w, false, bi)
                push!(Nodes, n)
            else
                s = length(Nodes)  # input layer
                w = rand(Float64, s)
                bi = rand(Float64)
                n = Node(w, false, bi)
                push!(Nodes, n)
            end
        end
    end

    function turnOn(values)
        activate(values, bias)
        if isnothing(connectedTo)
            connectedTo.turnOn(activeNodes)
        else
            output = sum(activeNodes)
        end
    end

    function activate(values, bias)
        for n in Nodes
            m = n.mapVals(values)
            activation = activations.activation_functions[act_func]
            r = activation(m)
            if r >= thold
                n.active = true
                push!(activeNodes, r)
            end
        end
    end

    function backpropagate(error, lr, expected)
        for n in Nodes
            n.backProp(error, lr, expected, act_func)
        end
    end
end

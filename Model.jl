using Random
include("HiddenLayer.jl")

struct Model
    lr::Float64
    bias::Float64
    layers::Union{HiddenLayer, Nothing}
    thold::Float64

    function Model(lr::Float64 = 0.0, bias::Float64 = 0.0, thold::Float64 = 0.0)
        new(lr, bias, nothing, thold)
    end

    function addLayer(model::Model, size::Int, act::String)
        h = HiddenLayer([], nothing, model.bias, act, model.thold, [])
        h.initialize(size)

        if model.layers == nothing
            model.layers = h
        else
            l = model.layers
            while l.connectedTo != nothing
                l = l.connectedTo
            end
            l.connectedTo = h
        end

        return model
    end

    function fit(model::Model, x_train, y_train)
        for i in 1:length(x_train)
            input = x_train[i]
            target = y_train[i]
            pred = forwardPass(model, input)
            error = find_error(pred, target)
            backpropagate_weights(model, error, target)
        end
    end

    function forwardPass(model::Model, input)
        l = model.layers
        while l != nothing
            l.turnOn(input)
            input = l.activeNodes
            l = l.connectedTo
        end
        return input
    end

    function find_error(pred, act)
        e = ((pred - act) * (pred - act)) / 2
        return e
    end

    function predict(model::Model, x_test, y_test)
        predictions = []
        for i in 1:length(x_test)
            input = x_test[i]
            pred = forwardPass(model, input)
            push!(predictions, pred)
        end
        return predictions
    end
    
    function backpropagate_weights(model::Model, error, expected)
        l = model.layers
        while l != nothing
            l.backpropagate(error, model.lr, expected)
            l = l.connectedTo
        end
    end
end

m = Model()

m.addLayer(32,"relu")
m.addLayer(16,"tanh")
m.addLayer(8,"sigmoid")
m.addLayer(1,"relu")

i = m.layers
while !isnothing(i.connectedTo)
    println(length!(i.Nodes))
    i = i.connectedTo
end
# Get dataset
using Pkg
using Word2Vec
using CSV
using WordTokenizers
using Random

vecAvgs = Dict()

function activation(Weighted_Inputs,bias)
    active_inputs = []
    for x in Weighted_Inputs
        t = tanh(x)
        if(t > 0)
            append!(active_inputs,t+bias)
        end
    end
    return active_inputs
end

function add_arrays(A,B)
    a = length(A)
    b = length(B)
    C = []
    if(a > b)
        for x in 1:b
            append!(C,A[x]+B[x])
        end
    else
        for x in 1:a
            append!(C,A[x]+B[x])
        end
    end
    return C
end

function average(vec)
    i = 0
    count  = 0
    for num in vec
        i+= num
        count+=1
    end
    return i/count
end

function error(expected, predicted)
    len = length(expected)
    tot = []
    for i in 1:len
        append!(tot,expected[i]-predicted[i])
    end
    sum = 0
    count = 0
    for i in in 1:len
        sum += tot[i]
        count+=1
    end
    return sum/count
end

function deriv(x)
    return x*(1-x)
end

function format_word(word)
    w = lowercase(word)
    newWord = ""
    lets = "abcdefghijklmnopqrstuvwxyz"
    for i in w
        if i in lets
            newWord *= i
        end
    end
    return newWord
end

function generate_first_input_weights(numputs)
    input_weights = []
    for x = 1 : numputs
        r = random_float()
        append!(input_weights,r)
    end
    return input_weights
end

function get_training_data(fileName)
    v = Dict()
    f = open(fileName,"r")
    contents = readLines(f)
    le = Int(round((length(contents)/2)+1))
    for x in 1:le:2
        ff = parse(Float64, contents[x+1])
        v[contents[x]] = ff
    end
    return v
end

function hidden_layer(input,weights,learningRate, iterations, bias, expectation)
    W = weights
    er = 0
    for i in 1:iterations
        Weighted_Inputs = multiply_arrays(input,W)
        weighted_sum = sum(Weighted_Inputs)
        size = length(Weighted_Inputs) / 2
        hiddenLayer = []
        for i in 1:size
            append!(hiddenLayer,weighted_sum+bias)
        end
        activeLayer = activation(hidden_layer,bias)
        output = []
        for i in 1:5
            aSum = sum(activeLayer)
            append!(output,relu(aSum))
        end
        realerror = error(expectation, output)
        er = realerror
        println("Predicted ",output, " with error rate",er)
        W = update_weights(W,learningRate,realerror)
    end
end

function multiply_arrays(A,B)
    a = length(A)
    b = length(B)
    C = []
    if(a > b)
        for x in 1:b
            append!(C,A[x]*B[x])
        end
    else
        for x in 1:a
            append!(C,A[x]*B[x])
        end
    end
    return C
end
function tt()
    csv_reader = CSV.File("dataset.csv")
    # f = open("usedvectors.txt","w")
    count = 1
    println(csv_reader[2].Title)
end
tt();
function obtain_vectors()
    csv_reader = CSV.File("dataset.csv")
    # f = open("usedvectors.txt","w")
    count = 1
    for row in csv_reader
        println("Row ",count)
        s = row.Text
        tr = split(s)
        input = []
        for t in tr
            g = 0
            try
                i = vecAvgs[t]
                g = average(i)
            catch
                i = wtvt(t)
                g = average(i)
                vecAvgs[string(t)] = g
            end
            append!(input,t)
            g1 = string(g)
            # write(f,g1)
            # write(f,"\n")
        end
        count+=1
    end
    K = keys(vecAvgs)
    for k in K
        f = open("usedvectors.txt","w")
        write(f,k)
        write(f,"\n")
        write(f,vecAvgs[k])
        write(f,"\n")
        f.close()
    end
end

function random_float()
    Random.seed!()
    randfloat = rand()
    if randfloat === 0.0
        randfloat = 1.0
    end
    return randfloat
end

function relu(x)
    if(x >0)
        return x
    else
        return 0
    end
end

function sum(A)
    sum = 0
    for a in A
        sum +=A
    end
    return sum
end

function train(learningRate, epochs, bias)
    x=1
    vecAvgs = get_training_data("usedvectors.txt")
    rows = csv_reader.rows
    columns = csv_reader.columns
    weights = []
    for row in csv_reader 
        expected_output = []
        value = row.ScienceFiction
        append!(expected_output,value)
        value = row.Fantasy
        append!(expected_output,value)
        value = row.News
        append!(expected_output,value)
        value = row.ScientificPaper
        append!(expected_output,value)
        value = row.Biography
        append!(expected_output,value)
        s = row.Text
        tr = split(s)
        input = []
        for t in tr
            try
                i = vecAvgs[t]
                g = average(i)
                if(g != 0.0)
                    append!(input, g)
                end
            catch
                i = wtvt(t)
                g = average(i)
                if(g != 0.0)
                    append!(input, g)
                end
            end
        end
        println("Expected ",expected_output);
        weights = generate_first_input_weights()
        weights = hidden_layer(input,weights,learningRate,epochs, bias, expected)
        break
    end
    return weights
end

function update_weight(weight,learningRate,error)
    derivE = deriv(error)
    derivOldW = deriv(weight)
    div = derivE/derivOldW
    sub = learningRate * div
    return weight - sub
end

function update_weights(weights, learningRate,error)
    new_weights = []
    for w in weights
        nw = update_weight(w,learningRate,error)
        append!(new_weights,nw)
    end
    return new_weights
end

function wtvt(word)
    model = wordvectors("text8-vec.txt")
    w = lowercase(word)
    try
        return get_vector(model,w)
    catch
        return 0.0
    end
end

learningRate = .6
epochs = 10
bias = .88
# weights = train(learningRate,epochs,bias);
# print("hi")
#Use the weights for further analysis
# print("hello world")


# obtain_vectors();
# Weighted = [-0.55,0.7,0.05,-0.03,0.9,0.45,-0.001]
# A = activation(Weighted,1)
# println(A)
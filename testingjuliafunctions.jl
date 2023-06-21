using Images, FileIO

mutable struct Node(inputs)
    weights = []
    function set_initial_weights()  
        for i in 1:inputs
            append!(weights,random_float())
        end
        return weights
    end
    function set_weights(new)
        weights.clear()
        for n in new
            append!(weights,n)
        end
    end
    function get_weights()
        return weights
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

function format(i)
    Image = load(i)
    i2 = Image[1,2]
    t = []
    a = Float32.(i2.r)+0.0
    append!(t,a)
    a = Float32.(i2.g)+0.0
    append!(t,a)
    a = Float32.(i2.b)+0.0
    append!(t,a)
    return t
end

function display(A)
    for x in A
        print(x, " ")
    end
end
A = [1,2,3]
B = [4,3,2,1]
c = multiply_arrays(A,B)
# cd("..")
# cd("trainingdata/")
# format("cat1.png")
n = Node(12)
println(n.get_weights())
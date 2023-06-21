using Pkg
using CSV
using Random
using DataFrames



function frequency(word, document):
    i=0
    for w in document:
        if(w == word):
            i+=1
        end
    end
    return i
end

function appearences(word, documents):
    i=0
    for document in documents:
        if(word in documents):
            i+=1
        end
    end
    return i
end

function tfidf(documents, word, document):
    return frequency(word,document)*log(len(documents),appearences(word, documents) )
end

print("hello world!")
DF = CSV.read("dataset.csv", DataFrame)
print(DF.Column2[2])
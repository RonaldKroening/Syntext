using TextAnalysis
struct Dataset
    keys = []
    features = {}

    function remove_stopwords(text)
        stopwords = ["i, me, my, myself, we, our, ours, ourselves, you, you're, you've, you'll, you'd, your, yours, yourself, yourselves, he, him, his, himself, she, she's, her, hers, herself, it, it's, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, that'll, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don, don't, should, should've, now, d, ll, m, o, re, ve, y, ain, aren, aren't, couldn, couldn't, didn, didn't, doesn, doesn't, hadn, hadn't, hasn, hasn't, haven, haven't, isn, isn't, ma, mightn, mightn't, mustn, mustn't, needn, needn't, shan, shan't, shouldn, shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't"]
        return " ".join([word for word in str(text).split() if word not in stopwords])
    end

    function remove_punctuation(text)
        remove = "!#$%&'()*+,-./:;<=>?@[\\]^_{|}~`"
        return join([t for t in text if !(t in remove || t == ' ')])
    end 

    function clean_dataset()
        for key in keys
            NewData = split(data[key]," ")
            idx=0
            for text in NewData
                t1 = remove_punctuation(text)
                t2 = remove_stopwords(t1)
                t3 = Lemmatizer()(t2)
                NewData[idx] = t3
                idx+=1
            data[key] = NewData
    struct Dataset(d)


end
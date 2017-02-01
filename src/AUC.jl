using Base.Cartesian

#"""
#AUC.jl provides fast and memory efficient versions to compute the Area under der receiver operator curve and true/false positive rates after
#Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874. http://doi.org/10.1016/j.patrec.2005.10.010
#"""

macro removeTrailing(x...)
    ex=Expr(:block)
    for i in 1:length(x)
        push!(ex.args,:(endswith($(esc(x[i])),"\0") && ($(esc(x[i]))=$(esc(x[i]))[1:end-1])))
    end
    ex
end

"""
    auc(scores, events, increasing = true)

compute the Area Under the receiver operator Curve (AUC), given some output `scores` array and some ground truth (`events`).
By default, it is assumed, that the `scores` are ordered increasingly (`increasing = true`), i.e. high scores represent events.

# Examples
```
julia> scores = rand(10, 2)
julia> events = rand(0:1, 10, 2)
julia> auc(scores, events)
julia> auc(scores, boolevents(events))
```
"""
function auc(scores,events; increasing::Bool = true)
    s = sortperm(reshape(scores,length(scores)),rev=increasing);
    length(scores) == length(events) || error("Scores and events must have same number of elements")
    f=scores[s]
    L=events[s]
    fp=0
    tp=0
    fpprev=0
    tpprev=0
    A=0.0
    fprev=-Inf
    P=sum(L)
    N=length(L)-P
    for i=1:length(L)
        if f[i]!=fprev
            A+=trap_area(fp,fpprev,tp,tpprev)
            @inbounds fprev=f[i]
            fpprev=fp
            tpprev=tp
        end
        if isextreme(L[i])
            tp+=1
        else
            fp+=1
        end
    end
    A+=trap_area(N,fpprev,P,tpprev)
    A=A/(P*N)
end

function trap_area(x1,x2,y1,y2)
    b=abs(x1-x2)
    h=0.5*(y1+y2)
    return b*h
end

"""
    auc_fpr_tpr(scores, events, quant = 0.9, increasing = true)

Similar like `auc()`, but return additionally the true positive and false positive rate at a given quantile (default: `quant = 0.9`).

# Examples

```
julia> scores = rand(10, 2)
julia> events = rand(0:1, 10, 2)
julia> auc_fpr_tpr(scores, events, 0.8)
```
"""
# function assumes that outliers have highest scores (increasing = true)
# additionally to the AUC the false positive rate and the true positive and true positive rate of a specified quantile is returned
function auc_fpr_tpr(scores,events, quant = 0.90; increasing = true)
    quant = 1.0 - quant # as we start with the highest score in the loop the quantile is reverted
    s = sortperm(reshape(scores,length(scores)),rev=increasing);
    length(scores) == length(events) || error("Scores and events must have same number of elements")
    f=scores[s]
    L=events[s]
    fp=0
    tp=0
    fpr = 0.0
    tpr = 0.0
    fpprev=0
    tpprev=0
    A=0.0
    fprev=-Inf
    P=sum(L)
    N=length(L)-P
    for i=1:length(L)
        if f[i]!=fprev
            A+=trap_area(fp,fpprev,tp,tpprev)
            @inbounds fprev=f[i]
            fpprev=fp
            tpprev=tp
        end
        if isextreme(L[i])
            tp+=1
        else
            fp+=1
        end
        if i == Int(round(quant * length(L), 0))
            fpr = fp/N
            tpr = tp/P
        end
    end
    A+=trap_area(N,fpprev,P,tpprev)
    A=A/(P*N)
    return(A, fpr, tpr)
end

isextreme(l::Bool)=l
isextreme(l::Integer)=l>0

"""
    boolevents(events)

convert an `events` array into a boolean array.
"""
function boolevents(events)
    eventsb=falses(size(events,1),size(events,2),size(events,3))
    nindep=size(events,4)
    Base.Cartesian.@nloops 3 i eventsb begin
        for i_4=1:nindep
            (Base.Cartesian.@nref(4,events,i) > 0.0) && (Base.Cartesian.@nref(3,eventsb,i)=true; break)
        end
    end
    eventsb
end



###################################
# end

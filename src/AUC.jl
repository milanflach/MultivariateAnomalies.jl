#using Base.Cartesian



"""
    auc(scores, events, increasing = true)

compute the Area Under the receiver operator Curve (AUC), given some output `scores` array and some ground truth (`events`).
By default, it is assumed, that the `scores` are ordered increasingly (`increasing = true`), i.e. high scores represent events.
AUC is computed according to Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874. http://doi.org/10.1016/j.patrec.2005.10.010

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



isextreme(l::Bool)=l
isextreme(l::Integer)=l>0




###################################
# end


<a id='AUC-1'></a>

## AUC


Compute true positive rates, false positive rates and the area under the curve to evaulate the algorihtms performance. Efficient implementation according to


Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874. http://doi.org/10.1016/j.patrec.2005.10.010


<a id='Functions-1'></a>

## Functions

<a id='MultivariateAnomalies.auc' href='#MultivariateAnomalies.auc'>#</a>
**`MultivariateAnomalies.auc`** &mdash; *Function*.



```
auc(scores, events, increasing = true)
```

compute the Area Under the receiver operator Curve (AUC), given some output `scores` array and some ground truth (`events`). By default, it is assumed, that the `scores` are ordered increasingly (`increasing = true`), i.e. high scores represent events.

**Examples**

```jlcon
julia> scores = rand(10, 2)
julia> events = rand(0:1, 10, 2)
julia> auc(scores, events)
julia> auc(scores, boolevents(events))
```

<a id='MultivariateAnomalies.auc_fpr_tpr' href='#MultivariateAnomalies.auc_fpr_tpr'>#</a>
**`MultivariateAnomalies.auc_fpr_tpr`** &mdash; *Function*.



```
auc_fpr_tpr(scores, events, quant = 0.9, increasing = true)
```

Similar like `auc()`, but return additionally the true positive and false positive rate at a given quantile (default: `quant = 0.9`).

**Examples**

```jlcon
julia> scores = rand(10, 2)
julia> events = rand(0:1, 10, 2)
julia> auc_fpr_tpr(scores, events, 0.8)
```

<a id='MultivariateAnomalies.boolevents' href='#MultivariateAnomalies.boolevents'>#</a>
**`MultivariateAnomalies.boolevents`** &mdash; *Function*.



```
boolevents(events)
```

convert an `events` array into a boolean array.


# Module AUC
scores = rand(10, 2)
events = rand(0:1, 10, 2)
auc(scores, events)
auc(scores, boolevents(events))
auc_fpr_tpr(scores, events)

# Module Scores
scores1 = rand(10, 2)
scores2 = rand(10, 2)
quantile_scores1 = get_quantile_scores(scores1)
quantile_scores2 = get_quantile_scores(scores2)
compute_ensemble(quantile_scores1, quantile_scores2, ensemble = "mean")
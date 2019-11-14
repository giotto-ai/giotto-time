def smape(pred, ground_truth):
    assest len(pred) == len(ground_truth)
    tot = 0
    for f, a in zip(pred, ground_truth):
        tot += abs( f-a )/( abs(a) - abs(f) )

    return (s/len(pred))*tot


#

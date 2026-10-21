from hemishift.eval.safetransfer import choose


def test_ties_go_to_the_candidate_listed_first():
    assert choose({"own": 18, "transfer": 18}, 20) == ("own", 0.9)


def test_transfer_is_chosen_when_it_scores_higher():
    assert choose({"own": 10, "transfer": 17}, 20) == ("transfer", 0.85)


def test_abstains_when_the_best_is_not_above_chance():
    name, accuracy = choose({"own": 12, "transfer": 13}, 20)
    assert name is None and accuracy == 0.65


def test_more_candidates_need_a_higher_calibration_score():
    # 15/20 beats chance at alpha 0.05 for one candidate (threshold 0.75) but not after
    # Bonferroni over three (threshold 0.80).
    assert choose({"own": 15}, 20)[0] == "own"
    assert choose({"own": 15, "transfer": 14, "ensemble": 14}, 20)[0] is None

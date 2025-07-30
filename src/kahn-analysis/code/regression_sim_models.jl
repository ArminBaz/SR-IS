using MixedModels

fm_rwdonly = @formula(action1TowardsPrevEnd ~ rewardₜ₋₁ + (rewardₜ₋₁ | subject))

fm_interactions_neighborboatlag5policylag5sameboatlag5 = @formula(action1TowardsPrevEnd ~ 
+ rewardₜ₋₁ * (lag1_neighborboat_reg + lag2_neighborboat_reg + lag3_neighborboat_reg + lag4_neighborboat_reg + lag5_neighborboat_reg)
+ rewardₜ₋₁ * (lag1_sameboat_reg + lag2_sameboat_reg + lag3_sameboat_reg + lag4_sameboat_reg + lag5_sameboat_reg)
+ rewardₜ₋₁ * (lag1_policy_reg + lag2_policy_reg + lag3_policy_reg + lag4_policy_reg + lag5_policy_reg)
+ lag1_oppislandavgboat_reg + lag2_oppislandavgboat_reg + lag3_oppislandavgboat_reg + lag4_oppislandavgboat_reg + lag5_oppislandavgboat_reg
+ lag1_choice_autoreg + lag2_choice_autoreg + lag3_choice_autoreg + lag4_choice_autoreg + lag5_choice_autoreg
+ (rewardₜ₋₁ * lag1_neighborboat_reg + rewardₜ₋₁ * lag1_policy_reg
| subject))

interaction_vars = [
    :lag1_neighborboat_reg, :lag2_neighborboat_reg, :lag3_neighborboat_reg, :lag4_neighborboat_reg, :lag5_neighborboat_reg,
    :lag6_neighborboat_reg, :lag7_neighborboat_reg, :lag8_neighborboat_reg, :lag9_neighborboat_reg, :lag10_neighborboat_reg,
    :lag1_sameboat_reg, :lag2_sameboat_reg, :lag3_sameboat_reg, :lag4_sameboat_reg, :lag5_sameboat_reg,
    :lag6_sameboat_reg, :lag7_sameboat_reg, :lag8_sameboat_reg, :lag9_sameboat_reg, :lag10_sameboat_reg,
    :lag1_policy_reg, :lag2_policy_reg, :lag3_policy_reg, :lag4_policy_reg, :lag5_policy_reg,
    :lag6_policy_reg, :lag7_policy_reg, :lag8_policy_reg, :lag9_policy_reg, :lag10_policy_reg,
]
rwd_vars = [:rewardₜ₋₁,
    :lag1_oppislandsameboat_reg, :lag2_oppislandsameboat_reg, :lag3_oppislandsameboat_reg, :lag4_oppislandsameboat_reg, :lag5_oppislandsameboat_reg,
    :lag6_oppislandsameboat_reg, :lag7_oppislandsameboat_reg, :lag8_oppislandsameboat_reg, :lag9_oppislandsameboat_reg, :lag10_oppislandsameboat_reg,
    :lag1_oppislandoppboat_reg, :lag2_oppislandoppboat_reg, :lag3_oppislandoppboat_reg, :lag4_oppislandoppboat_reg, :lag5_oppislandoppboat_reg,
    :lag6_oppislandoppboat_reg, :lag7_oppislandoppboat_reg, :lag8_oppislandoppboat_reg, :lag9_oppislandoppboat_reg, :lag10_oppislandoppboat_reg,
    :lag1_oppislandavgboat_reg, :lag2_oppislandavgboat_reg, :lag3_oppislandavgboat_reg, :lag4_oppislandavgboat_reg, :lag5_oppislandavgboat_reg,
    :lag6_oppislandavgboat_reg, :lag7_oppislandavgboat_reg, :lag8_oppislandavgboat_reg, :lag9_oppislandavgboat_reg, :lag10_oppislandavgboat_reg,
]
choice_vars = [
    :lag1_boatonly_same_reg, :lag2_boatonly_same_reg, :lag3_boatonly_same_reg, :lag4_boatonly_same_reg, :lag5_boatonly_same_reg,
    :lag6_boatonly_same_reg, :lag7_boatonly_same_reg, :lag8_boatonly_same_reg, :lag9_boatonly_same_reg, :lag10_boatonly_same_reg,
    :lag1_boatonly_opp_reg, :lag2_boatonly_opp_reg, :lag3_boatonly_opp_reg, :lag4_boatonly_opp_reg, :lag5_boatonly_opp_reg,
    :lag6_boatonly_opp_reg, :lag7_boatonly_opp_reg, :lag8_boatonly_opp_reg, :lag9_boatonly_opp_reg, :lag10_boatonly_opp_reg,
    :lag1_choice_autoreg, :lag2_choice_autoreg, :lag3_choice_autoreg, :lag4_choice_autoreg, :lag5_choice_autoreg, :lag6_choice_autoreg, :lag7_choice_autoreg, :lag8_choice_autoreg, :lag9_choice_autoreg, :lag10_choice_autoreg,
    :lag1_boat_autoreg, :lag2_boat_autoreg, :lag3_boat_autoreg, :lag4_boat_autoreg, :lag5_boat_autoreg, :lag6_boat_autoreg, :lag7_boat_autoreg, :lag8_boat_autoreg, :lag9_boat_autoreg, :lag10_boat_autoreg,
    :lag1_traversaltrial_choicereg, :lag2_traversaltrial_choicereg, :lag3_traversaltrial_choicereg, :lag4_traversaltrial_choicereg, :lag5_traversaltrial_choicereg,
    :lag6_traversaltrial_choicereg, :lag7_traversaltrial_choicereg, :lag8_traversaltrial_choicereg, :lag9_traversaltrial_choicereg, :lag10_traversaltrial_choicereg,
    :lag1_boattrial_choicereg, :lag2_boattrial_choicereg, :lag3_boattrial_choicereg, :lag4_boattrial_choicereg, :lag5_boattrial_choicereg,
    :lag6_boattrial_choicereg, :lag7_boattrial_choicereg, :lag8_boattrial_choicereg, :lag9_boattrial_choicereg, :lag10_boattrial_choicereg,
]

function centerrwd(df_orig)
    df = copy(df_orig)
    for v in rwd_vars
        df[!, v] .= df[:, v] .- 0.5
    end
    for v in choice_vars
        df[!, v] .= df[:, v] .* 1.0
    end
    for v in interaction_vars
        df[!, v] .= df[:, v] .* 1.0
    end
    return df
end

function centerrwdchoice(df_orig)
    df = copy(df_orig)
    for v in rwd_vars
        df[!, v] .= df[:, v] .- 0.5
    end
    for v in choice_vars
        df[!, v] .= df[:, v] .- 0.5
    end
    for v in interaction_vars
        df[!, v] .= df[:, v] .* 1.0
    end
    return df
end

function centerall(df_orig)
    df = copy(df_orig)
    for v in rwd_vars
        df[!, v] .= df[:, v] .- 0.5
    end
    for v in choice_vars
        df[!, v] .= df[:, v] .- 0.5
    end
    for v in interaction_vars
        df[!, v] .= df[:, v] .- 0.5
    end
    return df
end

function centerrwdtrue(df_orig)
    df = copy(df_orig)
    for v in rwd_vars
        df[!, v] .= df[:, v] .- mean(skipmissing(df[:, v]))
    end
    for v in choice_vars
        df[!, v] .= df[:, v] .* 1.0
    end
    for v in interaction_vars
        df[!, v] .= df[:, v] .* 1.0
    end
    return df
end

function centerrwdchoicetrue(df_orig)
    df = copy(df_orig)
    for v in rwd_vars
        df[!, v] .= df[:, v] .- mean(skipmissing(df[:, v]))
    end
    for v in choice_vars
        df[!, v] .= df[:, v] .- mean(skipmissing(df[:, v]))
    end
    for v in interaction_vars
        df[!, v] .= df[:, v] .* 1.0
    end
    return df
end

function centeralltrue(df_orig)
    df = copy(df_orig)
    for v in rwd_vars
        df[!, v] .= df[:, v] .- mean(skipmissing(df[:, v]))
    end
    for v in choice_vars
        df[!, v] .= df[:, v] .- mean(skipmissing(df[:, v]))
    end
    for v in interaction_vars
        df[!, v] .= df[:, v] .- mean(skipmissing(df[:, v]))
    end
    return df
end

function make_reg_suffix(centering, biastype, interactiontype)
    suffix = ""
    center_fn = x -> x
    fm = nothing
    if centering == :rwd
        center_fn = centerrwd
        suffix *= "_center-rwd"
    elseif centering == :rwdchoice
        center_fn = centerrwdchoice
        suffix *= "_center-rwd-choice"
    elseif centering == :all
        center_fn = centerall
        suffix *= "_center-all"
    elseif centering == :rwdtrue
        center_fn = centerrwdtrue
        suffix *= "_center-rwd-true"
    elseif centering == :rwdchoicetrue
        center_fn = centerrwdchoicetrue
        suffix *= "_center-rwd-choice-true"
    elseif centering == :alltrue
        center_fn = centeralltrue
        suffix *= "_center-all-true"
    elseif centering == :none
        suffix *= "_center-none"
    end

    if biastype == :rwdonly
        fm = fm_rwdonly
        suffix *= "_rwdonly"
    elseif biastype == :neighborboatlag5policylag5sameboatlag5
        suffix *= "_neighborboatlag5policylag5sameboatlag5"
        fm_interactions_neighborboatlag5policylag5sameboatlag5
        fm = fm_interactions_neighborboatlag5policylag5sameboatlag5
    end
    return suffix, center_fn, fm
end
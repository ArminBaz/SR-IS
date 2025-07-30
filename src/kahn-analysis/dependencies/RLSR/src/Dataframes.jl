function RunToDataFrame(episode_record::Vector{Episode}; subject=0)
    #      1
    #  2       3
    # 4 5     6 7
    # 8 9    10 11
    df = DataFrame()

    df.state1 = [ep.S[1] for ep in episode_record]
    df.state2 = [ep.S[2] < 8 ? ep.S[2] : missing for ep in episode_record]
    df.state3 = [length(ep.S) > 2 && ep.S[3] < 8 ? ep.S[3] : missing for ep in episode_record]

    # endState is 2-7 (2/3 for island-only trials, 4-7 for other trials)
    df.endState = [ep.S[end] > 3 ? ep.S[end-1] : ep.S[end] for ep in episode_record]
    df.endBranchLeft = [(ep.S[end] == 2) || (ep.S[end-1] == 4) || (ep.S[end-1] == 5) for ep in episode_record]
    df.reward = [ep.R[end] for ep in episode_record]
    # As it stands, an even state means we went left, odd right
    df.action1Left = mod.(df[!, :state2], 2) .== 0
    df.action2Left = mod.(df[!, :state3], 2) .== 0
    df.rewardₜ₋₁ = lag(df[!, :reward])
    df.rewardₜ₋₂ = lag(df[!, :rewardₜ₋₁])
    df.reward_lag3 = lag(df[!, :rewardₜ₋₂])
    df.reward_lag4 = lag(df[!, :reward_lag3])
    df.reward_lag5 = lag(df[!, :reward_lag4])
    df.reward_lag6 = lag(df[!, :reward_lag5])
    df.reward_lag7 = lag(df[!, :reward_lag6])
    df.reward_lag8 = lag(df[!, :reward_lag7])
    df.state1ₜ₋₁ = lag(df[!, :state1])
    df.state1ₜ₋₂ = lag(df[!, :state1ₜ₋₁])
    df.state1ₜ₋₃ = lag(df[!, :state1ₜ₋₂])
    df.state2ₜ₋₁ = lag(df[!, :state2])
    df.state2ₜ₋₂ = lag(df[!, :state2ₜ₋₁])
    df.state3ₜ₋₁ = lag(df[!, :state3])
    df.state3ₜ₋₂ = lag(df[!, :state3ₜ₋₁])
    df.endStateₜ₋₁ = lag(df[!, :endState])
    df.endStateₜ₋₂ = lag(df[!, :endStateₜ₋₁])
    df.endBranchLeftₜ₋₁ = lag(df[!, :endBranchLeft])
    df.endBranchLeftₜ₋₂ = lag(df[!, :endBranchLeftₜ₋₁])
    df.endBranchLeft_lag3 = lag(df[!, :endBranchLeftₜ₋₂])
    df.endBranchLeft_lag4 = lag(df[!, :endBranchLeft_lag3])
    df.endBranchLeft_lag5 = lag(df[!, :endBranchLeft_lag4])
    df.endBranchLeft_lag6 = lag(df[!, :endBranchLeft_lag5])
    df.endBranchLeft_lag7 = lag(df[!, :endBranchLeft_lag6])
    df.endBranchLeft_lag8 = lag(df[!, :endBranchLeft_lag7])
    
    df.trial = 1:nrow(df)
    df[!, :subject] .= string(subject)
    
    # Crucial prediction variable: Is our action towards the same side as the prior sampled state?
    df.action1TowardsPrevEnd = df[!, :action1Left] .== df[!, :endBranchLeftₜ₋₁]
    # What about our last island choice?
    df.action1Persistance = df[!, :action1Left] .== df[!, :endBranchLeftₜ₋₂]

    # Was the sampled state already our chosen side?
    df.endBranchLag1Lag2Equal = df[!, :endBranchLeftₜ₋₁] .== df[!, :endBranchLeftₜ₋₂]
    df.endBranchLag1Lag3Equal = df[!, :endBranchLeftₜ₋₁] .== df[!, :endBranchLeft_lag3]
    df.endBranchLag1Lag4Equal = df[!, :endBranchLeftₜ₋₁] .== df[!, :endBranchLeft_lag4]
    df.endBranchLag1Lag5Equal = df[!, :endBranchLeftₜ₋₁] .== df[!, :endBranchLeft_lag5]
    df.endBranchLag1Lag6Equal = df[!, :endBranchLeftₜ₋₁] .== df[!, :endBranchLeft_lag6]
    df.endBranchLag1Lag7Equal = df[!, :endBranchLeftₜ₋₁] .== df[!, :endBranchLeft_lag7]
    df.endBranchLag1Lag8Equal = df[!, :endBranchLeftₜ₋₁] .== df[!, :endBranchLeft_lag8]

    # If it was on the sampled side and rewarded, or on the un-sampled side and not rewarded
    df.choiceReg_lag1 = (df.endBranchLag1Lag2Equal .& (df.rewardₜ₋₂ .== 1.0))    .| (.!df.endBranchLag1Lag2Equal .& (df.rewardₜ₋₂ .< 1))
    df.choiceReg_lag2 = (df.endBranchLag1Lag4Equal .& (df.reward_lag4 .== 1.0)) .| (.!df.endBranchLag1Lag4Equal .& (df.reward_lag4 .< 1))
    df.choiceReg_lag3 = (df.endBranchLag1Lag6Equal .& (df.reward_lag6 .== 1.0)) .| (.!df.endBranchLag1Lag6Equal .& (df.reward_lag6 .< 1))
    df.choiceReg_lag4 = (df.endBranchLag1Lag8Equal .& (df.reward_lag8 .== 1.0)) .| (.!df.endBranchLag1Lag8Equal .& (df.reward_lag8 .< 1))
    df.boatReg_lag1 = (df.endBranchLag1Lag3Equal .& (df.reward_lag3 .== 1.0)) .| (.!df.endBranchLag1Lag3Equal .& (df.reward_lag3 .< 1))
    df.boatReg_lag2 = (df.endBranchLag1Lag5Equal .& (df.reward_lag5 .== 1.0)) .| (.!df.endBranchLag1Lag5Equal .& (df.reward_lag5 .< 1))
    df.boatReg_lag3 = (df.endBranchLag1Lag7Equal .& (df.reward_lag7 .== 1.0)) .| (.!df.endBranchLag1Lag7Equal .& (df.reward_lag7 .< 1))
    
    #######
    # Prior to the current episode, what was the most recent move at each state?
    priorMoveMat = Matrix{Union{Missing, Int}}(missing, length(episode_record), 3)
    priorMoveVec = Vector{Union{Missing, Int}}(missing, 3)
    for (i, ep) in enumerate(episode_record)
        if ep.S[1] == 1
            # If a full traversal, update moves for start and island
            if length(ep.S) > 2
                endstate = ep.S[end]
                choice_1 = 2 + (endstate > 9)  # 2 or 3
                choice_2 = endstate - 4  # 4,5,6,7
                priorMoveVec[1] = choice_1
                priorMoveVec[choice_1] = choice_2
            # Otherwise just update moves for start
            else
                priorMoveVec[1] = ep.S[2]
            end
        end
        priorMoveMat[i, :] = priorMoveVec[:]
    end
    df.priorMoveAt1 = lag(priorMoveMat[:, 1])
    df.priorMoveAt2 = lag(priorMoveMat[:, 2])
    df.priorMoveAt3 = lag(priorMoveMat[:, 3])
    # df.priorMoveAt1Left = mod.(df[!, :priorMoveAt1], 2) .== 0
    # df.priorMoveAt2Left = mod.(df[!, :priorMoveAt2], 2) .== 0
    # df.priorMoveAt3Left = mod.(df[!, :priorMoveAt3], 2) .== 0
    # Prior to the current episode, what was the most recent move at the parent
    # of the state we finished this episode in?
    parentMap = Dict(2 => :priorMoveAt1,
                     3 => :priorMoveAt1,
                     4 => :priorMoveAt2,
                     5 => :priorMoveAt2,
                     6 => :priorMoveAt3,
                     7 => :priorMoveAt3)
    df.parentPriorMove = [df[i, parentMap[df[i, :endState]]] for i in 1:nrow(df)]
    # df.priorMoveAtParentLeft = mod.(df[!, :priorMoveAtParent], 2) .== 0
    # df.priorMoveAtParentLeftₜ₋₁ = lag(df[!, :priorMoveAtParentLeft])
    df.parentPriorMoveToEndState = df[!, :parentPriorMove] .== df[!, :endState]
    df.parentPriorMoveToEndStateₜ₋₁ = lag(df[!, :parentPriorMoveToEndState])
    
    ######
    # Prior to the current episode, what was the reward observed at a given state?
    priorRewardMat = Array{Union{Missing, Float64}}(missing, length(episode_record), 6, 3)
    priorRewardVec = Array{Union{Missing, Float64}}(missing, 6, 3)
    for (i, ep) in enumerate(episode_record)
        if ep.S[end] > 3  # For trials that end on boats, 8 -> 3, 9 -> 4, 10 -> 5, 11 -> 6
            ind = ep.S[end] - 5
            priorRewardVec[ind, 3] = priorRewardVec[ind, 2]
            priorRewardVec[ind, 2] = priorRewardVec[ind, 1]
            priorRewardVec[ind, 1] = ep.R[end]
        else  # For trials that end on islands, 2 -> 1, 3 -> 2
            ind = ep.S[end] - 1
            priorRewardVec[ind, 3] = priorRewardVec[ind, 2]
            priorRewardVec[ind, 2] = priorRewardVec[ind, 1]
            priorRewardVec[ind, 1] = ep.R[end]
        end
        priorRewardMat[i, :, :] = priorRewardVec[:, :]
    end
    # If at trial 10, priorRewardAtX contains the last update including trial 9
    # Note that these are already lagged
    df.priorRewardAt2 = lag(priorRewardMat[:, 1, 1])
    df.priorRewardAt3 = lag(priorRewardMat[:, 2, 1])
    df.priorRewardAt4 = lag(priorRewardMat[:, 3, 1])
    df.priorRewardAt5 = lag(priorRewardMat[:, 4, 1])
    df.priorRewardAt6 = lag(priorRewardMat[:, 5, 1])
    df.priorRewardAt7 = lag(priorRewardMat[:, 6, 1])
    # Which index in recentRewardMat should each end-state look into?
    rewardSiblingMap = Dict(2 => :priorRewardAt3,
                            3 => :priorRewardAt2,
                            4 => :priorRewardAt5,
                            5 => :priorRewardAt4,
                            6 => :priorRewardAt7,
                            7 => :priorRewardAt6)
    # If on trial 10 we ended on 
    df.endStateSiblingPriorReward = [df[i, rewardSiblingMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.endStateSiblingPriorRewardₜ₋₁ = lag(df[!, :endStateSiblingPriorReward])
    # 
    rewardMap = Dict(2 => :priorRewardAt2,
                     3 => :priorRewardAt3,
                     4 => :priorRewardAt4,
                     5 => :priorRewardAt5,
                     6 => :priorRewardAt6,
                     7 => :priorRewardAt7)
    df.endStatePriorReward = [df[i, rewardMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.endStatePriorRewardₜ₋₁ = lag(df[!, :endStatePriorReward])
    df.endStatePriorRewardₜ₋₂ = lag(df[!, :endStatePriorRewardₜ₋₁])

    # Again, already lagged
    df.twoBackRewardAt2 = lag(priorRewardMat[:, 1, 2])
    df.twoBackRewardAt3 = lag(priorRewardMat[:, 2, 2])
    df.twoBackRewardAt4 = lag(priorRewardMat[:, 3, 2])
    df.twoBackRewardAt5 = lag(priorRewardMat[:, 4, 2])
    df.twoBackRewardAt6 = lag(priorRewardMat[:, 5, 2])
    df.twoBackRewardAt7 = lag(priorRewardMat[:, 6, 2])
    twoBackRewardMap = Dict(2 => :twoBackRewardAt2,
                            3 => :twoBackRewardAt3,
                            4 => :twoBackRewardAt4,
                            5 => :twoBackRewardAt5,
                            6 => :twoBackRewardAt6,
                            7 => :twoBackRewardAt7)
    df.endStateTwoBackReward = [df[i, twoBackRewardMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.endStateTwoBackReward_lag1 = lag(df[!, :endStateTwoBackReward])
    df.threeBackRewardAt2 = lag(priorRewardMat[:, 1, 3])
    df.threeBackRewardAt3 = lag(priorRewardMat[:, 2, 3])
    df.threeBackRewardAt4 = lag(priorRewardMat[:, 3, 3])
    df.threeBackRewardAt5 = lag(priorRewardMat[:, 4, 3])
    df.threeBackRewardAt6 = lag(priorRewardMat[:, 5, 3])
    df.threeBackRewardAt7 = lag(priorRewardMat[:, 6, 3])
    threeBackRewardMap = Dict(2 => :threeBackRewardAt2,
                            3 => :threeBackRewardAt3,
                            4 => :threeBackRewardAt4,
                            5 => :threeBackRewardAt5,
                            6 => :threeBackRewardAt6,
                            7 => :threeBackRewardAt7)
    df.endStateThreeBackReward = [df[i, threeBackRewardMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.endStateThreeBackReward_lag1 = lag(df[!, :endStateThreeBackReward])
    

    #####
    # Prior reward observed on the left/right branches
    #
    priorRewardBranchMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 2)
    priorRewardBranchVec = Vector{Union{Missing, Float64}}(missing, 2)
    for (i, ep) in enumerate(episode_record)
        if ep.S[end] > 3
            ind = 1 + (ep.S[end] > 9)
            priorRewardBranchVec[ind] = ep.R[end]
        end
        priorRewardBranchMat[i, :] = priorRewardBranchVec[:]
    end
    df.priorRewardLeftBranch = lag(priorRewardBranchMat[:, 1])
    df.priorRewardRightBranch = lag(priorRewardBranchMat[:, 2])
    # Which index in priorRewardBranchMat should each end-state look into?
    rewardBranchMap = Dict(2 => :priorRewardLeftBranch,
                           3 => :priorRewardRightBranch,
                           4 => :priorRewardLeftBranch,
                           5 => :priorRewardLeftBranch,
                           6 => :priorRewardRightBranch,
                           7 => :priorRewardRightBranch)
    df.endStateBranchPriorReward = [df[i, rewardBranchMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.endStateBranchPriorRewardₜ₋₁ = lag(df[!, :endStateBranchPriorReward])
    df.endStateBranchPriorRewardₜ₋₂ = lag(df[!, :endStateBranchPriorRewardₜ₋₁])

    # Last Traversal Boat Reward
    # During our last traversal at boats 1-4, what was the outcome?
    priorTraversalBoatRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 4)
    priorTraversalBoatRewardVec = Vector{Union{Missing, Float64}}(missing, 4)
    for (i, ep) in enumerate(episode_record)
        if ep.S[1] == 1
            ind = ep.S[end] - 7
            priorTraversalBoatRewardVec[ind] = ep.R[end]
        end
        priorTraversalBoatRewardMat[i, :] = priorTraversalBoatRewardVec[:]
    end
    df.priorTraversalRewardAtBoat1 = lag(priorTraversalBoatRewardMat[:, 1])
    df.priorTraversalRewardAtBoat2 = lag(priorTraversalBoatRewardMat[:, 2])
    df.priorTraversalRewardAtBoat3 = lag(priorTraversalBoatRewardMat[:, 3])
    df.priorTraversalRewardAtBoat4 = lag(priorTraversalBoatRewardMat[:, 4])
    priorTraversalBoatRewardMap = Dict(
                     4 => :priorTraversalRewardAtBoat1,
                     5 => :priorTraversalRewardAtBoat2,
                     6 => :priorTraversalRewardAtBoat3,
                     7 => :priorTraversalRewardAtBoat4)
    df.priorTraversalRewardAtBoat = [df[i, priorTraversalBoatRewardMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.priorTraversalRewardAtBoat_lag1 = lag(df[!, :priorTraversalRewardAtBoat])

    # Last Traversal Island Reward
    # During our last traversal at island 1/2, what was the outcome?
    prior1TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior2TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior3TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior4TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior5TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior6TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior7TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior8TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior9TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior10TraversalIslandRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 3)
    prior1TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior2TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior3TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior4TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior5TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior6TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior7TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior8TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior9TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    prior10TraversalIslandRewardVec = Vector{Union{Missing, Float64}}(missing, 3)
    for (i, ep) in enumerate(episode_record)
        if ep.S[1] == 1
            if ep.S[2] == 2
                prior10TraversalIslandRewardVec[1] = prior9TraversalIslandRewardVec[1]
                prior9TraversalIslandRewardVec[1] = prior8TraversalIslandRewardVec[1]
                prior8TraversalIslandRewardVec[1] = prior7TraversalIslandRewardVec[1]
                prior7TraversalIslandRewardVec[1] = prior6TraversalIslandRewardVec[1]
                prior6TraversalIslandRewardVec[1] = prior5TraversalIslandRewardVec[1]
                prior5TraversalIslandRewardVec[1] = prior4TraversalIslandRewardVec[1]
                prior4TraversalIslandRewardVec[1] = prior3TraversalIslandRewardVec[1]
                prior3TraversalIslandRewardVec[1] = prior2TraversalIslandRewardVec[1]
                prior2TraversalIslandRewardVec[1] = prior1TraversalIslandRewardVec[1]
                prior1TraversalIslandRewardVec[1] = ep.R[end]
            elseif ep.S[2] == 3
                prior10TraversalIslandRewardVec[2] = prior9TraversalIslandRewardVec[2]
                prior9TraversalIslandRewardVec[2] = prior8TraversalIslandRewardVec[2]
                prior8TraversalIslandRewardVec[2] = prior7TraversalIslandRewardVec[2]
                prior7TraversalIslandRewardVec[2] = prior6TraversalIslandRewardVec[2]
                prior6TraversalIslandRewardVec[2] = prior5TraversalIslandRewardVec[2]
                prior5TraversalIslandRewardVec[2] = prior4TraversalIslandRewardVec[2]
                prior4TraversalIslandRewardVec[2] = prior3TraversalIslandRewardVec[2]
                prior3TraversalIslandRewardVec[2] = prior2TraversalIslandRewardVec[2]
                prior2TraversalIslandRewardVec[2] = prior1TraversalIslandRewardVec[2]
                prior1TraversalIslandRewardVec[2] = ep.R[end]
            end
            prior10TraversalIslandRewardVec[3] = prior9TraversalIslandRewardVec[3]
            prior9TraversalIslandRewardVec[3] = prior8TraversalIslandRewardVec[3]
            prior8TraversalIslandRewardVec[3] = prior7TraversalIslandRewardVec[3]
            prior7TraversalIslandRewardVec[3] = prior6TraversalIslandRewardVec[3]
            prior6TraversalIslandRewardVec[3] = prior5TraversalIslandRewardVec[3]
            prior5TraversalIslandRewardVec[3] = prior4TraversalIslandRewardVec[3]
            prior4TraversalIslandRewardVec[3] = prior3TraversalIslandRewardVec[3]
            prior3TraversalIslandRewardVec[3] = prior2TraversalIslandRewardVec[3]
            prior2TraversalIslandRewardVec[3] = prior1TraversalIslandRewardVec[3]
            prior1TraversalIslandRewardVec[3] = ep.R[end]
        end
        prior1TraversalIslandRewardMat[i, :] = prior1TraversalIslandRewardVec[:]
        prior2TraversalIslandRewardMat[i, :] = prior2TraversalIslandRewardVec[:]
        prior3TraversalIslandRewardMat[i, :] = prior3TraversalIslandRewardVec[:]
        prior4TraversalIslandRewardMat[i, :] = prior4TraversalIslandRewardVec[:]
        prior5TraversalIslandRewardMat[i, :] = prior5TraversalIslandRewardVec[:]
        prior6TraversalIslandRewardMat[i, :] = prior6TraversalIslandRewardVec[:]
        prior7TraversalIslandRewardMat[i, :] = prior7TraversalIslandRewardVec[:]
        prior8TraversalIslandRewardMat[i, :] = prior8TraversalIslandRewardVec[:]
        prior9TraversalIslandRewardMat[i, :] = prior9TraversalIslandRewardVec[:]
        prior10TraversalIslandRewardMat[i, :] = prior10TraversalIslandRewardVec[:]
    end
    # If at trial 10, priorRewardAtX contains the last update including trial 9
    df.prior1TraversalRewardAtIsland1 = lag(prior1TraversalIslandRewardMat[:, 1])
    df.prior2TraversalRewardAtIsland1 = lag(prior2TraversalIslandRewardMat[:, 1])
    df.prior3TraversalRewardAtIsland1 = lag(prior3TraversalIslandRewardMat[:, 1])
    df.prior4TraversalRewardAtIsland1 = lag(prior4TraversalIslandRewardMat[:, 1])
    df.prior5TraversalRewardAtIsland1 = lag(prior5TraversalIslandRewardMat[:, 1])
    df.prior6TraversalRewardAtIsland1 = lag(prior6TraversalIslandRewardMat[:, 1])
    df.prior7TraversalRewardAtIsland1 = lag(prior7TraversalIslandRewardMat[:, 1])
    df.prior8TraversalRewardAtIsland1 = lag(prior8TraversalIslandRewardMat[:, 1])
    df.prior9TraversalRewardAtIsland1 = lag(prior9TraversalIslandRewardMat[:, 1])
    df.prior10TraversalRewardAtIsland1 = lag(prior10TraversalIslandRewardMat[:, 1])
    df.prior1TraversalRewardAtIsland2 = lag(prior1TraversalIslandRewardMat[:, 2])
    df.prior2TraversalRewardAtIsland2 = lag(prior2TraversalIslandRewardMat[:, 2])
    df.prior3TraversalRewardAtIsland2 = lag(prior3TraversalIslandRewardMat[:, 2])
    df.prior4TraversalRewardAtIsland2 = lag(prior4TraversalIslandRewardMat[:, 2])
    df.prior5TraversalRewardAtIsland2 = lag(prior5TraversalIslandRewardMat[:, 2])
    df.prior6TraversalRewardAtIsland2 = lag(prior6TraversalIslandRewardMat[:, 2])
    df.prior7TraversalRewardAtIsland2 = lag(prior7TraversalIslandRewardMat[:, 2])
    df.prior8TraversalRewardAtIsland2 = lag(prior8TraversalIslandRewardMat[:, 2])
    df.prior9TraversalRewardAtIsland2 = lag(prior9TraversalIslandRewardMat[:, 2])
    df.prior10TraversalRewardAtIsland2 = lag(prior10TraversalIslandRewardMat[:, 2])

    df.prior1TraversalReward = lag(prior1TraversalIslandRewardMat[:, 3])
    priorTraversalIslandRewardMap = Dict(2 => :prior1TraversalRewardAtIsland1,
                     3 => :prior1TraversalRewardAtIsland2,
                     4 => :prior1TraversalRewardAtIsland1,
                     5 => :prior1TraversalRewardAtIsland1,
                     6 => :prior1TraversalRewardAtIsland2,
                     7 => :prior1TraversalRewardAtIsland2)
    df.prior1TraversalRewardAtIsland = [df[i, priorTraversalIslandRewardMap[df[i, :endState]]] for i in 1:nrow(df)]

    df.prior1TraversalRewardAtIsland1_lag1 = lag(df[!, :prior1TraversalRewardAtIsland1])
    df.prior1TraversalRewardAtIsland2_lag1 = lag(df[!, :prior1TraversalRewardAtIsland2])
    df.prior1TraversalRewardAtIsland1_lag1_choiceReg = (df[!, :endBranchLeftₜ₋₁] .& (df[!, :prior1TraversalRewardAtIsland1_lag1] .== 1)) .| (.!df[!, :endBranchLeftₜ₋₁] .& (df[!, :prior1TraversalRewardAtIsland1_lag1] .< 1))
    df.prior1TraversalRewardAtIsland2_lag1_choiceReg = (.!df[!, :endBranchLeftₜ₋₁] .& (df[!, :prior1TraversalRewardAtIsland2_lag1] .== 1)) .| (df[!, :endBranchLeftₜ₋₁] .& (df[!, :prior1TraversalRewardAtIsland2_lag1] .< 1))
    df.prior2TraversalRewardAtIsland1_lag1 = lag(df[!, :prior2TraversalRewardAtIsland1])
    df.prior2TraversalRewardAtIsland2_lag1 = lag(df[!, :prior2TraversalRewardAtIsland2])
    df.prior2TraversalRewardAtIsland1_lag1_choiceReg = (df[!, :endBranchLeftₜ₋₁] .& (df[!, :prior2TraversalRewardAtIsland1_lag1] .== 1)) .| (.!df[!, :endBranchLeftₜ₋₁] .& (df[!, :prior2TraversalRewardAtIsland1_lag1] .< 1))
    df.prior2TraversalRewardAtIsland2_lag1_choiceReg = (.!df[!, :endBranchLeftₜ₋₁] .& (df[!, :prior2TraversalRewardAtIsland2_lag1] .== 1)) .| (df[!, :endBranchLeftₜ₋₁] .& (df[!, :prior2TraversalRewardAtIsland2_lag1] .< 1))


    # Choice Regs below?
    df.prior1TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior1TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior1TraversalRewardAtIsland1] .< 1))
    df.prior2TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior2TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior2TraversalRewardAtIsland1] .< 1))
    df.prior3TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior3TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior3TraversalRewardAtIsland1] .< 1))
    df.prior4TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior4TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior4TraversalRewardAtIsland1] .< 1))
    df.prior5TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior5TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior5TraversalRewardAtIsland1] .< 1))
    df.prior6TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior6TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior6TraversalRewardAtIsland1] .< 1))
    df.prior7TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior7TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior7TraversalRewardAtIsland1] .< 1))
    df.prior8TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior8TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior8TraversalRewardAtIsland1] .< 1))
    df.prior9TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior9TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior9TraversalRewardAtIsland1] .< 1))
    df.prior10TraversalRewardAtIsland1_choiceReg = (df[!, :endBranchLeft] .& (df[!, :prior10TraversalRewardAtIsland1] .== 1)) .| (.!df[!, :endBranchLeft] .& (df[!, :prior10TraversalRewardAtIsland1] .< 1))
    df.prior1TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior1TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior1TraversalRewardAtIsland2] .< 1))
    df.prior2TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior2TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior2TraversalRewardAtIsland2] .< 1))
    df.prior3TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior3TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior3TraversalRewardAtIsland2] .< 1))
    df.prior4TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior4TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior4TraversalRewardAtIsland2] .< 1))
    df.prior5TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior5TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior5TraversalRewardAtIsland2] .< 1))
    df.prior6TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior6TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior6TraversalRewardAtIsland2] .< 1))
    df.prior7TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior7TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior7TraversalRewardAtIsland2] .< 1))
    df.prior8TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior8TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior8TraversalRewardAtIsland2] .< 1))
    df.prior9TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior9TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior9TraversalRewardAtIsland2] .< 1))
    df.prior10TraversalRewardAtIsland2_choiceReg = (.!df[!, :endBranchLeft] .& (df[!, :prior10TraversalRewardAtIsland2] .== 1)) .| (df[!, :endBranchLeft] .& (df[!, :prior10TraversalRewardAtIsland2] .< 1))

    prior1TraversalIslandReward_choiceReg_Map = Dict(true => :prior1TraversalRewardAtIsland1_choiceReg, false => :prior1TraversalRewardAtIsland2_choiceReg)
    prior2TraversalIslandReward_choiceReg_Map = Dict(true => :prior2TraversalRewardAtIsland1_choiceReg, false => :prior2TraversalRewardAtIsland2_choiceReg)
    prior3TraversalIslandReward_choiceReg_Map = Dict(true => :prior3TraversalRewardAtIsland1_choiceReg, false => :prior3TraversalRewardAtIsland2_choiceReg)
    prior4TraversalIslandReward_choiceReg_Map = Dict(true => :prior4TraversalRewardAtIsland1_choiceReg, false => :prior4TraversalRewardAtIsland2_choiceReg)
    prior5TraversalIslandReward_choiceReg_Map = Dict(true => :prior5TraversalRewardAtIsland1_choiceReg, false => :prior5TraversalRewardAtIsland2_choiceReg)
    prior6TraversalIslandReward_choiceReg_Map = Dict(true => :prior6TraversalRewardAtIsland1_choiceReg, false => :prior6TraversalRewardAtIsland2_choiceReg)
    prior7TraversalIslandReward_choiceReg_Map = Dict(true => :prior7TraversalRewardAtIsland1_choiceReg, false => :prior7TraversalRewardAtIsland2_choiceReg)
    prior8TraversalIslandReward_choiceReg_Map = Dict(true => :prior8TraversalRewardAtIsland1_choiceReg, false => :prior8TraversalRewardAtIsland2_choiceReg)
    prior9TraversalIslandReward_choiceReg_Map = Dict(true => :prior9TraversalRewardAtIsland1_choiceReg, false => :prior9TraversalRewardAtIsland2_choiceReg)
    prior10TraversalIslandReward_choiceReg_Map = Dict(true => :prior10TraversalRewardAtIsland1_choiceReg, false => :prior10TraversalRewardAtIsland2_choiceReg)
    prior1TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior1TraversalRewardAtIsland2_choiceReg, false => :prior1TraversalRewardAtIsland1_choiceReg)
    prior2TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior2TraversalRewardAtIsland2_choiceReg, false => :prior2TraversalRewardAtIsland1_choiceReg)
    prior3TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior3TraversalRewardAtIsland2_choiceReg, false => :prior3TraversalRewardAtIsland1_choiceReg)
    prior4TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior4TraversalRewardAtIsland2_choiceReg, false => :prior4TraversalRewardAtIsland1_choiceReg)
    prior5TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior5TraversalRewardAtIsland2_choiceReg, false => :prior5TraversalRewardAtIsland1_choiceReg)
    prior6TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior6TraversalRewardAtIsland2_choiceReg, false => :prior6TraversalRewardAtIsland1_choiceReg)
    prior7TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior7TraversalRewardAtIsland2_choiceReg, false => :prior7TraversalRewardAtIsland1_choiceReg)
    prior8TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior8TraversalRewardAtIsland2_choiceReg, false => :prior8TraversalRewardAtIsland1_choiceReg)
    prior9TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior9TraversalRewardAtIsland2_choiceReg, false => :prior9TraversalRewardAtIsland1_choiceReg)
    prior10TraversalOppIslandReward_choiceReg_Map = Dict(true => :prior10TraversalRewardAtIsland2_choiceReg, false => :prior10TraversalRewardAtIsland1_choiceReg)

    df.prior1TraversalRewardAtIsland_choiceReg = [df[i, prior1TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior2TraversalRewardAtIsland_choiceReg = [df[i, prior2TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior3TraversalRewardAtIsland_choiceReg = [df[i, prior3TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior4TraversalRewardAtIsland_choiceReg = [df[i, prior4TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior5TraversalRewardAtIsland_choiceReg = [df[i, prior5TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior6TraversalRewardAtIsland_choiceReg = [df[i, prior6TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior7TraversalRewardAtIsland_choiceReg = [df[i, prior7TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior8TraversalRewardAtIsland_choiceReg = [df[i, prior8TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior9TraversalRewardAtIsland_choiceReg = [df[i, prior9TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior10TraversalRewardAtIsland_choiceReg = [df[i, prior10TraversalIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior1TraversalRewardAtOppIsland_choiceReg = [df[i, prior1TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior2TraversalRewardAtOppIsland_choiceReg = [df[i, prior2TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior3TraversalRewardAtOppIsland_choiceReg = [df[i, prior3TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior4TraversalRewardAtOppIsland_choiceReg = [df[i, prior4TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior5TraversalRewardAtOppIsland_choiceReg = [df[i, prior5TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior6TraversalRewardAtOppIsland_choiceReg = [df[i, prior6TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior7TraversalRewardAtOppIsland_choiceReg = [df[i, prior7TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior8TraversalRewardAtOppIsland_choiceReg = [df[i, prior8TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior9TraversalRewardAtOppIsland_choiceReg = [df[i, prior9TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]
    df.prior10TraversalRewardAtOppIsland_choiceReg = [df[i, prior10TraversalOppIslandReward_choiceReg_Map[df[i, :endBranchLeft]]] for i in 1:nrow(df)]

    df.prior1TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior1TraversalRewardAtIsland_choiceReg])
    df.prior2TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior2TraversalRewardAtIsland_choiceReg])
    df.prior3TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior3TraversalRewardAtIsland_choiceReg])
    df.prior4TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior4TraversalRewardAtIsland_choiceReg])
    df.prior5TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior5TraversalRewardAtIsland_choiceReg])
    df.prior6TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior6TraversalRewardAtIsland_choiceReg])
    df.prior7TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior7TraversalRewardAtIsland_choiceReg])
    df.prior8TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior8TraversalRewardAtIsland_choiceReg])
    df.prior9TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior9TraversalRewardAtIsland_choiceReg])
    df.prior10TraversalRewardAtIsland_choiceReg_lag1 = lag(df[!, :prior10TraversalRewardAtIsland_choiceReg])
    df.prior1TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior1TraversalRewardAtOppIsland_choiceReg])
    df.prior2TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior2TraversalRewardAtOppIsland_choiceReg])
    df.prior3TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior3TraversalRewardAtOppIsland_choiceReg])
    df.prior4TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior4TraversalRewardAtOppIsland_choiceReg])
    df.prior5TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior5TraversalRewardAtOppIsland_choiceReg])
    df.prior6TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior6TraversalRewardAtOppIsland_choiceReg])
    df.prior7TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior7TraversalRewardAtOppIsland_choiceReg])
    df.prior8TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior8TraversalRewardAtOppIsland_choiceReg])
    df.prior9TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior9TraversalRewardAtOppIsland_choiceReg])
    df.prior10TraversalRewardAtOppIsland_choiceReg_lag1 = lag(df[!, :prior10TraversalRewardAtOppIsland_choiceReg])

    # Logic - Keep track of the most recent reward from traversals
    # This should be updated on every trial
    #
    # Now on a boat-only trial, this doesn't get updated, so we can just look 1 trial back
    # That is, should be the lag of 1 on 

    # On each traversal, we end up at boat x.
    # What was the previously-observed reward at boat x?
    # df.traversalPriorRewardAtBoat

    # Last Traversal Island Reward
    # During our last traversal at island 1/2, what was the outcome?
    df[!, :td0Island1Reg] = Vector{Union{Missing, Float64}}(missing, length(episode_record))
    df[!, :td0Island2Reg] = Vector{Union{Missing, Float64}}(missing, length(episode_record))
    prior1TraversalIsland1BoatReward = missing
    prior1TraversalIsland2BoatReward = missing
    for i in 1:nrow(df)
        if df[i, :state1] == 1
            if df[i, :state2] == 2
                prior1TraversalIsland1BoatReward = df[i, :endStatePriorReward]
            else
                prior1TraversalIsland2BoatReward = df[i, :endStatePriorReward]
            end
        end
        df[i, :td0Island1Reg] = prior1TraversalIsland1BoatReward 
        df[i, :td0Island2Reg] = prior1TraversalIsland2BoatReward 
    end
    # prior1TraversalIslandBoatRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 2)
    # prior1TraversalIslandBoatRewardVec = Vector{Union{Missing, Float64}}(missing, 2)
    # prior2TraversalIslandBoatRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 2)
    # prior2TraversalIslandBoatRewardVec = Vector{Union{Missing, Float64}}(missing, 2)
    # for (i, ep) in enumerate(episode_record)
    #     if ep.S[1] == 1
    #         if ep.S[2] == 2
    #             prior2TraversalIslandRewardVec[1] = prior1TraversalIslandRewardVec[1]
    #             prior1TraversalIslandRewardVec[1] = ep.R[end]
    #         elseif ep.S[2] == 3
    #             prior2TraversalIslandRewardVec[2] = prior1TraversalIslandRewardVec[2]
    #             prior1TraversalIslandRewardVec[2] = ep.R[end]
    #         end
    #         prior2TraversalIslandRewardVec[3] = prior1TraversalIslandRewardVec[3]
    #         prior1TraversalIslandRewardVec[3] = ep.R[end]
    #     end
    #     prior1TraversalIslandRewardMat[i, :] = prior1TraversalIslandRewardVec[:]
    #     prior2TraversalIslandRewardMat[i, :] = prior2TraversalIslandRewardVec[:]
    # end
    # df.endStatePriorReward on a traversal trial

    # Last Sampled Reward
    # priorSampledRewardMat = Matrix{Union{Missing, Int}}(missing, length(episode_record), 4)
    # priorSampledRewardVec = Vector{Union{Missing, Int}}(missing, 4)
    # for (i, ep) in enumerate(episode_record)
    #     if ep.S[1] > 3
    #         ind = ep.S[end] - 7
    #         priorSampledRewardVec[ind] = ep.R[end]
    #     end
    #     priorSampledRewardMat[i, :] = recentSampledRewardVec[:]
    # end
    # df.priorSampledRewardAt4 = lag(recentSampledRewardMat[:, 1])
    # df.priorSampledRewardAt5 = lag(recentSampledRewardMat[:, 2])
    # df.priorSampledRewardAt6 = lag(recentSampledRewardMat[:, 3])
    # df.priorSampledRewardAt7 = lag(recentSampledRewardMat[:, 4])
    # sampledRewardSiblingMap = Dict(4 => :priorSampledRewardAt5,
    #                                5 => :priorSampledRewardAt4,
    #                                6 => :priorSampledRewardAt7,
    #                                7 => :priorSampledRewardAt6)
    # df.endSiblingPriorSampledReward = [df[i, rewardSiblingMap[df[i, :endState]]] for i in 1:nrow(df)]
    # df.endSiblingPriorSampledRewardₜ₋₁ = lag(df[!, :endSiblingRecentSampledReward])
   
    # df[!, :prior1TraversalRewardAtIsland_choiceReg_lag1_new] .= false
    # df[!, :prior2TraversalRewardAtIsland_choiceReg_lag1_new] .= false
    # trav1backleft = false
    # trav2backleft = false
    # trav1backright = false
    # trav2backright = false
    # for (i, ep) in enumerate(episode_record)
    #     if ep.S[1] == 1
    #         if ep.S[2] == 2
    #             trav2backleft = trav1backleft
    #             trav1backleft = ep.R[end] > 0
    #         else
    #             trav2backright = trav1backright
    #             trav1backright = ep.R[end] > 0
    #         end
    #     end
    #     if df[i, :endBranchLeftₜ₋₁] == true
    #         df[i, :prior1TraversalRewardAtIsland_choiceReg_lag1_new] = trav1backleft
    #         df[i, :prior1TraversalRewardAtOppIsland_choiceReg_lag1_new] = trav1backright
    #     else
    #         df[i, :prior1TraversalRewardAtIsland_choiceReg_lag1_new] = trav1backright
    #         df[i, :prior1TraversalRewardAtOppIsland_choiceReg_lag1_new] = trav1backleft
    #     end
    #     prior1TraversalIslandRewardMat[i, :] = prior1TraversalIslandRewardVec[:]
    #     prior2TraversalIslandRewardMat[i, :] = prior2TraversalIslandRewardVec[:]
    # end

    df
end


function RunToDataFrameBig(episode_record::Vector{Episode}; subject=0)
    #      1
    #  2       3
    # 4 5     6 7
    # 8 9    10 11
    df = DataFrame()

    df.state1 = [ep.S[1] for ep in episode_record]
    df.state2 = [ep.S[2] < 8 ? ep.S[2] : missing for ep in episode_record]
    df.state3 = [length(ep.S) > 2 && ep.S[3] < 8 ? ep.S[3] : missing for ep in episode_record]
    df.state4 = [length(ep.S) > 3 && ep.S[3] < 16 ? ep.S[4] : missing for ep in episode_record]

    # endState is 2-7 (2/3 for island-only trials, 4-7 for other trials)
    df.endState = [ep.S[end] > 3 ? ep.S[end-1] : ep.S[end] for ep in episode_record]
    df.reward = [ep.R[end] for ep in episode_record]

    df.trial = 1:nrow(df)
    df[!, :subject] .= string(subject)

    df
end
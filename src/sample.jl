function sample!(sol, tree)
    empty!(tree.sampled)
    L = tree.V_lower[1]
    U = L + sol.epsilon*root_diff(tree)
    sample_points(sol, tree, 1, L, U, 0, sol.epsilon*root_diff(tree))
end

function sample_points(sol::SARSOPSolver, tree::SARSOPTree, b_idx::Int, L, U, t, ϵ)
    tree.b_pruned[b_idx] = false
    if !tree.is_real[b_idx]
        tree.is_real[b_idx] = true
        push!(tree.real, b_idx)
    end

    tree.is_terminal[b_idx] && return

    fill_belief!(tree, b_idx)
    V̲, V̄ = tree.V_lower[b_idx], tree.V_upper[b_idx]
    γ = discount(tree)

    V̂ = V̄ #TODO: BAD, binning method
    if V̂ ≤ V̲ + sol.kappa*ϵ*γ^(-t) || (V̂ ≤ L && V̄ ≤ max(U, V̲ + ϵ*γ^(-t)))
        return
    else
        Q̲, Q̄, a′ = max_r_and_q(tree, b_idx)
        Δt = tree.pomdp.Δts[a′]
        ba_idx = tree.b_children[b_idx][a′] #line 10
        tree.ba_pruned[ba_idx] = false

        Rba′ = belief_reward(tree, tree.b[b_idx], a′)

        L′ = max(L, Q̲)
        U′ = max(U, Q̲ + γ^(-t)*ϵ)

        op_idx = best_obs(tree, b_idx, ba_idx, ϵ, t+Δt)
        Lt, Ut = get_LtUt(tree, ba_idx, Rba′, L′, U′, op_idx, Δt)

        bp_idx = tree.ba_children[ba_idx][op_idx]
        push!(tree.sampled, b_idx)
        sample_points(sol, tree, bp_idx, Lt, Ut, t+Δt, ϵ)
    end
end

belief_reward(tree, b, a) = dot(@view(tree.pomdp.R[:,a]), b)

function max_r_and_q(tree::SARSOPTree, b_idx::Int)
    Q̄′s = [tree.Qa_upper[b_idx][i] for i in 1:length(tree.b_children[b_idx])]
    Q̲′s = [tree.Qa_lower[b_idx][i] for i in 1:length(tree.b_children[b_idx])]

    expQs = exp.(Q̄′s)
    probs = expQs ./ sum(expQs)
    a′ = rand(Categorical(probs))
    return Q̲′s[a′], Q̄′s[a′], a′

    # Q̲ = -Inf
    # Q̄ = -Inf
    # a′ = 0
    # for (i,ba_idx) in enumerate(tree.b_children[b_idx])
        
    #     Q̄′ = tree.Qa_upper[b_idx][i]
    #     Q̲′ = tree.Qa_lower[b_idx][i]
    #     # println("action $i, upper: $(Q̄′), lower: $(Q̲′)")
    #     if Q̲′ > Q̲
    #         Q̲ = Q̲′
    #     end
    #     if Q̄′ > Q̄
    #         Q̄ = Q̄′
    #         a′ = i
    #     end
    # end
    # return Q̲, Q̄, a′
end

function best_obs(tree::SARSOPTree, b_idx, ba_idx, ϵ, t)
    S = states(tree)
    O = observations(tree)
    γ = discount(tree)

    best_o = 0
    best_gap = -Inf

    for o in O
        poba = tree.poba[ba_idx][o]
        bp_idx = tree.ba_children[ba_idx][o]
        gap = poba*(tree.V_upper[bp_idx] - tree.V_lower[bp_idx] - ϵ*γ^(-(t)))
        if gap > best_gap
            best_gap = gap
            best_o = o
        end
    end
    return best_o
end

obs_prob(tree::SARSOPTree, ba_idx::Int, o_idx::Int) = tree.poba[ba_idx][o_idx]

function get_LtUt(tree, ba_idx, Rba, L′, U′, o′, Δt)
    γ = discount(tree)
    Lt = (L′ - Rba)/γ^Δt
    Ut = (U′ - Rba)/γ^Δt

    for o in observations(tree)
        if o′ != o
            bp_idx = tree.ba_children[ba_idx][o]
            V̲ = tree.V_lower[bp_idx]
            V̄ = tree.V_upper[bp_idx]
            poba = obs_prob(tree, ba_idx, o)
            Lt -= poba*V̲
            Ut -= poba*V̄
        end
    end
    poba = obs_prob(tree, ba_idx, o′)
    return Lt / poba, Ut / poba
end

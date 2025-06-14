using Optimisers
using Zygote
using Statistics
using Random, Distributions

include("./networks.jl")

# Convert state to one-hot encoding for policy input
function state_to_onehot(state::Vector{Int}, num_amino_acids::Int)
    n = length(state)
    return reduce(hcat, map(j -> [i == state[j] ? 1.0f0 : 0.0f0 for i in 1:num_amino_acids], 1:n))
end

# Sample action from policy
function sample_action(policy, state, θ, ξ, rng)
    state_input = state_to_onehot(state, length(AMINO_ACIDS))
    logits, _ = policy(state_input, θ, ξ)
    action = zeros(Int, length(state))
    for i in 1:length(state)
        # @debug "Logits: $(logits[:, i])"
        logits_stable = logits[:, i] .- maximum(logits[:, i])  # subtract the maximum to prevent overflow
        probs = softmax(logits_stable)
        action[i] = rand(rng, Categorical(probs))
    end
    return action
end

# Compute log probability of an action
function log_prob(policy, state, action, θ, ξ)
    state_input = state_to_onehot(state, length(AMINO_ACIDS))
    logits, _ = policy(state_input, θ, ξ)
    logp = 0.0
    for i in eachindex(action)
        probs = softmax(logits[:, i])
        logp += log(probs[action[i]] + 1e-8)  # Add small ε to avoid log(0)
    end
    return logp
end

# Collect trajectories from the environment
function collect_trajectories(env, actor, θ, ξ, rng, timesteps_for_epoch)
    states = Vector{Vector{Int}}()
    actions = Vector{Vector{Int}}()
    rewards = Vector{Float32}()
    next_states = Vector{Vector{Int}}()
    dones = Vector{Bool}()

    state = reset!(env)
    for t in 1:timesteps_for_epoch
        action = sample_action(actor, state, θ, ξ, rng)
        next_state, reward, done = step!(env, action)

        push!(states, copy(state))
        push!(actions, copy(action))
        push!(rewards, reward)
        push!(next_states, copy(next_state))
        push!(dones, done)

        state = next_state
        if done
            state = reset!(env)
        end
    end
    return states, actions, rewards, next_states, dones
end

# Compute advantages and returns using GAE
function gae(states, next_states, rewards, dones, critic, ϕ, ζ, timesteps_for_epoch, gamma, gae_lambda)
    A_hat = zeros(Float32, timesteps_for_epoch)
    R_hat = zeros(Float32, timesteps_for_epoch)
    num_amino_acids = length(AMINO_ACIDS)

    for t in timesteps_for_epoch:-1:1
        state_input = state_to_onehot(states[t], num_amino_acids)
        next_state_input = state_to_onehot(next_states[t], num_amino_acids)

        v_t, _ = critic(state_input, ϕ, ζ)
        v_t_next, _ = dones[t] ? (0.0f0, nothing) : critic(next_state_input, ϕ, ζ)

        δ_t = rewards[t] + gamma * v_t_next - v_t
        A_hat[t] = δ_t + (t < timesteps_for_epoch ? gamma * gae_lambda * A_hat[t+1] : 0.0f0)
        R_hat[t] = A_hat[t] + v_t
    end

    return A_hat, R_hat
end

function clip_gradients(grads, clip_value=1.0)
    if grads isa Nothing
        return grads
    elseif grads isa AbstractArray
        return clamp.(grads, -clip_value, clip_value)
    elseif grads isa NamedTuple
        return map(g -> clip_gradients(g, clip_value), grads)
    else
        return grads
    end
end

# Optimize policy and value networks for a single minibatch
function optimize_networks(state, action, advantage, return_value, actor, θ, ξ, critic, ϕ, ζ, actor_optimizer_state, critic_optimizer_state, clip_ratio)
    # Compute old log probability (before update)
    log_prob_old = log_prob(actor, state, action, θ, ξ)

    # Policy loss
    function policy_clip_loss(θ_new)
        log_prob_new = log_prob(actor, state, action, θ_new, ξ)
        r_θ = exp(clamp(log_prob_new - log_prob_old, -10, 10))  # compute probability ratio, clamp to prevent explosion
        clipped = clamp(r_θ, 1 - clip_ratio, 1 + clip_ratio) * advantage
        return -min(r_θ * advantage, clipped)
    end

    # Update policy network weights
    # @debug "Policy Weights: $(θ)"
    loss_θ, grads_θ = Zygote.withgradient(θ -> policy_clip_loss(θ), θ)
    # @debug "Policy Gradients: $(grads_θ[1])"
    grads_θ_clipped = clip_gradients(grads_θ[1], 2.0)
    actor_optimizer_state, θ = Optimisers.update(actor_optimizer_state, θ, grads_θ_clipped)

    # Value loss
    function value_loss(ϕ_new)
        state_input = state_to_onehot(state, length(AMINO_ACIDS))
        v, _ = critic(state_input, ϕ_new, ζ)
        return (v - return_value)^2
    end

    # Update value network weights
    loss_ϕ, grads_ϕ = Zygote.withgradient(ϕ -> value_loss(ϕ), ϕ)
    grads_ϕ_clipped = clip_gradients(grads_ϕ[1], 2.0)
    critic_optimizer_state, ϕ = Optimisers.update(critic_optimizer_state, ϕ, grads_ϕ_clipped)

    return actor_optimizer_state, θ, critic_optimizer_state, ϕ, loss_θ, loss_ϕ
end

function get_minibatch_indices(minibatch_idx, minibatch_size, timesteps_for_epoch, shuffled_indices)
    start_idx = (minibatch_idx - 1) * minibatch_size + 1
    end_idx = min(minibatch_idx * minibatch_size, timesteps_for_epoch)
    return shuffled_indices[start_idx:end_idx]
end

# PPO Training Function with Shuffled Minibatches
function train_ppo(; 
    n_iterations::Int=16, 
    timesteps_for_epoch::Int=16, 
    epochs::Int=16, 
    minibatch_size::Int=4, 
    clip_ratio::Float32=0.2f0, 
    policy_lr::Float32=1.0f-3, 
    value_lr::Float32=1.0f-2, 
    gamma::Float32=0.99f0, 
    gae_lambda::Float32=0.95f0
)
    # Initialize environment
    env = PeptideSequenceEnv()
    n = env.n
    num_amino_acids = length(AMINO_ACIDS)

    # Initialize policy and value networks
    rng = Random.default_rng()
    actor = create_policy_network(n, num_amino_acids)
    critic = create_value_network(n, num_amino_acids)
    θ, ξ = Lux.setup(rng, actor)
    ϕ, ζ = Lux.setup(rng, critic)

    # Initialize optimizers
    actor_optimizer = Adam(policy_lr)
    critic_optimizer = Adam(value_lr)
    actor_optimizer_state = Optimisers.setup(actor_optimizer, θ)
    critic_optimizer_state = Optimisers.setup(critic_optimizer, ϕ)

    for iteration in 1:n_iterations
        # (a) Collect trajectories
        states, actions, rewards, next_states, dones = collect_trajectories(env, actor, θ, ξ, rng, timesteps_for_epoch)

        # Compute average reward for this iteration
        avg_reward = round(mean(rewards), digits=2)

        # (b) Compute advantages and returns using GAE
        A_hat, R_hat = gae(states, next_states, rewards, dones, critic, ϕ, ζ, timesteps_for_epoch, gamma, gae_lambda)

        # (c) Normalize advantages
        A_hat .= (A_hat .- mean(A_hat)) ./ (std(A_hat) .+ 1e-8)
        
        # (d) Optimize Surrogate Objective with shuffled minibatches
        shuffled_indices = shuffle(1:timesteps_for_epoch)
        num_minibatches = ceil(Int, timesteps_for_epoch / minibatch_size)
        
        # Track losses over the iteration
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_updates = 0

        for epoch_idx in 1:epochs
            for minibatch_idx in 1:num_minibatches
                minibatch_indices = get_minibatch_indices(minibatch_idx, minibatch_size, timesteps_for_epoch, shuffled_indices)

                for i in minibatch_indices
                    # Update networks and collect losses
                    actor_optimizer_state, θ, critic_optimizer_state, ϕ, policy_loss, value_loss = optimize_networks(
                        states[i], actions[i], A_hat[i], R_hat[i], actor, θ, ξ, critic, ϕ, ζ, 
                        actor_optimizer_state, critic_optimizer_state, clip_ratio
                    )
                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                    num_updates += 1
                end
            end  # (minibatch)
        end  # (epoch)

        # Compute average losses
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates

        # Print iteration progress with tracked metrics
        println("Iteration $iteration/$n_iterations | Avg Reward: $avg_reward | Avg Policy Loss: $avg_policy_loss | Avg Value Loss: $avg_value_loss")

    end  # (iteration)

    return actor, critic, θ, ξ, ϕ, ζ
end
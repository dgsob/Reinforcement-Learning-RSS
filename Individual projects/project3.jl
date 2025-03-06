# Project 3: Approximation Methods and Policy Gradients

using Random
using LinearAlgebra
using Statistics
using Plots
using StatsBase

###########################################
# Gridworld with Monster Environment
###########################################
mutable struct GridWorld
    N::Int                    # Grid size (N×N)
    T::Int                    # Max episode length
    player_pos::Vector{Int}   # Player position [x, y]
    monster_pos::Vector{Int}  # Monster position [x, y]
    apple_pos::Vector{Int}    # Apple position [x, y]
    step_count::Int           # Current step count
    terminal::Bool            # If caught by monster
    
    # Constructor
    function GridWorld(N::Int, T::Int)
        return new(N, T, [0, 0], [0, 0], [0, 0], 0, false)
    end
end

# Initialize a new episode
function initialize!(env::GridWorld)
    env.step_count = 0
    env.terminal = false
    
    # Random positions for player, monster, and apple
    all_positions = [(i, j) for i in 0:(env.N-1) for j in 0:(env.N-1)]
    
    # Randomly sample three unique positions
    sampled_positions = sample(all_positions, 3, replace=false)
    
    env.player_pos = [sampled_positions[1][1], sampled_positions[1][2]]
    env.monster_pos = [sampled_positions[2][1], sampled_positions[2][2]]
    env.apple_pos = [sampled_positions[3][1], sampled_positions[3][2]]
    
    return get_state(env)
end

# Get the current state vector [x_p, y_p, x_m, y_m, x_a, y_a]
function get_state(env::GridWorld)
    return [env.player_pos[1], env.player_pos[2], 
            env.monster_pos[1], env.monster_pos[2], 
            env.apple_pos[1], env.apple_pos[2]]
end

# Take a step in the environment
function step!(env::GridWorld, action::Int)
    if env.terminal || env.step_count >= env.T
        return get_state(env), 0, true
    end
    
    # Make a move
    move_player!(env, action) # positions are 0 to N-1
    move_monster!(env)
    
    env.step_count += 1
    
    # Check if player caught by monster
    if env.player_pos == env.monster_pos
        env.terminal = true
        return get_state(env), -1, true
    end
    
    # Check if player found an apple
    reward = 0
    if env.player_pos == env.apple_pos
        reward = 1
        spawn_new_apple!(env)
    end
    
    # Check if episode is over
    done = env.terminal || env.step_count >= env.T
    
    return get_state(env), reward, done
end

# Move the player according to the action
function move_player!(env::GridWorld, action::Int)
    x, y = env.player_pos
    
    if action == 1 && x > 0  # Left
        env.player_pos[1] -= 1
    elseif action == 2 && y > 0  # Up
        env.player_pos[2] -= 1
    elseif action == 3 && x < env.N - 1  # Right
        env.player_pos[1] += 1
    elseif action == 4 && y < env.N - 1  # Down
        env.player_pos[2] += 1
    end
end

# Move the monster randomly
function move_monster!(env::GridWorld)
    x, y = env.monster_pos
    
    dir = rand(1:4)
    
    if dir == 1 && x > 0  # Left
        env.monster_pos[1] -= 1
    elseif dir == 2 && y > 0  # Up
        env.monster_pos[2] -= 1
    elseif dir == 3 && x < env.N - 1  # Right
        env.monster_pos[1] += 1
    elseif dir == 4 && y < env.N - 1  # Down
        env.monster_pos[2] += 1
    end
end

# Spawn a new apple in a random empty cell
function spawn_new_apple!(env::GridWorld)
    while true
        # Generate random position
        x = rand(0:(env.N-1))
        y = rand(0:(env.N-1))

        # Check if position is not occupied by player or monster
        if [x, y] != env.player_pos && [x, y] != env.monster_pos
            env.apple_pos = [x, y]
            return
        end
    end
end

###########################################
# Feature Functions and Policy
###########################################

# Calculate distance between two points for features
function manhattan_distance(x1, y1, x2, y2)
    return abs(x2 - x1) + abs(y2 - y1)
end

# Check if action moves player closer to a target
# TODO: N is never provided in any call of this function
function moves_closer(player_x, player_y, target_x, target_y, action, N=10)
    # Calculate current distance
    current_dist = manhattan_distance(player_x, player_y, target_x, target_y)
    
    # Calculate new position after action
    new_x, new_y = player_x, player_y
    if action == 1 && player_x > 0  # Left
        new_x -= 1
    elseif action == 2 && player_y > 0  # Up
        new_y -= 1
    elseif action == 3 && player_x < N-1  # Right
        new_x += 1
    elseif action == 4 && player_y < N-1  # Down
        new_y += 1
    end
    
    # Calculate new distance
    new_dist = manhattan_distance(new_x, new_y, target_x, target_y)
    
    # Return true if new distance is less than current distance
    return new_dist < current_dist
end

# Feature function for state-action value approximation
function state_action_features(state, action)
    player_x, player_y, monster_x, monster_y, apple_x, apple_y = state
    
    # Feature 1: Inverse distance to apple (closer = higher value)
    dist_to_apple = manhattan_distance(player_x, player_y, apple_x, apple_y)
    f1 = 1.0 / (dist_to_apple + 1.0)
    
    # Feature 2: Inverse distance to monster (closer = higher value)
    dist_to_monster = manhattan_distance(player_x, player_y, monster_x, monster_y)
    f2 = 1.0 / (dist_to_monster + 1.0)
    
    # Feature 3: 1 if action takes player closer to apple, 0 otherwise
    f3 = moves_closer(player_x, player_y, apple_x, apple_y, action) ? 1.0 : 0.0
    
    # Feature 4: 1 if action takes player closer to monster, 0 otherwise
    f4 = moves_closer(player_x, player_y, monster_x, monster_y, action) ? 1.0 : 0.0
    
    return [f1, f2, f3, f4]
end

# Feature function for state value approximation
function state_features(state)
    player_x, player_y, monster_x, monster_y, apple_x, apple_y = state
    
    # Feature 1: Inverse distance to apple
    dist_to_apple = manhattan_distance(player_x, player_y, apple_x, apple_y)
    f1 = 1.0 / (dist_to_apple + 1.0)
    
    # Feature 2: Inverse distance to monster
    dist_to_monster = manhattan_distance(player_x, player_y, monster_x, monster_y)
    f2 = 1.0 / (dist_to_monster + 1.0)
    
    # Feature 3: Can move closer to apple with any action?
    f3 = 0.0
    for a in 1:4
        if moves_closer(player_x, player_y, apple_x, apple_y, a)
            f3 = 1.0
            break
        end
    end
    
    # Feature 4: Can avoid moving closer to monster?
    f4 = 1.0
    for a in 1:4
        if !moves_closer(player_x, player_y, monster_x, monster_y, a)
            f4 = 1.0
            break
        end
    end
    
    return [f1, f2, f3, f4]
end

# Function h(s,a;θ) for softmax policy
function h(state, action, θ)
    # Linear combination of features and weights
    features = state_action_features(state, action)
    return dot(θ, features)
end

# Softmax calculation
function softmax(state, θ)
    # Calculate h(s,a,θ) for all actions
    h_values = [h(state, a, θ) for a in 1:4]
    
    # Apply softmax
    exp_values = exp.(h_values)
    probs = exp_values ./ sum(exp_values)
    
    return probs
end

# Expected features under policy π
function expected_features(state, θ)
    # E[x(s,b)] = Σ_b π(b|s)x(s,b)
    probs = softmax(state, θ)
    
    exp_feat = zeros(length(state_action_features(state, 1)))
    for a in 1:4
        exp_feat += probs[a] * state_action_features(state, a)
    end
    
    return exp_feat
end

# ε-greedy action selection based on value
function epsilon_greedy_policy(state, w, ε)
    if rand() < ε
        # Random action (exploration)
        return rand(1:4)
    else
        # Greedy action (exploitation)
        q_values = [dot(w, state_action_features(state, a)) for a in 1:4]
        return argmax(q_values)
    end
end

# Softmax action selection based on policy π(a|s;θ)
function softmax_policy(state, θ)
    probs = softmax(state, θ)
    return sample(1:4, Weights(probs))
end

###########################################
# Algorithm Implementations
###########################################

# Semi-gradient n-step SARSA (page 247 in Sutton RLI)
function semi_gradient_n_step_sarsa(env::GridWorld, n::Int, α::Float64, ε::Float64, γ::Float64, num_episodes::Int)
    # Initialize weights for action-value function
    w = 0.01 * randn(4)  # Small random weights for 4 features
    
    # For learning curve
    returns_per_episode = zeros(num_episodes)
    
    for episode in 1:num_episodes
        # Initialize state
        state = initialize!(env)
        action = epsilon_greedy_policy(state, w, ε)
        
        # Circular buffers
        S = Vector{Vector{Int}}(undef, n+1)
        A = Vector{Int}(undef, n+1)
        R = Vector{Float64}(undef, n+1)
        
        # Initialize
        for i in 1:n+1
            S[i] = copy(state)
            A[i] = action
            R[i] = 0.0
        end
        
        t = 0
        T_terminal = typemax(Int)  # represents infinity
        episode_return = 0.0
        
        # Main loop
        while true
            # Take action if we're not at terminal state
            if t < T_terminal
                # Take action A_t
                buffer_idx = mod1(t+1, n+1)
                next_state, reward, terminal = step!(env, A[buffer_idx])
                episode_return += reward
                
                # Store reward and next state
                next_buffer_idx = mod1(t+2, n+1)
                R[next_buffer_idx] = reward
                
                if terminal
                    T_terminal = t + 1
                else
                    S[next_buffer_idx] = copy(next_state)
                    A[next_buffer_idx] = epsilon_greedy_policy(next_state, w, ε)
                end
            end
            
            # State whose estimate is being updated
            τ = t - n + 1
            
            if τ >= 0
                # Calculate return
                G = 0.0
                for i in τ+1:min(τ+n, T_terminal)
                    i_idx = mod1(i+1, n+1)
                    G += γ^(i-τ-1) * R[i_idx]
                end
                
                # Add bootstrapped estimate if needed
                if τ + n < T_terminal
                    bootstrap_idx = mod1(τ+n+1, n+1)
                    bootstrap_state = S[bootstrap_idx]
                    bootstrap_action = A[bootstrap_idx]
                    G += γ^n * dot(w, state_action_features(bootstrap_state, bootstrap_action))
                end
                
                # Update weights
                τ_idx = mod1(τ+1, n+1)
                τ_state = S[τ_idx]
                τ_action = A[τ_idx]
                features = state_action_features(τ_state, τ_action)
                current_value = dot(w, features)
                w += α * (G - current_value) * features
            end
            
            # Termination check
            if τ == T_terminal - 1
                break
            end
            
            t += 1 # else we proceed
        end
        
        returns_per_episode[episode] = episode_return
    end
    
    return w, returns_per_episode
end

# REINFORCE (page 328 in Sutton RLI)
function reinforce(env::GridWorld, α::Float64, γ::Float64, num_episodes::Int)
    # Initialize policy weights
    θ = 0.01 * randn(4)  # Small random weights for 4 features
    
    # For learning curve
    returns_per_episode = zeros(num_episodes)
    
    for episode in 1:num_episodes
        # Generate an episode
        S = Vector{Vector{Int}}()
        A = Vector{Int}()
        R = Vector{Float64}()
        
        # Initialize state
        state = initialize!(env)
        push!(S, state)
        
        episode_return = 0.0
        terminal = false
        
        while !terminal
            # Select action from policy
            action = softmax_policy(state, θ)
            push!(A, action)
            
            # Take action, observe next state and reward
            next_state, reward, terminal = step!(env, action)
            push!(R, reward)
            episode_return += reward
            
            if !terminal
                push!(S, next_state)
                state = next_state
            end
        end
        
        # Policy gradient updates
        for t in 1:length(S)
            # Calculate discounted return from time t
            G = 0.0
            for k in t:length(R)
                G += γ^(k-t) * R[k]
            end
            
            # Update policy weights
            state = S[t]
            action = A[t]
            
            # Calculate x(s,a) - Σ_b π(b|s)x(s,b)
            x = state_action_features(state, action)
            exp_x = expected_features(state, θ)
            
            # θ += α * G * ∇ln(π(a|s)), 
            # we know that ∇ln(π(a|s)) = x(s,a) - Σ_b π(b|s)x(s,b)
            θ += α * G * (x - exp_x)
        end
        
        returns_per_episode[episode] = episode_return
    end
    
    return θ, returns_per_episode
end

# REINFORCE with Baseline (page 330 in Sutton RLI)
function reinforce_with_baseline(env::GridWorld, α_θ::Float64, α_w::Float64, γ::Float64, num_episodes::Int)
    # Initialize policy and value weights
    θ = 0.01 * randn(4)  # Policy weights (4 features)
    w = 0.01 * randn(4)  # Value function weights (4 features)
    
    # For learning curve
    returns_per_episode = zeros(num_episodes)
    
    for episode in 1:num_episodes
        # Generate an episode
        S = Vector{Vector{Int}}()
        A = Vector{Int}()
        R = Vector{Float64}()
        
        # Initialize state
        state = initialize!(env)
        push!(S, state)
        
        episode_return = 0.0
        terminal = false
        
        while !terminal
            # Select action from policy
            action = softmax_policy(state, θ)
            push!(A, action)
            
            # Take action, observe next state and reward
            next_state, reward, terminal = step!(env, action)
            push!(R, reward)
            episode_return += reward
            
            if !terminal
                push!(S, next_state)
                state = next_state
            end
        end
        
        # Policy gradient updates with baseline
        for t in 1:length(S)
            # Calculate discounted return from time t
            G = 0.0
            for k in t:length(R)
                G += γ^(k-t) * R[k]
            end
            
            # Update baseline (value function)
            state = S[t]
            x_v = state_features(state) # v bcs baseline must be independent of action
            δ = G - dot(w, x_v)  # TD error (w*x(s) used as baseline only and nowhere else)
            w += α_w * δ * x_v
            
            # Update policy weights
            action = A[t]
            
            # Calculate x(s,a) - Σ_b π(b|s)x(s,b)
            x = state_action_features(state, action)
            exp_x = expected_features(state, θ)
            
            # θ += α * δ * ∇ln(π(a|s)) where δ = G - v(s)
            θ += α_θ * δ * (x - exp_x)
        end
        
        returns_per_episode[episode] = episode_return
    end
    
    return θ, w, returns_per_episode
end

# One-step Actor-Critic (page 332 in Sutton RLI)
function one_step_actor_critic(env::GridWorld, α_θ::Float64, α_w::Float64, γ::Float64, num_episodes::Int)
    # Initialize policy and value weights
    θ = 0.01 * randn(4)  # Policy weights
    w = 0.01 * randn(4)  # Value function weights
    
    # For learning curve
    returns_per_episode = zeros(num_episodes)
    
    for episode in 1:num_episodes
        # Initialize state
        state = initialize!(env)
        I = 1.0  # Importance sampling ratio (always 1 in our case)
        
        episode_return = 0.0
        step_count = 0
        terminal = false
        
        while !terminal && step_count < env.T
            # Select action from policy
            action = softmax_policy(state, θ)
            
            # Take action, observe next state and reward
            next_state, reward, terminal = step!(env, action)
            episode_return += reward
            
            # TD error calculation
            x_v = state_features(state)
            
            if terminal
                # Terminal state
                target = reward  # v(terminal) = 0 or -1
            else
                # Non-terminal state
                x_v_next = state_features(next_state)
                target = reward + γ * dot(w, x_v_next) # w*x(s) used as critic
            end
            
            # TD error
            δ = target - dot(w, x_v) # w*x(s) used as baseline as well
            
            # Update value function weights
            w += α_w * δ * x_v
            
            # Update policy weights
            x = state_action_features(state, action)
            exp_x = expected_features(state, θ)
            
            # θ += α * I * δ * ∇ln(π(a|s))
            θ += α_θ * I * δ * (x - exp_x)
            
            # Update for next iteration
            state = next_state
            I *= γ
            step_count += 1
        end
        
        returns_per_episode[episode] = episode_return
    end
    
    return θ, w, returns_per_episode
end

###########################################
# Experiments
###########################################

# Run experiments with different learning rates
function experiment_learning_rates(num_episodes::Int, num_runs::Int=5)
    # Learning rates to test
    α_values = [0.01, 0.1, 0.2, 0.5, 0.9]
    γ = 0.95
    N = 10  # Grid size
    T = 200  # Maximum episode length
    
    # Results dictionary for each algorithm
    results = Dict()
    best_alphas = Dict()
    
    # 1. Semi-gradient 1-step SARSA
    println("== Testing learning rates for semi-gradient 1-step SARSA ==")
    sarsa_results = Dict()
    best_sarsa_perf = -Inf
    best_sarsa_α = 0.01
    
    for α in α_values
        println("Running with α = $α")
        all_returns = []
        for run in 1:num_runs
            Random.seed!(38 + run * 95)
            env = GridWorld(N, T)
            _, returns = semi_gradient_n_step_sarsa(env, 1, α, 0.1, γ, num_episodes)
            push!(all_returns, returns)
        end
        # Average the returns across runs
        avg_returns = mean(all_returns)
        sarsa_results[α] = avg_returns
        
        # Check if this is the best performance
        final_perf = mean(avg_returns[end-99:end])
        if final_perf > best_sarsa_perf
            best_sarsa_perf = final_perf
            best_sarsa_α = α
        end
    end
    
    results["1-step SARSA"] = sarsa_results
    best_alphas["SARSA"] = best_sarsa_α
    println("Best learning rate for SARSA: $(best_sarsa_α)")
    
    # 2. REINFORCE
    println("== Testing learning rates for REINFORCE ==")
    reinforce_results = Dict()
    best_reinforce_perf = -Inf
    best_reinforce_α = 0.01
    
    for α in α_values
        println("Running with α = $α")
        all_returns = []
        for run in 1:num_runs
            Random.seed!(38 + run * 95)
            env = GridWorld(N, T)
            _, returns = reinforce(env, α, γ, num_episodes)
            push!(all_returns, returns)
        end
        # Average the returns across runs
        avg_returns = mean(all_returns)
        reinforce_results[α] = avg_returns
        
        # Check if this is the best performance
        final_perf = mean(avg_returns[end-99:end])
        if final_perf > best_reinforce_perf
            best_reinforce_perf = final_perf
            best_reinforce_α = α
        end
    end
    
    results["REINFORCE"] = reinforce_results
    best_alphas["REINFORCE"] = best_reinforce_α
    println("Best learning rate for REINFORCE: $(best_reinforce_α)")
    
    # 3. REINFORCE with baseline
    println("== Testing learning rates for REINFORCE with baseline ==")
    reinforce_baseline_results = Dict()
    best_reinforce_baseline_perf = -Inf
    best_reinforce_baseline_α = 0.01
    
    for α in α_values
        println("Running with α = $α")
        all_returns = []
        for run in 1:num_runs
            Random.seed!(38 + run * 95)
            env = GridWorld(N, T)
            _, _, returns = reinforce_with_baseline(env, α, α, γ, num_episodes)
            push!(all_returns, returns)
        end
        # Average the returns across runs
        avg_returns = mean(all_returns)
        reinforce_baseline_results[α] = avg_returns
        
        # Check if this is the best performance
        final_perf = mean(avg_returns[end-99:end])
        if final_perf > best_reinforce_baseline_perf
            best_reinforce_baseline_perf = final_perf
            best_reinforce_baseline_α = α
        end
    end
    
    results["REINFORCE with baseline"] = reinforce_baseline_results
    best_alphas["REINFORCE with baseline"] = best_reinforce_baseline_α
    println("Best learning rate for REINFORCE with baseline: $(best_reinforce_baseline_α)")
    
    # 4. One-step Actor-Critic
    println("== Testing learning rates for One-step Actor-Critic ==")
    actor_critic_results = Dict()
    best_actor_critic_perf = -Inf
    best_actor_critic_α = 0.01
    
    for α in α_values
        println("Running with α = $α")
        all_returns = []
        for run in 1:num_runs
            Random.seed!(38 + run * 95)
            env = GridWorld(N, T)
            _, _, returns = one_step_actor_critic(env, α, α, γ, num_episodes)
            push!(all_returns, returns)
        end
        # Average the returns across runs
        avg_returns = mean(all_returns)
        actor_critic_results[α] = avg_returns
        
        # Check if this is the best performance
        final_perf = mean(avg_returns[end-99:end])
        if final_perf > best_actor_critic_perf
            best_actor_critic_perf = final_perf
            best_actor_critic_α = α
        end
    end
    
    results["Actor-Critic"] = actor_critic_results
    best_alphas["Actor-Critic"] = best_actor_critic_α
    println("Best learning rate for Actor-Critic: $(best_actor_critic_α)")
    
    return results, best_alphas
end

# Run experiments to compare value-based vs policy gradient methods
function experiment_value_vs_policy(best_alphas::Dict, num_episodes::Int, num_runs::Int=5)
    γ = 0.95
    ε = 0.1
    N = 10  # Grid size
    T = 200  # Maximum episode length
    
    # Get best learning rates
    sarsa_α = best_alphas["SARSA"]
    reinforce_α = best_alphas["REINFORCE"]
    reinforce_baseline_α = best_alphas["REINFORCE with baseline"]
    actor_critic_α = best_alphas["Actor-Critic"]
    
    # Results dictionaries
    value_based_results = Dict()
    policy_gradient_results = Dict()
    
    # Value-based methods (SARSA variants)
    println("== Running Value-based methods ==")
    for n in [1, 2, 3]
        println("Running semi-gradient $n-step SARSA")
        all_returns = []
        for run in 1:num_runs
            Random.seed!(38 + run * 95)
            env = GridWorld(N, T)
            _, returns = semi_gradient_n_step_sarsa(env, n, sarsa_α, ε, γ, num_episodes)
            push!(all_returns, returns)
        end
        value_based_results["$n-step SARSA"] = mean(all_returns)
    end
    
    # Policy gradient methods
    println("== Running Policy Gradient methods ==")
    println("Running REINFORCE")
    all_returns = []
    for run in 1:num_runs
        Random.seed!(38 + run * 95)
        env = GridWorld(N, T)
        _, returns = reinforce(env, reinforce_α, γ, num_episodes)
        push!(all_returns, returns)
    end
    policy_gradient_results["REINFORCE"] = mean(all_returns)
    
    println("Running REINFORCE with baseline")
    all_returns = []
    for run in 1:num_runs
        Random.seed!(38 + run * 95)
        env = GridWorld(N, T)
        _, _, returns = reinforce_with_baseline(env, reinforce_baseline_α, reinforce_baseline_α, γ, num_episodes)
        push!(all_returns, returns)
    end
    policy_gradient_results["REINFORCE with baseline"] = mean(all_returns)
    
    println("Running One-step Actor-Critic")
    all_returns = []
    for run in 1:num_runs
        Random.seed!(38 + run * 95)
        env = GridWorld(N, T)
        _, _, returns = one_step_actor_critic(env, actor_critic_α, actor_critic_α, γ, num_episodes)
        push!(all_returns, returns)
    end
    policy_gradient_results["Actor-Critic"] = mean(all_returns)
    
    return value_based_results, policy_gradient_results
end

###########################################
# Plotting Functions
###########################################

# Moving average
function moving_average(returns, window_size=10)
    smoothed = zeros(length(returns))
    for i in 1:length(returns)
        start_idx = max(1, i - window_size + 1)
        smoothed[i] = mean(returns[start_idx:i])
    end
    return smoothed
end

# Plot learning rate comparison
function plot_learning_rates(results, window_size=25)
    for (algo_name, algo_results) in results
        p = plot(title="Learning Rate Comparison - $algo_name", 
                 xlabel="Episode", ylabel="Average Return",
                 legend=:bottomright, size=(800, 500), dpi=600)
        
        # Sort alphas from smallest to largest
        sorted_alphas = sort(collect(keys(algo_results)))
        
        for α in sorted_alphas
            returns = algo_results[α]
            smoothed = moving_average(returns, window_size)
            plot!(p, smoothed, label="α = $α", linewidth=2)
        end
        
        display(p)
        savefig(p, "learning_rate_$(replace(algo_name, " " => "_")).png")
    end
end

# Plot value-based vs policy-gradient methods
function plot_value_vs_policy(value_results, policy_results, window_size=25)
    p = plot(title="Value-Based vs Policy Gradient Methods", 
             xlabel="Episode", ylabel="Average Return",
             legend=:bottomright, size=(800, 500), dpi=600)
    
    # Order of algorithms, apparently dicts are not ordered
    value_order = ["1-step SARSA", "2-step SARSA", "3-step SARSA"]
    policy_order = ["REINFORCE", "REINFORCE with baseline", "Actor-Critic"]
    
    # Value-based methods (solid lines)
    for algo_name in value_order
        if haskey(value_results, algo_name)
            returns = value_results[algo_name]
            smoothed = moving_average(returns, window_size)
            plot!(p, smoothed, label=algo_name, linewidth=2, linestyle=:solid)
        end
    end
    
    # Policy-gradient methods (dashed lines)
    for algo_name in policy_order
        if haskey(policy_results, algo_name)
            returns = policy_results[algo_name]
            smoothed = moving_average(returns, window_size)
            plot!(p, smoothed, label=algo_name, linewidth=2, linestyle=:dash)
        end
    end
    
    display(p)
    savefig(p, "value_vs_policy.png")
    
    # Calculate final performance
    println("== Final Performance (last 100 episodes) ==")
    println("Value-based methods:")
    for algo_name in value_order
        if haskey(value_results, algo_name)
            returns = value_results[algo_name]
            final_perf = mean(returns[end-99:end])
            println("$algo_name: $final_perf")
        end
    end
    
    println("Policy gradient methods:")
    for algo_name in policy_order
        if haskey(policy_results, algo_name)
            returns = policy_results[algo_name]
            final_perf = mean(returns[end-99:end])
            println("$algo_name: $final_perf")
        end
    end
end

###########################################
# Main Function
###########################################

function main()
    # Number of episodes for each experiment
    num_episodes = 150
    
    # Number of runs for averaging results (due to high variance)
    num_runs = 5

    # Moving average window
    window_size = 10
    
    # 1. Learning rate experiments
    println("=== Running learning rate experiments ===")
    lr_results, best_alphas = experiment_learning_rates(num_episodes, num_runs)
    plot_learning_rates(lr_results, window_size)
    
    # 2. Value-based vs Policy gradient experiments using best learning rates
    println("=== Running the main experiment ===")
    value_results, policy_results = experiment_value_vs_policy(best_alphas, num_episodes, num_runs)
    plot_value_vs_policy(value_results, policy_results, window_size)
    
    println("=== Experiments completed. Results saved as PNG files. ===")
end

main()
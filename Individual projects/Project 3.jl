using Random
using Statistics
using Plots
using LinearAlgebra

# ======== ENVIRONMENT ========
struct GridWorld
    N::Int  # Grid size
    T::Int  # Max episode length
    γ::Float64  # Discounting factor
end

# Default constructor
GridWorld() = GridWorld(10, 200, 0.95)

mutable struct State
    player_pos::Tuple{Int, Int}
    monster_pos::Tuple{Int, Int}
    apple_pos::Tuple{Int, Int}
end

# Actions
const ACTIONS = [:left, :up, :right, :down]
const ACTION_DELTAS = Dict(
    :left => (-1, 0),
    :up => (0, 1),
    :right => (1, 0),
    :down => (0, -1)
)

function initialize(env::GridWorld)
    positions = Set()
    
    # Generate random positions for player, monster, and apple
    player_pos = (rand(1:env.N), rand(1:env.N))
    push!(positions, player_pos)
    
    monster_pos = nothing
    while monster_pos === nothing || monster_pos ∈ positions
        monster_pos = (rand(1:env.N), rand(1:env.N))
    end
    push!(positions, monster_pos)
    
    apple_pos = nothing
    while apple_pos === nothing || apple_pos ∈ positions
        apple_pos = (rand(1:env.N), rand(1:env.N))
    end
    
    return State(player_pos, monster_pos, apple_pos)
end

function random_empty_position(env::GridWorld, state::State)
    positions = Set([state.player_pos, state.monster_pos])
    new_pos = nothing
    
    while new_pos === nothing || new_pos ∈ positions
        new_pos = (rand(1:env.N), rand(1:env.N))
    end
    
    return new_pos
end

function manhattan_distance(pos1, pos2)
    return abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
end

function step(env::GridWorld, state::State, action::Symbol)
    # Player movement
    dx, dy = ACTION_DELTAS[action]
    new_x = clamp(state.player_pos[1] + dx, 1, env.N)
    new_y = clamp(state.player_pos[2] + dy, 1, env.N)
    new_player_pos = (new_x, new_y)
    
    # Monster movement
    monster_action = rand(ACTIONS)
    mdx, mdy = ACTION_DELTAS[monster_action]
    new_mx = clamp(state.monster_pos[1] + mdx, 1, env.N)
    new_my = clamp(state.monster_pos[2] + mdy, 1, env.N)
    new_monster_pos = (new_mx, new_my)
    
    # Check if player is caught by monster
    if new_player_pos == new_monster_pos
        return State(new_player_pos, new_monster_pos, state.apple_pos), -1, true
    end
    
    # Check if player collects apple
    if new_player_pos == state.apple_pos
        new_apple_pos = random_empty_position(env, State(new_player_pos, new_monster_pos, state.apple_pos))
        return State(new_player_pos, new_monster_pos, new_apple_pos), 1, false
    end
    
    # Default case
    return State(new_player_pos, new_monster_pos, state.apple_pos), 0, false
end

# ======== FEATURE EXTRACTION ========
function will_get_closer(pos, target, action)
    dx, dy = ACTION_DELTAS[action]
    new_pos = (clamp(pos[1] + dx, 1, 10), clamp(pos[2] + dy, 1, 10))
    
    current_dist = manhattan_distance(pos, target)
    new_dist = manhattan_distance(new_pos, target)
    
    return new_dist < current_dist
end

function extract_features(state::State, action::Symbol)
    # New feature representation:
    # f1 = 1/(dist to apple + 1)
    # f2 = 1/(dist to monster + 1)
    # f3 = 1 if action takes you closer to apple else 0
    # f4 = 1 if action takes you closer to monster else 0
    
    apple_dist = manhattan_distance(state.player_pos, state.apple_pos)
    monster_dist = manhattan_distance(state.player_pos, state.monster_pos)
    
    f1 = 1.0 / (apple_dist + 1)
    f2 = 1.0 / (monster_dist + 1)
    f3 = will_get_closer(state.player_pos, state.apple_pos, action) ? 1.0 : 0.0
    f4 = will_get_closer(state.player_pos, state.monster_pos, action) ? 1.0 : 0.0
    
    return [f1, f2, f3, f4]
end

function extract_state_features(state::State)
    # Features that depend only on state, not action
    apple_dist = manhattan_distance(state.player_pos, state.apple_pos)
    monster_dist = manhattan_distance(state.player_pos, state.monster_pos)
    
    f1 = 1.0 / (apple_dist + 1)
    f2 = 1.0 / (monster_dist + 1)
    
    # Position features (normalized)
    f3 = state.player_pos[1] / 10
    f4 = state.player_pos[2] / 10
    
    return [f1, f2, f3, f4]
end

# ======== APPROXIMATORS ========
function q_value_function(state::State, action::Symbol, w::Vector{Float64})
    # q̂(s, a; w) = wᵀx(s, a)
    return dot(w, extract_features(state, action))
end

function v_value_function(state::State, w::Vector{Float64})
    # v̂(s; w) = wᵀx(s)
    return dot(w, extract_state_features(state))
end

function softmax_policy(state::State, θ::Vector{Float64})
    # π(a|s; θ) = exp(θᵀx(s, a)) / Σ_b exp(θᵀx(s, b))
    logits = [dot(θ, extract_features(state, a)) for a in ACTIONS]
    probs = softmax(logits)
    return Dict(ACTIONS[i] => probs[i] for i in 1:length(ACTIONS))
end

function softmax(x)
    ex = exp.(x .- maximum(x))
    return ex ./ sum(ex)
end

function epsilon_greedy_policy(state::State, w::Vector{Float64}, ε::Float64)
    if rand() < ε
        return rand(ACTIONS)
    else
        q_values = [q_value_function(state, a, w) for a in ACTIONS]
        best_action_idx = argmax(q_values)
        return ACTIONS[best_action_idx]
    end
end

function get_decaying_ε(state, w) # TODO: make sure this adapts to softmax exploration level correctly
    probs = softmax_policy(state, w)
    max_prob = maximum(values(probs))

    ε = 1 - max_prob
    return ε
end

function sample_action(probs_dict::Dict{Symbol, Float64})
    r = rand()
    cumsum = 0.0
    for (action, prob) in probs_dict
        cumsum += prob
        if r <= cumsum
            return action
        end
    end
    return first(keys(probs_dict))  # Fallback
end

function policy_gradient(state::State, action::Symbol, θ::Vector{Float64})
    # ∇_θ log π(a|s; θ) = x(s, a) - Σ_b π(b|s; θ)x(s, b)
    action_features = extract_features(state, action)
    
    probs = softmax_policy(state, θ)
    weighted_sum = zeros(length(action_features))
    
    for a in ACTIONS
        features = extract_features(state, a)
        weighted_sum .+= probs[a] .* features
    end
    
    return action_features .- weighted_sum
end

# ======== ALGORITHMS ========

# Semi-gradient n-step SARSA
function semi_gradient_nstep_sarsa(env::GridWorld, n::Int, num_episodes::Int, α)
    # Initialize weights
    d = length(extract_features(initialize(env), :left))
    w = zeros(d)
    rewards_per_episode = zeros(num_episodes)
    
    for episode in 1:num_episodes
        # Initialize
        S = initialize(env)
        A = epsilon_greedy_policy(S, w, get_decaying_ε(S, w))
        
        states = Vector{State}()
        actions = Vector{Symbol}()
        rewards = Vector{Float64}()
        
        push!(states, S)
        push!(actions, A)
        
        t = 0
        T = Inf
        total_reward = 0.0
        
        while t < T - 1
            if t < T
                # Take action
                S_next, R, terminal = step(env, states[end], actions[end])
                total_reward += R
                push!(states, S_next)
                push!(rewards, R)
                
                if terminal
                    T = t + 1
                else
                    A_next = epsilon_greedy_policy(S_next, w, get_decaying_ε(S_next, w))
                    push!(actions, A_next)
                end
            end
            
            # τ is the time whose estimate is being updated
            τ = t - n + 1
            if τ >= 0
                # Calculate return
                G = 0.0
                for i::Int in τ+1:min(τ+n, T)
                    G += env.γ^(i-τ-1) * rewards[i]
                end
                
                if τ + n < T
                    G += env.γ^n * q_value_function(states[τ+n+1], actions[τ+n+1], w)
                end
                
                # Update weights
                δ = G - q_value_function(states[τ+1], actions[τ+1], w)
                w .+= α * δ * extract_features(states[τ+1], actions[τ+1])
            end
            
            t += 1
        end
        
        rewards_per_episode[episode] = total_reward
    end
    
    return rewards_per_episode, w
end

# REINFORCE
function reinforce(env::GridWorld, num_episodes::Int, α)
    # Initialize policy parameters
    d = length(extract_features(initialize(env), :left))
    θ = zeros(d)
    rewards_per_episode = zeros(num_episodes)
    
    for episode in 1:num_episodes
        # Generate an episode
        S = initialize(env)
        states = [S]
        actions = Symbol[]
        rewards = Float64[]
        
        terminal = false
        step_count = 0
        total_reward = 0.0
        
        while !terminal && step_count < env.T
            probs = softmax_policy(S, θ)
            A = sample_action(probs)
            push!(actions, A)
            
            S_next, R, term = step(env, S, A)
            push!(states, S_next)
            push!(rewards, R)
            
            total_reward += R
            terminal = term
            S = S_next
            step_count += 1
        end
        
        # Calculate returns for each step
        G = zeros(length(rewards))
        G[end] = rewards[end]
        
        for t in (length(rewards)-1):-1:1
            G[t] = rewards[t] + env.γ * G[t+1]
        end
        
        # Update policy parameters
        for t in 1:length(actions)
            grad = policy_gradient(states[t], actions[t], θ)
            θ .+= α * G[t] * grad
        end
        
        rewards_per_episode[episode] = total_reward
    end
    
    return rewards_per_episode, θ
end

# REINFORCE with Baseline
function reinforce_with_baseline(env::GridWorld, num_episodes::Int, α_θ, α_w)
    # Initialize parameters
    action_feature_dim = length(extract_features(initialize(env), :left))
    state_feature_dim = length(extract_state_features(initialize(env)))
    
    θ = zeros(action_feature_dim)  # Policy parameters
    w = zeros(state_feature_dim)   # Value function parameters (state features only)
    rewards_per_episode = zeros(num_episodes)
    
    for episode in 1:num_episodes
        # Generate an episode
        S = initialize(env)
        states = [S]
        actions = Symbol[]
        rewards = Float64[]
        
        terminal = false
        step_count = 0
        total_reward = 0.0
        
        while !terminal && step_count < env.T
            probs = softmax_policy(S, θ)
            A = sample_action(probs)
            push!(actions, A)
            
            S_next, R, term = step(env, S, A)
            push!(states, S_next)
            push!(rewards, R)
            
            total_reward += R
            terminal = term
            S = S_next
            step_count += 1
        end
        
        # Calculate returns for each step
        G = zeros(length(rewards))
        G[end] = rewards[end]
        
        for t in (length(rewards)-1):-1:1
            G[t] = rewards[t] + env.γ * G[t+1]
        end
        
        # Update parameters
        for t in 1:length(actions)
            # State value function baseline
            state_features = extract_state_features(states[t])
            baseline = v_value_function(states[t], w)
            δ = G[t] - baseline
            
            # Update value function weights for the baseline
            w .+= α_w * δ * state_features
            
            # Update policy parameters
            grad = policy_gradient(states[t], actions[t], θ)
            θ .+= α_θ * δ * grad
        end
        
        rewards_per_episode[episode] = total_reward
    end
    
    return rewards_per_episode, θ, w
end

# One-step Actor-Critic
function one_step_actor_critic(env::GridWorld, num_episodes::Int, α_θ, α_w)
    # Initialize parameters
    action_feature_dim = length(extract_features(initialize(env), :left))
    state_feature_dim = length(extract_state_features(initialize(env)))
    
    θ = zeros(action_feature_dim)  # Policy parameters
    w = zeros(state_feature_dim)   # Value function parameters
    rewards_per_episode = zeros(num_episodes)
    
    for episode in 1:num_episodes
        S = initialize(env)
        terminal = false
        step_count = 0
        total_reward = 0.0
        I = 1.0  # Importance sampling ratio
        
        while !terminal && step_count < env.T
            # Select action
            probs = softmax_policy(S, θ)
            A = sample_action(probs)
            
            # Take action
            S_next, R, terminal = step(env, S, A)
            total_reward += R
            
            # Get current state value
            current_value = v_value_function(S, w)
            
            # Calculate TD error
            δ = R
            if !terminal
                next_value = v_value_function(S_next, w)
                δ += env.γ * next_value
            end
            δ -= current_value
            
            # Update value function weights
            state_features = extract_state_features(S)
            w .+= α_w * δ * state_features
            
            # Update policy parameters
            grad = policy_gradient(S, A, θ)
            θ .+= α_θ * I * δ * grad
            
            # Update for next iteration
            I *= env.γ
            S = S_next
            step_count += 1
        end
        
        rewards_per_episode[episode] = total_reward
    end
    
    return rewards_per_episode, θ, w
end

# ======== EVALUATION AND PLOTTING ========
function moving_average(x, window_size)
    result = zeros(length(x))
    for i in 1:length(x)
        window_start = max(1, i - window_size + 1)
        result[i] = mean(x[window_start:i])
    end
    return result
end

# Centralized plotting function
function plot_learning_curves(data_dict, title, filename, num_episodes; 
                              xlabel="Episode", 
                              ylabel="Average Reward",
                              shitf_x_pos_to_left_by=0, 
                              y_offsets=Dict(
                                "1-step SARSA" => 0.5,
                                "2-step SARSA" => 1.25,
                                "3-step SARSA" => 2,
                                "REINFORCE" => 2.75,
                                "REINFORCE with baseline" => 3.5,
                                "Actor-Critic" => 4.25
                            ),
                              methods_order=nothing)
    
    # Define color scheme
    colors = [:chocolate, :red, :green, :blueviolet, :orange, :pink1, :blue]
    
    # Create plot
    p = plot(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        titlefontsize=10,
        labelfontsize=8,
        linewidth=1.1,
        legend=false,
        dpi=600
    )
    
    # If methods_order is not provided, use the keys in data_dict
    if methods_order === nothing
        methods = collect(keys(data_dict))
    else
        methods = methods_order
    end
    
    # Plot each curve and add annotation
    for (i, method) in enumerate(methods)
        color_idx = mod(i - 1, length(colors)) + 1  # Cycle through colors if more methods than colors
        
        # Plot the curve
        plot!(p, 1:num_episodes, data_dict[method], 
              color=colors[color_idx], 
              linewidth=1.1)
        
        # Add annotation at the end of the line
        annotate!(p, 
                  num_episodes-shitf_x_pos_to_left_by, 
                  0 + get(y_offsets, method, 0), 
                  text(method, colors[color_idx], :right, 7))
    end
    
    savefig(p, filename)
    return p
end

# Test with different learning rates
function test_learning_rates(method_name::String, num_episodes=1000, window_size=50, num_runs=5, rates=[0.01, 0.1, 0.2, 0.3, 0.5])
    env = GridWorld(10, 200, 0.95)
    all_results = Dict()
    
    # Define method display names
    method_display_names = Dict(
        "sarsa_1" => "1-step SARSA",
        "sarsa_3" => "3-step SARSA",
        "reinforce" => "REINFORCE",
        "reinforce_with_baseline" => "REINFORCE with baseline",
        "actor_critic" => "Actor-Critic"
    )
    
    # Validate method name
    if !(method_name in keys(method_display_names))
        error("Invalid method name. Choose from: $(join(keys(method_display_names), ", "))")
    end
    
    println("Testing $method_name for $num_runs runs")
    
    for rate in rates
        all_results[rate] = []
        
        for run in 1:num_runs
            println("Testing $method_name with learning rate α = $rate, run $run/$num_runs")
            
            # Run the specified method
            if method_name == "sarsa_1"
                rewards, _ = semi_gradient_nstep_sarsa(env, 1, num_episodes, rate)
            elseif method_name == "sarsa_3"
                rewards, _ = semi_gradient_nstep_sarsa(env, 3, num_episodes, rate)
            elseif method_name == "reinforce"
                rewards, _ = reinforce(env, num_episodes, rate)
            elseif method_name == "reinforce_with_baseline"
                rewards, _ = reinforce_with_baseline(env, num_episodes, rate, rate)
            elseif method_name == "actor_critic"
                rewards, _, _ = one_step_actor_critic(env, num_episodes, rate, rate)
            end
            
            # Store the results for this run
            push!(all_results[rate], moving_average(rewards, window_size))
        end
    end
    
    # Create plots for each learning rate
    for rate in rates
        # Average results across runs
        avg_results = mean(all_results[rate])
        
        # Create data dictionary for this rate
        data_dict = Dict("Run $i" => all_results[rate][i] for i in 1:num_runs)
        data_dict["Average"] = avg_results
    end
    
    avg_data_dict = Dict("α = $rate" => mean(all_results[rate]) for rate in rates)

    offsets = [0.5, 1.25, 2, 2.75, 3.5]
    y_offsets_dict = Dict("α = $rate" => offsets[i] for (i, rate) in enumerate(rates))

    plot_learning_curves(
        avg_data_dict,
        "Average Learning Curves for $(method_display_names[method_name])",
        "$(method_name)_learning_rates_comparison.png",
        num_episodes=num_episodes,
        shitf_x_pos_to_left_by=5,
        y_offsets=y_offsets_dict
    )    
end

# Evaluate a trained policy
function evaluate_policy(env::GridWorld, policy_fn, num_episodes=100)
    total_rewards = zeros(num_episodes)
    
    for episode in 1:num_episodes
        S = initialize(env)
        terminal = false
        step_count = 0
        episode_reward = 0.0
        
        while !terminal && step_count < env.T
            A = policy_fn(S)
            S_next, R, terminal = step(env, S, A)
            
            episode_reward += R
            S = S_next
            step_count += 1
        end
        
        total_rewards[episode] = episode_reward
    end
    
    return mean(total_rewards), std(total_rewards)
end

function visualise_training(env, methods, num_episodes, num_runs, window_size=nothing)
    rewards_per_episode_dict = Dict{String, Vector{Float64}}()
    moving_averages_dict = Dict{String, Vector{Float64}}()

    if window_size === nothing
        window_size = Int(num_episodes/10)
    end
    
    for method in methods
        rewards_per_episode_dict[method] = zeros(num_episodes)
    end

    for i in 1:num_runs
        println("Training run $i/$num_runs")
        
        # Accumulate rewards
        for method in methods
            rewards = Vector{Float64}()
            
            if method == "1-step SARSA"
                rewards, _ = semi_gradient_nstep_sarsa(env, 1, num_episodes, 0.1)
            elseif method == "2-step SARSA"
                rewards, _ = semi_gradient_nstep_sarsa(env, 2, num_episodes, 0.1)
            elseif method == "3-step SARSA"
                rewards, _ = semi_gradient_nstep_sarsa(env, 3, num_episodes, 0.1)
            elseif method == "REINFORCE"
                rewards, _ = reinforce(env, num_episodes, 0.1)
            elseif method == "REINFORCE with baseline"
                rewards, _, _ = reinforce_with_baseline(env, num_episodes, 0.1, 0.1)
            elseif method == "Actor-Critic"
                rewards, _, _ = one_step_actor_critic(env, num_episodes, 1.0, 1.0)
            else
                error("Unsupported method: $method. Not in $methods.")
            end
            
            rewards_per_episode_dict[method] .+= rewards
        end
    end
    
    # Calculate averages
    for method in methods
        rewards_per_episode_dict[method] ./= num_runs
    end
    
    # Calculate moving averages
    for method in methods
        moving_averages_dict[method] = moving_average(rewards_per_episode_dict[method], window_size)
    end
    
    # Plot learning curves
    plot_learning_curves(
        moving_averages_dict,
        "Learning curves for different methods",
        "learning_curves.png",
        num_episodes,
        shitf_x_pos_to_left_by=5,
        methods_order=methods
    )
end

function train(env, methods, num_episodes)
    method_params = Dict{String, Vector{Float64}}()

    for method in methods        
        if method == "1-step SARSA"
            _, params = semi_gradient_nstep_sarsa(env, 1, num_episodes, 0.1)
        elseif method == "2-step SARSA"
            _, params = semi_gradient_nstep_sarsa(env, 2, num_episodes, 0.1)
        elseif method == "3-step SARSA"
            _, params = semi_gradient_nstep_sarsa(env, 3, num_episodes, 0.1)
        elseif method == "REINFORCE"
            _, params = reinforce(env, num_episodes, 0.1)
        elseif method == "REINFORCE with baseline"
            _, params, _ = reinforce_with_baseline(env, num_episodes, 0.1, 0.1)
        elseif method == "Actor-Critic"
            _, params, _ = one_step_actor_critic(env, num_episodes, 1.0, 1.0)
        else
            error("Unsupported method: $method. Not in $methods.")
        end

        method_params[method] = params
    end
    return method_params
end

function evaluate(env, methods, trained_params, num_eval_episodes, num_eval_runs)
    evaluation_results = Dict()
    
    for method in methods
        println("Evaluating $method...")
        mean_reward = 0.0
        std_reward = 0.0
        
        if method in ["1-step SARSA", "2-step SARSA", "3-step SARSA"]
            w = trained_params[method]
            policy = s -> epsilon_greedy_policy(s, w, 0.0)  # No exploration during evaluation
        elseif method in  ["REINFORCE", "REINFORCE with baseline", "Actor-Critic"]
            θ = trained_params[method]
            policy = s -> sample_action(softmax_policy(s, θ))
        else
            error("Unsupported method: $method. Not in $methods.")
        end
        
        for i in 1:num_eval_runs
            mean_temp, std_temp = evaluate_policy(env, policy, num_eval_episodes)
            mean_reward += mean_temp
            std_reward += std_temp
        end
        
        mean_reward /= num_eval_runs
        std_reward /= num_eval_runs
        evaluation_results[method] = (mean_reward, std_reward)
    end
    
    return evaluation_results
end

function print_evaluation(evaluation_results, methods)
    println("Evaluation Results:")
    println("==================")
    for method in methods
        mean_reward, std_reward = evaluation_results[method]
        println("$method: Mean reward = $mean_reward ± $std_reward")
    end
end

function run_experiment(num_train_episodes, num_train_runs, window_size, 
                       num_eval_episodes, num_eval_runs, obtain_learning_curves=true, perform_evaluation=false)
    env = GridWorld(10, 200, 0.95)
    methods = ["1-step SARSA", "2-step SARSA", "3-step SARSA", "REINFORCE",
               "REINFORCE with baseline", "Actor-Critic"]
    
    # Plot learning curves if obtain_learning_curves true
    if obtain_learning_curves
        println("Colleting plotting data from multiple trainings...")
        visualise_training(env, methods, num_train_episodes, num_train_runs, window_size)
    end

    # Evaluate if perform_evaluation true
    if perform_evaluation
        # Train algorithms
        println("Training...")
        trained_params = train(env, methods, num_train_episodes)
        # Evaluate algorithms
        println("Evaluating learned policies...")
        evaluation_results = evaluate(env, methods, trained_params, num_eval_episodes, num_eval_runs)
        # Display evaluation results
        print_evaluation(evaluation_results, methods)
    end
end

# Main function to run the experiment
function main()
    # # Test different learning rates
    # println("Testing different learning rates...")
    # test_learning_rates("sarsa_1", num_episodes, window_size, num_runs)

    # Run the main experiment
    println("Running main experiment...")
    run_experiment(150, 100, 10, 150, 100, true, true)
    
    println("Experiments completed. Results saved as PNG files.")
end

main()
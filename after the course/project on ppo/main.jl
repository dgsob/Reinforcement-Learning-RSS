using HTTP
using Gumbo
using Cascadia
using Logging

global_logger(ConsoleLogger(stderr, Logging.Info))

include("environment/bp_score.jl")
include("./environment/peptide_sequence_env.jl")
include("./ppo.jl")

# Test function to train PPO and evaluate the trained policy
function test_ppo_training_and_evaluation()
    println("Starting PPO training and evaluation...")

    # Initialize environment
    env = PeptideSequenceEnv(n=6, max_steps=10, target_reward=1.0f0)
    println("Environment initialized with sequence length: $(env.n)")

    # Run PPO training
    println("Training PPO...")
    actor, critic, θ, ξ, ϕ, ζ = train_ppo()
    println("PPO training completed.")

    # Evaluate the trained policy
    println("Evaluating the trained policy...")
    state = reset!(env)
    println("Initial sequence: ", sequence_to_string(state))

    # Run a few steps using the trained actor
    for step in 1:5
        # Sample an action using the trained actor
        action = sample_action(actor, state, θ, ξ, Random.default_rng())

        # Take a step in the environment
        new_state, reward, done = step!(env, action)
        println("Step $step: Sequence = $(sequence_to_string(new_state)), Reward = $reward, Done = $done")

        state = new_state
        if done
            println("Episode ended early.")
            break
        end
    end
end

test_ppo_training_and_evaluation()
using HTTP
using Gumbo
using Cascadia

include("./environment/peptide_sequence_env.jl")
include("./environment/solubility_api.jl")
include("./environment/secondary_structure_api.jl")

# Test function to evaluate solubility API
function run_test()
    # Initialize environment with longer sequence
    env = PeptideSequenceEnv(n=21, max_steps=10, target_reward=0.9f0)
    println("Testing PeptideSequenceEnv with solubility API...")

    # Reset environment to get initial sequence
    state = reset!(env)
    println("Initial sequence: ", sequence_to_string(state))

    # Run a few steps with simple actions
    for step in 1:5
        # Action: Propose a new sequence (e.g., random, then modify positions)
        action = rand(1:length(AMINO_ACIDS), env.n)  # Uniformly random sequence

        # Take a step in the environment
        new_state, reward, done = step!(env, action)
        println("Step $step: Sequence = $(sequence_to_string(new_state)), Reward = $reward, Done = $done")
        
        if done
            println("Episode ended early.")
            break
        end
    end
end

# Run the test
run_test()
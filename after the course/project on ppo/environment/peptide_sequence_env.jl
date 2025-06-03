using Random

const AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

mutable struct PeptideSequenceEnv
    n::Int                  # Length of peptide sequence
    sequence::Vector{Int}   # Current sequence (indices 1:20)
    max_steps::Int          # Maximum steps per episode
    step_count::Int         # Current step
    done::Bool              # Episode done flag
    target_reward::Float32  # Threshold for "good" peptide

    # Constructor
    function PeptideSequenceEnv(; n::Int=21, max_steps::Int=50, target_reward::Float32=0.9f0)
        if n < 21
            error("Sequence length must be at least 21 amino acids for Protein-Sol compatibility")
        end
        sequence = rand(1:length(AMINO_ACIDS), n)  # Uniformly random initial sequence
        new(n, sequence, max_steps, 0, false, target_reward)
    end
end

function reset!(env::PeptideSequenceEnv)
    env.sequence = rand(1:length(AMINO_ACIDS), env.n)
    env.step_count = 0
    env.done = false
    return copy(env.sequence)  # return current state
end

function sequence_to_string(seq::Vector{Int})
    return join(AMINO_ACIDS[seq], "")
end

# function calculate_reward(sol_score, stru_score)

# end

# Non-linear transformation using a sigmoid-like function
function calculate_reward(sol_score::Float64)
    sol_score = clamp(sol_score, 0.0, 1.0)  # ensure the solubility score is within [0, 1]
    steepness = 10.0  # controls how steep the sigmoid is
    threshold = 0.5   # solubility threshold
    scaled_reward = 1.0 / (1.0 + exp(-steepness * (sol_score - threshold)))
    
    return 2.0 * scaled_reward - 1.0  # shift and scale the reward to range from -1 to 1
end

function step!(env::PeptideSequenceEnv, action::Vector{Int})  # TODO: Rewrite to accept GA-like actions later if necessary
    env.step_count += 1
    if length(action) == env.n && all(1 .<= action .<= length(AMINO_ACIDS))
        env.sequence = action
    else
        error("Invalid action: sequence must be length $(env.n) with indices 1-20")
    end

    # Evaluate with real tools
    sequence_str = sequence_to_string(env.sequence)
    solubility_score = get_solubility(sequence_str)
    # structure_score = get_secondary_structure(sequence_str)
    reward = calculate_reward(solubility_score)
    if reward >= env.target_reward
        env.done = true
    elseif env.step_count >= env.max_steps
        env.done = true
    end

    return copy(env.sequence), Float32(reward), env.done
end
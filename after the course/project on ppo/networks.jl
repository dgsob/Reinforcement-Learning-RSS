using Lux

# Policy network: outputs logits for each position and amino acid
function create_policy_network(n::Int, num_amino_acids::Int)
    return Chain(
        x -> reshape(x, :,),  # Flatten the input (num_amino_acids * n) into a vector
        Dense(n * num_amino_acids => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => n * num_amino_acids),
        x -> reshape(x, num_amino_acids, n)  # Reshape output to (num_amino_acids, n) for logits
    )
end

# Value network: predicts the value of a state
function create_value_network(n::Int, num_amino_acids::Int)
    return Chain(
        x -> reshape(x, :,),  # Flatten the input (num_amino_acids * n) into a vector
        Dense(n * num_amino_acids => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => 1),
        x -> x[1]  # Extract the scalar from the (1,) vector
    )
end
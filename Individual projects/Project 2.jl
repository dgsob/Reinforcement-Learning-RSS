### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 237f2b15-e7b1-4022-9f88-8b2b3ce25743
# This is a different code style from Project 1. It used to be lame, now it's functional as it should be in Julia. Hopefully it won't require debugging to such a huge extend. 

# ╔═╡ 2ac49d21-013a-4190-842e-ac2312df66eb
# ------------- Environment Set Up ------------------

# ╔═╡ 45369aa5-2f0c-4787-9492-c9f1e19ace7d
# Define the gridworld environment
struct GridWorld
    N::Int  # Grid size (N x N)
    T::Int  # Episode length
    γ::Float64  # Discount factor
    α::Float64  # Learning rate
end

# ╔═╡ 8fd48449-cf2f-45aa-b4a4-23f121ce16e3
# Define the state space
struct State
    player_pos::Tuple{Int, Int}
    monster_pos::Tuple{Int, Int}
    apple_pos::Tuple{Int, Int}
end

# ╔═╡ 93380906-f805-4c38-a84e-88ce3e0ee556
# Define action space
const ACTIONS = [:left, :up, :right, :down]

# ╔═╡ 267184ab-f306-4ded-b1bb-72a25f82a527
# Initialize the environment
function initialize_gridworld(N, T, γ, α)
    return GridWorld(N, T, γ, α)
end

# ╔═╡ b59a2e38-c98f-48ff-9319-41e7cfecfa0b
# Initialize the state with random positions for player, monster, and apple
function initialize_state(N)
    player_pos = (rand(1:N), rand(1:N)) # we will number from 1, I'm sorry
    monster_pos = (rand(1:N), rand(1:N))
    apple_pos = (rand(1:N), rand(1:N))
    
    # Ensure that the player, monster, and apple are not in the same position
    while player_pos == monster_pos || player_pos == apple_pos || monster_pos == apple_pos
        monster_pos = (rand(1:N), rand(1:N))
        apple_pos = (rand(1:N), rand(1:N))
    end
    
    return State(player_pos, monster_pos, apple_pos)
end

# ╔═╡ de6869c5-afd7-4c16-9707-504c9abc6853
# Function to check if a position is within the grid
function is_within_grid(pos, N)
    x, y = pos
    return x ≥ 1 && x ≤ N && y ≥ 1 && y ≤ N
end

# ╔═╡ e865b457-2a89-4c1d-bc51-59dcd8b31abd
# Function to move the player based on the action
function move_player(player_pos, action, N)
    x, y = player_pos
    if action == :left
        new_pos = (x - 1, y)
    elseif action == :up
        new_pos = (x, y - 1)
    elseif action == :right
        new_pos = (x + 1, y)
    elseif action == :down
        new_pos = (x, y + 1)
    else
        error("Invalid action")
    end
    
    # If the new position is outside the grid, stay in place
    return is_within_grid(new_pos, N) ? new_pos : player_pos
end

# ╔═╡ c362401b-cb06-4119-8094-af7a3cf30628
# Function to move the monster randomly
function move_monster(monster_pos, N)
    possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    move = rand(possible_moves)
    new_pos = (monster_pos[1] + move[1], monster_pos[2] + move[2])
    
    # If the new position is outside the grid, stay in place
    return is_within_grid(new_pos, N) ? new_pos : monster_pos
end

# ╔═╡ 6a20ba58-fac4-406a-b68a-ba658d2dc60e
# Function to spawn a new apple in a random empty position
function spawn_apple(player_pos, monster_pos, N)
    empty_positions = [(x, y) for x in 1:N for y in 1:N if (x, y) != player_pos && (x, y) != monster_pos]
    return rand(empty_positions)
end

# ╔═╡ 88685918-bc20-49c9-bd4d-4fe5e7f5f0c3
# Function to simulate a step in the environment
function step(env::GridWorld, state::State, action::Symbol)
    player_pos = move_player(state.player_pos, action, env.N)
    monster_pos = move_monster(state.monster_pos, env.N)
    
    # Check if the player is caught by the monster
    if player_pos == monster_pos
        reward = -1
        terminal = true
        new_apple_pos = state.apple_pos  # player caught, apple pos doesn't matter
    else
        # Check if the player collects the apple
        if player_pos == state.apple_pos
            reward = 1
            new_apple_pos = spawn_apple(player_pos, monster_pos, env.N)
        else
            reward = 0
            new_apple_pos = state.apple_pos # apple not reached, stays the same
        end
        terminal = false
    end
    
    new_state = State(player_pos, monster_pos, new_apple_pos)
    return new_state, reward, terminal
end

# ╔═╡ cad3c032-7623-46ac-ad1c-0742d8dc914e
# Initialize state variables
begin
	N = 5 # small remark: it's {1, 2, 3, 4, 5} in our case, not {0, 1, 2, 3, 4} 
	T = 30
	γ = 0.9
	α = 0.1
end

# ╔═╡ c2a07cee-e83e-44f1-b6d9-1865a11de913
env = initialize_gridworld(N, T, γ, α)

# ╔═╡ 6b78b6d4-8e33-41ac-aa01-a2f30d946c2b
state = initialize_state(N)

# ╔═╡ e5668bc3-eeca-4354-88aa-3c39b66ec2d4
# Testing the env with a random policy
for t in 1:T
    action = rand(ACTIONS)  # Random action for demonstration
    new_state, reward, terminal = step(env, state, action)
    println("Step $t: Action = $action, Reward = $reward, Terminal = $terminal")
    state = new_state
    if terminal
        println("Episode ended at step $t")
        break
    end
end

# ╔═╡ ab1c9b18-b108-4844-9cdb-519f186d7a35
# --------------- Training phases -------------------

# ╔═╡ b9ec071d-b40a-4fad-8c34-72fa3877bca2


# ╔═╡ Cell order:
# ╠═237f2b15-e7b1-4022-9f88-8b2b3ce25743
# ╠═2ac49d21-013a-4190-842e-ac2312df66eb
# ╠═45369aa5-2f0c-4787-9492-c9f1e19ace7d
# ╠═8fd48449-cf2f-45aa-b4a4-23f121ce16e3
# ╠═93380906-f805-4c38-a84e-88ce3e0ee556
# ╠═267184ab-f306-4ded-b1bb-72a25f82a527
# ╠═b59a2e38-c98f-48ff-9319-41e7cfecfa0b
# ╠═de6869c5-afd7-4c16-9707-504c9abc6853
# ╠═e865b457-2a89-4c1d-bc51-59dcd8b31abd
# ╠═c362401b-cb06-4119-8094-af7a3cf30628
# ╠═6a20ba58-fac4-406a-b68a-ba658d2dc60e
# ╠═88685918-bc20-49c9-bd4d-4fe5e7f5f0c3
# ╠═cad3c032-7623-46ac-ad1c-0742d8dc914e
# ╠═c2a07cee-e83e-44f1-b6d9-1865a11de913
# ╠═6b78b6d4-8e33-41ac-aa01-a2f30d946c2b
# ╠═e5668bc3-eeca-4354-88aa-3c39b66ec2d4
# ╠═ab1c9b18-b108-4844-9cdb-519f186d7a35
# ╠═b9ec071d-b40a-4fad-8c34-72fa3877bca2

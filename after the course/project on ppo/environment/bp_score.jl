# A heuristic to filter out initial sequences that are unlikely to bind to anything
function get_binding_propensity_score(
    peptide_sequence::String;
    pH::Float64 = 7.4,
    ideal_gravy_range::Tuple{Float64, Float64} = (-0.5, 1.5), # Default: Moderately hydrophilic to mildly hydrophobic
    weight_gravy::Float64 = 0.7,
    weight_charge::Float64 = 0.3,
    gravy_cap_low::Float64 = -5.0,
    gravy_cap_high::Float64 = 5.0,
    # NEW PARAMETERS for charge preference:
    min_abs_charge_threshold::Float64 = 2.0, # Minimum absolute charge to get a high score
    max_abs_charge_threshold::Float64 = 8.0, # Maximum absolute charge to get a high score (beyond this, score drops)
    abs_charge_cap_high::Float64 = 12.0 # Absolute charge at which score is 0
)
    # Kyte-Doolittle Hydropathicity values
    gravy_scales = Dict{Char, Float64}(
        'A' => 1.8, 'R' => -4.5, 'N' => -3.5, 'D' => -3.5, 'C' => 2.5,
        'Q' => -3.5, 'E' => -3.5, 'G' => -0.4, 'H' => -3.2, 'I' => 4.5,
        'L' => 3.8, 'K' => -3.9, 'M' => 1.9, 'F' => 2.8, 'P' => -1.6,
        'S' => -0.8, 'T' => -0.7, 'W' => -0.9, 'Y' => -1.3, 'V' => 4.2
    )

    # Typical pKa values for ionizable groups (used for net charge)
    pKa_values = Dict{Char, Float64}(
        'D' => 3.9, 'E' => 4.3, 'H' => 6.0, 'K' => 10.5, 'R' => 12.5,
        'C' => 8.3, 'Y' => 10.1
    )
    pKa_N_terminus = 9.5
    pKa_C_terminus = 2.2

    # --- Initial Checks ---
    if isempty(peptide_sequence)
        return 0.0
    end

    # --- Property Calculations ---
    total_hydropathy = 0.0
    num_residues = 0
    net_charge = 0.0

    # N and C termini charge contribution
    net_charge += 1 / (1 + 10^(pH - pKa_N_terminus))
    net_charge += -1 / (1 + 10^(pKa_C_terminus - pH))

    for aa_char in peptide_sequence
        # GRAVY calculation
        if haskey(gravy_scales, aa_char)
            total_hydropathy += gravy_scales[aa_char]
            num_residues += 1
        end

        # Net charge calculation for side chains
        if haskey(pKa_values, aa_char)
            pKa = pKa_values[aa_char]
            if aa_char in ('D', 'E', 'C', 'Y') # Acidic groups
                net_charge += -1 / (1 + 10^(pKa - pH))
            elseif aa_char in ('H', 'K', 'R') # Basic groups
                net_charge += 1 / (1 + 10^(pH - pKa))
            end
        end
    end

    gravy = num_residues > 0 ? total_hydropathy / num_residues : 0.0

    # --- Scoring Components ---
    # GRAVY score component: higher if within ideal range, lower if outside
    gravy_min, gravy_max = ideal_gravy_range
    gravy_score_component = 0.0
    if gravy_min <= gravy <= gravy_max
        gravy_score_component = 1.0
    elseif gravy < gravy_min
        val = max(gravy, gravy_cap_low)
        gravy_score_component = (val - gravy_min) / (gravy_min - gravy_cap_low)
        gravy_score_component = clamp(gravy_score_component, 0.0, 1.0)
    else # gravy > gravy_max
        val = min(gravy, gravy_cap_high)
        gravy_score_component = (gravy_max - val) / (gravy_cap_high - gravy_max)
        gravy_score_component = clamp(gravy_score_component, 0.0, 1.0)
    end

    # Charge score component: penalize near-zero charge, reward sufficient absolute charge
    abs_net_charge = abs(net_charge)
    charge_score_component = 0.0

    if min_abs_charge_threshold <= abs_net_charge <= max_abs_charge_threshold
        charge_score_component = 1.0
    elseif abs_net_charge < min_abs_charge_threshold
        # Score increases linearly from 0.0 at 0 charge to 1.0 at min_abs_charge_threshold
        charge_score_component = abs_net_charge / min_abs_charge_threshold
        charge_score_component = clamp(charge_score_component, 0.0, 1.0)
    else # abs_net_charge > max_abs_charge_threshold
        # Score decreases linearly from 1.0 at max_abs_charge_threshold to 0.0 at abs_charge_cap_high
        val = min(abs_net_charge, abs_charge_cap_high)
        charge_score_component = (max_abs_charge_threshold - val) / (abs_charge_cap_high - max_abs_charge_threshold)
        charge_score_component = clamp(charge_score_component, 0.0, 1.0)
    end

    # --- Combine Scores ---
    final_score = (weight_gravy * gravy_score_component) + (weight_charge * charge_score_component)
    return final_score
end
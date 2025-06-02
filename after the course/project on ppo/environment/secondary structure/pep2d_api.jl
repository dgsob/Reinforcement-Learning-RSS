# Function to get secondary structure propensity (placeholder for PEP2D)
# Note: PEP2D lacks a direct API; this is a placeholder until integrated
function get_secondary_structure(sequence::String)
    @warn "PEP2D integration is not yet implemented. Returning placeholder value."
    return 0.0  # Placeholder; replace with actual PEP2D call when available
end

export get_secondary_structure
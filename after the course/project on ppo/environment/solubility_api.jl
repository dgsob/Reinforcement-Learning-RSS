using HTTP
using Gumbo
using Cascadia

function get_timestamp(html_response::String)
    try
        timestamp_match = match(r"var timestamp = \"([a-f0-9]+)\";", html_response)
        if timestamp_match !== nothing
            return timestamp_match.captures[1]
        else 
            @error "Timestamp retrieval from Protein-Sol HTTP response failed."
        end
    catch e
        @error "$e"
    end
end

function get_redirect_url(html_response::String, timestamp::String)
    try
        url_match = match(r"window\.location = \"(https?:\/\/[^\"]*?)\"\+timestamp\+\"([^\"]*)\";", html_response)
        if url_match !== nothing && !isempty(timestamp)
            base_url_part1 = url_match.captures[1]
            base_url_part2 = url_match.captures[2]
            return base_url_part1 * timestamp * base_url_part2
        else
            @error "Redirect URL retrieval from Protein-Sol HTTP response failed."
        end
    catch e
        @error "$e"
    end
end

function get_solubility(sequence::String)
    # Protein-Sol submission URL
    submit_url = "https://protein-sol.manchester.ac.uk/cgi-bin/solubility/sequenceprediction.php"

    # Prepare form data for submission
    headers = ["Content-Type" => "application/x-www-form-urlencoded"]
    body = "sequence-input=$(HTTP.escapeuri(sequence))&singleprediction=Submit"

    try
        # Submit the sequence and get the initial response
        response = HTTP.post(submit_url, headers, body; redirect=false, require_ssl_verification=false)
        response_body_str = String(response.body)
        # @debug "Initial response: $response_body_str"
        # Check for redirect in the response
        if response.status == 200 && occursin("window.location", response_body_str)
            # Extract the timestamp
            timestamp = get_timestamp(response_body_str)
            # Extract the base URL and substitute the timestamp
            redirect_url = get_redirect_url(response_body_str, timestamp)
            @debug "Redirect url: $redirect_url"

            if redirect_url !== nothing
                # Retry mechanism to handle server processing time
                max_attempts = 5
                for attempt in 1:max_attempts
                    try
                        results_response = HTTP.get(redirect_url; retries=2, require_ssl_verification=false)
                        results_body_str = String(results_response.body)
                        # @debug "Results response (Attempt $attempt): $results_body_str"

                        # Parse the results HTML
                        html = parsehtml(results_body_str)
                        selector = Selector(".protein > p:nth-child(5)")  # CSS selector for the result value
                        elements = eachmatch(selector, html.root)
                        
                        if !isempty(elements)
                            text = strip(nodeText(elements[1]))
                            @debug "Extracted text: $text" 
                            if text !== nothing
                                score = parse(Float64, text)
                                @info "Predicted scaled solubility from Protein-Sol: $score"
                                return score
                            end
                        else
                            @warn "No elements found with selector '.protein > p:nth-child(5)'. Attempt $attempt of $max_attempts."
                        end
                    catch e
                        @warn "Attempt $attempt failed: $e. Retrying in 2 seconds..."
                        sleep(2)
                    end
                end
            end
        end
        @warn "Could not parse solubility score from response. Returning fallback value. Check HTML structure."
        return 0.0  # Fallback if parsing fails
    catch e
        @error "Error querying Protein-Sol: $e"
        return 0.0  # Fallback on error
    end
end

export get_solubility
function evaluate_mse(
    r::CNMF_results
)
    return compute_loss(r.data, r.W, r.H)
end


function evaluate_test(
    r::CNMF_results,
    test::Array{Float64, 2};
    num_iter=30
)
    # Fit a new H on the test set
    testH = zeros(num_components(r), size(test, 2))
    
    
    meta = HALS.HALSMeta(test, r.W, testH)
    HALS._setup_H_update!(r.W, r.H, meta)
    
    for iter in 1:num_iter
        HALS._update_H_regular!(r.W, testH, meta.resids, meta.Wk_list, meta.W_norms, 0, 0)
    end

    return compute_loss(test, r.W, testH)
end
    


function evaluate_convergence(
    r::CNMF_results;
    thresh=0.01
)
    min_loss = r.loss_hist[end]

    i = 0
    for loss in r.loss_hist
        if (loss / min_loss < 1 + thresh)
            return i
        end
        i += 1
    end

    return length(loss_hist)
end

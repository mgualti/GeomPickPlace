function p = TruncatedNormalCdf(x, mu, sigma, a, b)

    epsilon = (x - mu) ./ sigma;
    alpha = (a - mu) ./ sigma;
    beta = (b - mu) ./ sigma;
    Z = phi(beta) - phi(alpha);
    p = (phi(epsilon) - phi(alpha)) ./ Z;

function y = phi(x)
    y = 0.5 .* (1 + erf(x ./ sqrt(2)));
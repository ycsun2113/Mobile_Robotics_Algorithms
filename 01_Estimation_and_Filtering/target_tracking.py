import numpy as np


def target_tracking():
    # known parameters
    mu0 = 20
    sigma0_square = 9
    F = 1
    Q = 4
    H = 1
    R = 1
    z1 = 22
    z2 = 23

    mu2 = 0
    sigma2_square = 0

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # prediction step (1st iteration)
    mu1_hat = F * mu0 
    sigma1_square_hat = F * sigma0_square * F + Q

    # correction step (1st iteration)
    K1 = sigma1_square_hat * H / (H * sigma1_square_hat * H + R)
    mu1 = mu1_hat + K1 * (z1 - H * mu1_hat)
    sigma1_square = (1 - K1 * H) * sigma1_square_hat

    # prediction step (2nd iteration)
    mu2_hat = F * mu1
    sigma2_square_hat = F * sigma1_square * F + Q

    # correction step (2nd iteration)
    K2 = sigma2_square_hat * H / (H * sigma2_square_hat * H + R)
    mu2 = mu2_hat + K2 * (z2 - H * mu2_hat)
    sigma2_square = (1 - K2 * H) * sigma2_square_hat

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return (mu2, sigma2_square)


if __name__ == '__main__':
    # Test your funtions here

    print('Answer for Problem 3:\n', target_tracking())

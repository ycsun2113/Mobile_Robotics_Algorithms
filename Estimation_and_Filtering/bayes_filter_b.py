import numpy as np
import matplotlib.pyplot as plt

# colors
green = np.array([0.2980, 0.6, 0])
darkblue = np.array([0, 0.2, 0.4])
VermillionRed = np.array([156, 31, 46]) / 255


def plot_fuction(belief, prediction, posterior_belief):
    """
    plot prior belief, prediction after action, and posterior belief after measurement
    """
    fig = plt.figure()

    # plot prior belief
    ax1 = plt.subplot(311)
    plt.bar(np.arange(0, 10), belief.reshape(-1), color=darkblue)
    plt.title(r'Prior Belief')
    plt.ylim(0, 1)
    plt.ylabel(r'$bel(x_{t-1})$')

    # plot likelihood
    ax2 = plt.subplot(312)
    plt.bar(np.arange(0, 10), prediction.reshape(-1), color=green)
    plt.title(r'Prediction After Action')
    plt.ylim(0, 1)
    plt.ylabel(r'$\overline{bel(x_t})}$')

    # plot posterior belief
    ax3 = plt.subplot(313)
    plt.bar(np.arange(0, 10), posterior_belief.reshape(-1), color=VermillionRed)
    plt.title(r'Posterior Belief After Measurement')
    plt.ylim(0, 1)
    plt.ylabel(r'$bel(x_t})$')

    plt.show()


def bayes_filter_b():
    """
    Follow steps of Bayes filter.  
    You can use the plot_fuction() above to help you check the belief in each step.
    Please print out the final answer.
    """

    # Initialize belief uniformly
    belief = 0.1 * np.ones(10)

    posterior_belief = np.zeros(10)
    #############################################################################
    #                    TODO: Implement you code here                          #
    #############################################################################

    print_results = False

    # sensor model
    p_zlandmark_given_x036 = 0.8
    p_znothing_given_x036 = 1 - p_zlandmark_given_x036
    p_zlandmark_given_xother = 0.4
    p_znothing_given_xother = 1 - p_zlandmark_given_xother
    p_zlandmark_given_x = np.array([0.8, 0.4, 0.4, 0.8, 0.4, 0.4, 0.8, 0.4, 0.4, 0.4])
    p_znothing_given_x = 1 - p_zlandmark_given_x

    # prior
    p_x036 = 0.3
    p_xother = 0.7

    # 1. The robot detects a landmark

    # update step
    belief_1 = p_zlandmark_given_x * belief

    # normalizaton
    nu = 1 / np.sum(belief_1)
    belief_1 = belief_1 * nu

    # plt.bar(np.arange(0, 10), belief_1)
    # plt.show()

    # 2. moves 3 grid cells counterclockwise and detects a landmark

    # prediction step
    belief_2_bar = np.zeros(10)
    for i in range(10):
        belief_2_bar[(i+3)%10] = belief_1[i]
    
    # plt.bar(np.arange(0, 10), belief_2_bar)
    # plt.show()

    # update step
    belief_2 = p_zlandmark_given_x * belief_2_bar

    # normalizaton
    nu = 1 / np.sum(belief_2)
    belief_2 = belief_2 * nu

    # plt.bar(np.arange(0, 10), belief_2)
    # plt.show()

    # plot_fuction(belief_1, belief_2_bar, belief_2)

    # 3. then moves 4 grid cells counterclockwise and finally perceives no landmark

    # prediction step
    belief_3_bar = np.zeros(10)
    for i in range(10):
        belief_3_bar[(i+4)%10] = belief_2[i]
    
    # plt.bar(np.arange(0, 10), belief_3_bar)
    # plt.show()

    # update step
    belief_3 = p_znothing_given_x * belief_3_bar

    # normalizaton
    nu = 1 / np.sum(belief_3)
    belief_3 = belief_3 * nu

    # print the results in each step:
    if print_results:
        print('belief 1:')
        print(belief_1)

        print('belief 2 bar:')
        print(belief_2_bar)

        print('belief 2:')
        print(belief_2)

        print('belief 3 bar:')
        print(belief_3_bar)

        print('belief 3:')
        print(belief_3)

        print('belief sum:')
        print(np.sum(belief_3))

    posterior_belief = belief_3

    # plt.bar(np.arange(0, 10), belief_3)
    # plt.show()

    # plot_fuction(belief_2, belief_3_bar, belief_3)


    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return posterior_belief


if __name__ == '__main__':
    # Test your funtions here
    belief = bayes_filter_b()
    print('Answer for Problem 2b:')
    for i in range(10):
        print("%6d %18.3f\n" % (i, belief[i]))
    plt.bar(np.arange(0, 10), belief)

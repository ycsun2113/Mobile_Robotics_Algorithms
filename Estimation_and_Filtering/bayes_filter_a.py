import numpy as np


def bayes_filter_a(
        p_xcolor_given_xblank_upaint,
        p_zcolor_given_xblank,
        p_zcolor_given_xcolor
):
    """
    p_xcolor_given_xblank_upaint: p(x_t+1=colored|x_t=blank,u_t+1=paint)
    p_zcolor_given_xblank: p(z=colored|x=blank)
    p_zcolor_given_xcolor: p(z=colored|x=colored)
    """

    bel_x1_blank = 0
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    # note: In this question, you may choose to input the numerical results.
    #       Make sure your answers are accurate to three decimal places.

    # since we have no prior knowledge,
    # we assume that the initial belief is equally distributed
    bel_x0_color = 0.5
    bel_x0_blank = 0.5

    # action model
    # p_xcolor_given_xblank_upaint is given
    p_xblank_given_xblank_upaint = 1 - p_xcolor_given_xblank_upaint
    p_xcolor_given_xcolor_upaint = 1
    p_xblank_given_xcolor_upaint = 1 - p_xcolor_given_xcolor_upaint

    # sensor model
    # p_zcolor_given_xblank is given
    p_zblank_given_xblank = 1 - p_zcolor_given_xblank
    # p_zcolor_given_xcolor is given
    p_zblank_given_xcolor = 1 - p_zcolor_given_xcolor

    # prediction step
    bel_x1_blank = p_xblank_given_xblank_upaint * bel_x0_blank + p_xblank_given_xcolor_upaint * bel_x0_color
    bel_x1_color = p_xcolor_given_xblank_upaint * bel_x0_blank + p_xcolor_given_xcolor_upaint * bel_x0_color
    # print(bel_x1_blank)
    # print(bel_x1_color)

    # correction step
    bel_x1_blank = p_zcolor_given_xblank * bel_x1_blank
    bel_x1_color = p_zcolor_given_xcolor * bel_x1_color
    # print(bel_x1_blank)
    # print(bel_x1_color)

    # normalization
    nu = 1 / (bel_x1_blank + bel_x1_color)
    bel_x1_blank = bel_x1_blank * nu
    bel_x1_color = bel_x1_color * nu
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return bel_x1_blank


if __name__ == '__main__':
    # Test your funtions here
    print('Answer for Problem 2a:\n', bayes_filter_a(0.9, 0.2, 0.7))

'''
Info: Replication the result of a published paper 

Paper: https://www.frontiersin.org/articles/10.3389/frobt.2015.00027/full
Original repo: https://github.com/tgenewein/BoundedRationalityAbstractionAndHierarchicalDecisionMaking

@Zeming

General convections: 

1. The perception model follows O --> S --> A
    - O stands for observation
    - S stands for mental believed state
    - A stands for the action

2. for vector, I usually use the column convection, which means Nx1

3. for probability distribution, I use
    - p for generic notation of probiity
    - psi for perception and state encoder
    - pi for policy state to action 

4. understanding the conditional probability variable
    - psi_s1o means the probability of state given obs, 1 means |
      the first var will be col, and the secod var will be row. I know this is a 
      little bit counter-intuitive, but I feel hard to change my habbits.
    - to differentiate conditional and joint distribution,  I will not include 1,
      for example, I will use p_so to show joint distirbution. 

5. about variable sequence. When comming across multiple variables,
    I will sort them in the following sequnece
    - highest priority -1: Util_matrix 
    - priority -2: O observation
    - priority -3: S internal mental state
    - lowest priority -4: action 

'''
import os 
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.special import logsumexp # for partition function
from matplotlib.widgets import Slider # interact plot 
 
# define the saving path
path = os.path.dirname(os.path.abspath(__file__))

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SEC0: BASIC FUNCTION   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def I( p_x, p_y1x):
    '''MUTUAL INFORMATION

    Inputs:

        p_x:   sender distribution nX x 1
        p_y1x: channal nX x nY
    '''
    p_y   = p_x.T @ p_y1x
    H_y   = -np.sum( p_y * np.log( p_y + 1e-20))
    H_y1x = -np.sum( p_x * p_y1x * np.log( p_y1x + 1e-20)) 
    return H_y - H_y1x

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SEC1: ENV FUNCTION   %
%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

## PREDATOR PREY ENV 

def setup_predator_prey_example(mating_Utility=False):
    '''SET UP ENV

    1. PREDATOR-PREY 

    ACT VARS: 
        a1: wait and attack
        a2: stalk and attack
        a3: flee

    UTILITY MATRIX:
        small w: 
            a1++    might not come towards you 
            a2+++   will not hear you 
            a3-     no food
        medium w:
            a1++    might not come towards you 
            a2+     hear you and flee
            a3-     no food
        large w:
            a1--    die
            a2--    die
            a3++++  survive
            
    '''
    # set up observation 

    obs_vals = [ 2, 3, 4, 6, 7, 8, 10, 11, 12] # opponent size 
    nO = len(obs_vals)                         # obs' cardinality
    obs_vars = [ str(obs_val) for obs_val in obs_vals] # obs' semantic meaning
    p_obs = np.ones([ nO, 1]) / nO             # obs distribution  

    # set up action 
    if mating_Utility:
        act_vars = [ 'display', 'flee']
        act_vals = [ 400, 500]
    else:
        act_vars = [ f'sneak up w={sz}' for sz in obs_vals[0:3]] + \
                    [ f'ambush w={sz}' for sz in obs_vals[3:6]] + \
                        [ 'ambush', 'sneak up', 'flee'] # obs' semantic meaning
        act_vals = obs_vals[ 0: 6] + [ 100, 200, 300]   # action utility

    if mating_Utility:
        pass
    else:
        def utility_fn( obs, act):
            # set val 
            sur_util = 5
            best_hunt_util = 3.5
            sneak_small_util = 3
            ambush_util = 2.3
            sneak_medium_util = 1.5
            flee_small_medium_util = .5
            eaten_util = 0.

            # for small group, the best act is sneakup, 
            # for each size, there is a best special sneak up skills
            if (obs < 5) and (act==obs):
                return best_hunt_util

            # for medium group, the best act is ambush,
            # for specific size, there is a best ambush skills
            if (obs < 9) and (act==obs):
                return best_hunt_util

            # for both small, medium, there is a generic sneak skill 
            # for small, sneak up +++
            # for medium, sneak up +
            if act == act_vals[-2]:
                if obs < 5:
                    return sneak_small_util
                elif (5 < obs) and (obs < 9):
                    return sneak_medium_util
            
            # for both small, medium, there is a generic ambush skill
            # for small, ambush ++
            # for medium, ambush ++
            if (act == act_vals[-3]) and (obs < 9):
                return ambush_util

            # for both small, medium, flee is a bad choice, because there
            # is not food, flee /.+ 
            if (act == act_vals[-1]) and (obs < 9):
                return flee_small_medium_util

            # for large, flee is the only choice
            if (9 < obs):
                if (act == act_vals[-1]):
                    return sur_util
                else:
                    return eaten_util

            # for small group, the best act is sneakup, 
            # for each size, there is a best special sneak up skills
            # if the wrong specific act is used in small group, effect * 80%
            # if the wrong specific act is used in medium group, equals to generic 
            if (obs < 5):
                if (act < 5):
                    return sneak_small_util * .8
                else:
                    return ambush_util
            
            if (obs < 9):
                if (act < 5):
                    return sneak_medium_util
                else:
                    return ambush_util * .8

    # make util table 
    util_mat = make_util_mat( utility_fn, obs_vals, act_vals)

    return obs_vals, obs_vars, p_obs, act_vals, act_vars, util_mat

## medical system env

def setup_medical_example(uniform_obs=True):
    '''DISEASE AND TREATMENT

    There are three kinds disease H, L12, L34
    each diseases has two sub types of diseases,

    For each specific type, the specific treatment works the best (high utilty)
    Within the same kind of dissease, cross type treatment is ok but less effective (medium utility)
    Cross kind treatment results in bad results (low utility)

    There are also general treatments for each disease, h, l12, l34, l

    '''
    # set up observations
    obs_vars = [ 'h1', 'h2', 'l1', 'l2', 'l3', 'l4']  # obs' semantic meaning
    nO = len(obs_vars)                                # obs' cardinality
    obs_vals = np.arange(1, nO+1)                     # opponent size 
    if uniform_obs:
        p_obs = np.ones([ nO, 1]) / nO                # obs distribution 
    else:
        p_obs = np.ones( [nO, 1])                     # load the non uniform
        p_obs[ 0:2, 0] = 3
        p_obs = p_obs / np.sum( p_obs)                # normalize 
    
    # set up actions
    act_vars = [ f'treat={o}' for o in obs_vars ] \
                + [ 'treat=l12', 'treat=l34', 'treat=h', 'treat=l'] 
    nA = len( act_vars)
    act_vals = np.arange( 1, nA+1)

    def utility_fn( obs, act):
        correct_util        = 3                  # correct 
        wrong_heart_util    = correct_util * .3  # 
        general_heart_util  = 1.5 
        wrong_lung_util1    = correct_util * .5
        wrong_lung_util2    = correct_util * 0. 
        general_lung_util   = 1.5
        general_lung_util12 = 2.5
        general_lung_util34 = 2.5

        # correct treatment
        if obs == act:
            return correct_util

        # heart-disease, within kind wrong treatment 
        if ( obs < 3) and ( act < 3):
            return wrong_heart_util

        # lung-disease12, wrong treatment 
        if ( 2 < obs) and ( obs < 5) and ( 2 < act) and ( act < 5):
            return wrong_lung_util1

        # lung-disease34, lung 
        if ( 2 < obs) and ( obs < 5) and ( 4 < act) and ( act < 7):
            return wrong_lung_util2 

        # lung-disease12, wrong treatment 
        if ( 4 < obs) and ( obs < 7) and ( 4 < act) and ( act < 7):
            return wrong_lung_util1

        # lung-disease34, lung 
        if ( 4 < obs) and ( obs < 7) and ( 2 < act) and ( act < 5):
            return wrong_lung_util2 
        
        # general heart treatment
        if ( obs < 3) and ( act == 9):
            return general_heart_util

        # general lung treatments 
        if ( 2 < obs) and ( obs < 7):
            if ( act == 7) and ( obs < 5):
                return general_lung_util12
            if ( act == 8) and ( 4 < obs):
                return general_lung_util34
            if act == 10:
                return general_lung_util
        
        # wrong treatment for wrong cause 
        return 0 

    util_mat = make_util_mat( utility_fn, obs_vals, act_vals)

    return obs_vals, obs_vars, p_obs, act_vals, act_vars, util_mat

# from utility function to utility matrix 
def make_util_mat( utility_fn, obs_vals, act_vals):
    '''MAKE UTILITY MATRIX 
    '''
    util_mat = np.zeros( [len(obs_vals), len(act_vals)])
    for o_idx, o in enumerate(obs_vals):
        for a_idx, a in enumerate(act_vals):
            util_mat[ o_idx, a_idx] = utility_fn( o, a)
    return util_mat

# handcrafted state encode and p(s|o)
def psi_hand( obs_vals, state_vals, lamb):
    '''STATE ENCODER p(s|o,λ)

    state is the noisy peception of observation
    The distribution is approximate using the sampling method.
    '''

    nO = len( obs_vals)
    nS = len( state_vals)
    qs1o = np.zeros( [nO, nS])
    nsamples = 5000 # num of samples used to collect data 

    # The implementation of sampling is reject sampling 
    # accept the reasonable state, in fact this is a epsilon-greedy
    for io in range(nO):
        idx = 0
        qs1o_samples = np.zeros( [nsamples,])
        while idx < nsamples-1:
            sample= np.round( obs_vals[io] + np.random.randn(1)/lamb)
            if (sample>0) and (sample<nS+1):
                qs1o_samples[idx+1] = sample
                idx +=1 
    
        # count frequencies over state and 
        # noramlize it as probability distribution
        bins = np.arange(.5, nS+1.5)
        freq, _ = np.histogram( qs1o_samples, bins)
        qs1o[ io, :] = freq / np.sum( freq)

    return qs1o 

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SEC2: BA ALGS and VARIANTS   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def BA_algs( util_mat, p_x, q_y1x, 
             beta,
             tol = 1e-4, max_iter=10000):
    '''BLAHUT ARIMOTO ALGORITHM

    px --> NX x 1: 
        sender's distritbuion
    py --> NY x 1:    
        receiver's distribution
    util_mat --> NX x NY:
        utility matrix  
    beta --> scalar: 
        inverse teperature
    tol  --> scalar:
        tolerance for convergence checking
    max_iter --> scalar:
        maximum iteration number
    '''
    # init for iteration
    i       = 0
    done    = False
    p_y     = p_x.T @ q_y1x

    while not done:
        # cache data to check convergence
        old_q_y1x = q_y1x
        # update the channel p(y|x)
        log_q_y1x = beta * util_mat + np.log( p_y + 1e-20)
        q_y1x = np.exp( log_q_y1x - logsumexp( log_q_y1x, axis=1, keepdims=True)) 
        # update the marginal policy
        p_y = p_x.T @ q_y1x    
        # update counter
        i += 1
        # check convergence 
        if np.sum(abs(old_q_y1x - q_y1x)) < tol:
            done = True 
        if i >= max_iter:
            print( f'BA alg reached maximum iteration {max_iter}, results might be inaccurate')
            done = True
    
    return q_y1x, p_y.T 

def get_pi_a1s( util_mat, psi_s1o, p_a1os, 
                p_o, p_s, p_a,
                beta2, beta3):
    '''COMPUTE π(a|s)
    '''
    # inference observation given mental state using bayes rule
    # ψ(o|s) = ψ（s|o)p(o)/p(s) : nSxnO
    psi_o1s = (p_o * psi_s1o / p_s.T).T
    
    if beta3 == 0:
        '''
        This is the special case, because S block O-->A
        O --> S --> A
        '''
        # bel_U(s,a) = ∑_o ψ(s|o)U(o,a)
        bel_util = psi_o1s @ util_mat
        # π(a|s) ∝ p(a)exp( β2*Bel_U(s,a)) nSxnA
        log_pi_a1s = beta2 * bel_util + np.log( p_a.T + 1e-20)
        pi_a1s = np.exp( log_pi_a1s - logsumexp( log_pi_a1s, axis=-1, keepdims=True))  
    else: 
        '''
        A more general case 

        This time, O --> S
                    \   /
                      v    
                      A 
        '''
        # π(a|s) = ∑_o ψ(s|o)p(a|o,s) nSxnA
        '''
        !!!!!!!!!!!!!!!!!!!!!!This could be problematic !!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        pi_a1s = np.sum( p_a1os * psi_o1s.T[ :, :, np.newaxis], axis=0) + 1e-20
        pi_a1s = pi_a1s / np.sum( pi_a1s ,axis=-1, keepdims=True)
        
    return pi_a1s

def get_p_a1os( util_mat, pi_a1s, p_a,
                beta2, beta3):
    '''COMPUTE p(a|o,s)
    '''
    nO = util_mat.shape[0]
    nA = util_mat.shape[1]
    nS = pi_a1s.shape[0]

    if beta3 == 0:
        p_a1os = np.zeros( [ nO, nS, nA])
        for oi in range( nO):
            p_a1os[ oi, :, :] = pi_a1s
    else:
        # p(a|o,s) ∝ π(a|s) exp( β3 U(o,a) - β3/β2 log（π(a|s)/p(a))
        log_p_a1os = beta3 * util_mat[ :, np.newaxis, :] \
                     - beta3 / beta2 * np.log( pi_a1s[ np.newaxis, :, :] + 1e-20)\
                     + beta3 / beta2 * np.log( p_a.reshape([-1])[ np.newaxis, np.newaxis, :] + 1e-20)\
                     + np.log( pi_a1s[ np.newaxis, :, :] + 1e-20)
        p_a1os = np.exp( log_p_a1os - logsumexp( log_p_a1os, axis=-1, keepdims=True))

    return p_a1os 

def get_psi_s1o( util_mat, pi_a1s, p_a1os, 
                 p_s, p_a, 
                 beta1, beta2, beta3):
    '''COMPUTE ψ(s|o)
    '''
    if beta3 == 0:
        # compute EU(o,s): ∑_a π(a|s)U(o,a) nO x nS
        EU =  util_mat @ pi_a1s.T 
        # compute D[π(a|s)||p(a)] 1 x nS 
        DKL1 = np.sum( pi_a1s * np.log( pi_a1s + 1e-20) 
               - pi_a1s * np.log( p_a.T + 1e-20), axis=-1).reshape([1,-1])
    else:
        # compute EU(o,s): ∑_a p(a|o,s)U(o,a) nOxnS
        EU = np.sum( util_mat[ :, np.newaxis, :] * p_a1os, axis=-1)
        # compute D[p(a|o,s)||p(a)] nOxnS 
        DKL1 = np.sum( p_a1os * np.log( p_a1os + 1e-20) 
               - p_a1os * np.log( p_a.reshape([-1,])[ np.newaxis, np.newaxis, :] + 1e-20), axis=-1)
    
    # compute D[p(a|o,s)||π(a|s)] nOxnS
    DKL2 = np.sum( p_a1os * np.log( p_a1os + 1e-20)
               - p_a1os * np.log( pi_a1s[ np.newaxis, :, :] + 1e-20), axis=-1)
    
    if beta3 == 0:
        if np.sum( DKL2) > 0:
            raise Exception( 'In sequntial case, p(a|o,s) should equal to π(a|s)')
        # Fser = EU(o,s) - 1/β2 D[π(a|s)||p(a)], nOxnS
        F = EU - 1/beta2 * DKL1    
    else: 
        # Fpar = EU(o,s) - 1/β2 D[p(a|o,s)||p(a)] - (1/β3 - 1/β2) * D[p(a|o,s)||π(a|s)] nOxnS
        F = EU - 1/beta2 * DKL1 - ( 1/beta3 - 1/beta2) * DKL2

    # ψ(s|o) ∝ p(s)exp( β1 F )
    log_psi_s1o = beta1 * F + np.log( p_s.T + 1e-20)
    psi_s1o = np.exp( log_psi_s1o - logsumexp( log_psi_s1o, axis=-1, keepdims=True))

    return psi_s1o

def general_BA_algs( util_matrix, p_o, psi_s1o, p_a1os,
                     beta1, beta2, beta3,
                     tol, max_iter):
    '''General BA ALG

    The original BA algorithm can only be applied
    to one channel. It the architecture includes
    multiple channels, either cascade or parallel
    structure. This algorithm is introduced by the 
    titled paper as a general solution to the multi
    channel structure

    The shape of some vars

    p_o: nO x 1
    p_s: nS x 1
    p_a: nA x 1 
    psi_s1o: nO x nS (see the convenction disclamer at 
                    the beginning for more details)
    psi_o1s: nS x nO 
    pi_a1s:  nS x nA 
    p_a1os: nO x nS x nA

    '''
    # reshape the vector and compute all we need for iteration, 
    # they are not necessarily need to be correct, because
    # this is just for initialization

    # compute p_s, p_a for initialization, p_o is given
    # p_o                      # nOx1
    p_s = (p_o.T @ psi_s1o).T  # nSx1
    # p(a|o) = ∑_s ψ(s|o)p(a|o,s) 
    p_a1o = np.sum( psi_s1o[ :, :, np.newaxis] * p_a1os, axis=1)
    p_a = (p_o.T @ p_a1o).T    # nAx1

    ## compute π(a|s) or π(a|s,o) depends on case
    pi_a1s = get_pi_a1s( util_matrix, psi_s1o, p_a1os, 
                    p_o, p_s, p_a, 
                    beta2, beta3)
    if beta3 == 0:
        # sequential case, make sure p(a|o,s) = π(a|s)
        p_a1os = get_p_a1os( util_matrix, pi_a1s, p_a,
                             beta2, beta3)
    
    # start iteration
    done = False 
    i = 0
    while not done:

        # cache the current val for convergence checks
        old_p_a1os   = p_a1os
        old_psi_s1o  = psi_s1o

        # follow the sequence of the original paper
        
        # update ψ(s|o)
        psi_s1o = get_psi_s1o( util_matrix, 
                               pi_a1s, p_a1os, p_s, p_a,
                               beta1, beta2, beta3)

        if beta3 == 0:
            # update π(a|s)
            pi_a1s = get_pi_a1s( util_matrix, psi_s1o, p_a1os,
                                    p_o, p_s, p_a,
                                    beta2, beta3)
            # update p(a|o,s)
            p_a1os = get_p_a1os( util_matrix, pi_a1s, p_a, 
                                    beta2, beta3)
        else:
            # update p(a|o,s)
            p_a1os = get_p_a1os( util_matrix, pi_a1s, p_a, 
                                    beta2, beta3)

            # update π(a|s)
            pi_a1s = get_pi_a1s( util_matrix, psi_s1o, p_a1os,
                                    p_o, p_s, p_a,
                                    beta2, beta3)
            
        # update marginal p(s), p(a), when calculating mariginal policy
        # we add a small value to prevent the distribution to becomes 0
        # note that this is very important, during iteration.
        p_s = (p_o.T @ psi_s1o).T + 1e-20 
        p_s = p_s / np.sum(p_s)           # nSx1 
        # p(a|o) = ∑_s ψ(s|o)p(a|o,s) 
        p_a1o = np.sum( psi_s1o[ :, :, np.newaxis] * p_a1os, axis=1) # nOxnA
        p_a = (p_o.T @ p_a1o).T + 1e-20
        p_a = p_a / np.sum( p_a)          # nAx1

        # update counter
        i += 1

        # check convergence 
        if np.sum(abs(p_a1os - old_p_a1os)) + np.sum(abs(psi_s1o - old_psi_s1o)) < tol:
            done = True 
        if i >= max_iter:
            print( f'General BA alg reached maximum iteration {max_iter}, results might be inaccurate')
            done = True        

    return psi_s1o, pi_a1s, p_a1os, p_s, p_a

        
'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       SEC3: ILLUSTRATION       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def show_predator_prey_util() :
    _, obs_vars, _, _, act_vars, util_mat = setup_predator_prey_example()
    
    plt.figure( figsize=( 21, 7))
    plt.subplot( 1,2,1)
    plt.imshow( util_mat.T, cmap='Blues', origin='lower')
    plt.title('Utility of predator prey')
    plt.xticks( np.arange(len(obs_vars))+.5, obs_vars)
    plt.yticks( np.arange(len(act_vars))+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.xlabel( 'observed animal size')
    plt.ylabel( 'action')
    plt.colorbar()
    plt.subplot( 1,2,2)
    log_pi = 100*util_mat
    opt_pi = np.exp( log_pi - logsumexp( log_pi, axis=1, keepdims=True))
    plt.imshow( opt_pi.T, cmap='Blues', origin='lower'
                , vmin=0, vmax=1)
    plt.title('Optimal policy')
    plt.xticks( np.arange(len(obs_vars))+.5, obs_vars)
    plt.yticks( np.arange(len(act_vars))+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.xlabel( 'observed animal size')
    plt.ylabel( 'action')
    plt.colorbar()
    fig_name = f'{path}/figures/predator_prey_utility.png'
    try:
        plt.savefig( fig_name)
    except:
        os.mkdir(f'{path}/figures')
        plt.savefig

def show_medical_util():
    _, obs_vars, _, _, act_vars, util_mat = setup_medical_example()
    
    plt.figure( figsize=( 21, 7))
    plt.subplot( 1,2,1)
    plt.imshow( util_mat.T, cmap='Blues', origin='lower')
    plt.title( 'Utility of medical example')
    plt.xticks( np.arange(len(obs_vars))+.5, obs_vars)
    plt.yticks( np.arange(len(act_vars))+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.xlabel( 'Disease type')
    plt.ylabel( 'Treatment')
    plt.colorbar()
    plt.subplot( 1,2,2)
    log_pi = 100*util_mat
    opt_pi = np.exp( log_pi - logsumexp( log_pi, axis=1, keepdims=True))
    plt.imshow( opt_pi.T, cmap='Blues', origin='lower'
                , vmin=0, vmax=1)
    plt.title('Optimal policy')
    plt.xticks( np.arange(len(obs_vars))+.5, obs_vars)
    plt.yticks( np.arange(len(act_vars))+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.xlabel( 'Disease type')
    plt.ylabel( 'Treatment')
    plt.colorbar()
    fig_name = f'{path}/figures/medical_utility.png'
    try:
        plt.savefig( fig_name)
    except:
        os.mkdir(f'{path}/figures')
        plt.savefig

def illustrate_cascade_channel( lamb, beta1, beta2, beta3):
    '''REPLICATE THE FIG6

    This function replicate fig to illustrate
    how cascade channel may work, the env is 
    the prey pradator environment.
    '''
    # set hyperparameters
    # lamb : precision of the hand-crafted perceptual model
    # beta1: price for I(O;S)
    # beta2: price for I(S;A)
    # beta3: price for I(A;S,O)
    tol   = 1e-4  # tolerance for convergence 
    max_iter = 10000 # maximum number of BA iterations

    # load env and perception 
    obs_vals, obs_vars, p_o, act_vals, act_vars, util_mat = setup_predator_prey_example()

    # init the internal representation state 
    state_vals  = np.arange( 1, 14)
    state_vars  = [ str(s) for s in state_vals]

    # obtain the cardinality of each variable
    nO = len( obs_vals)
    nS = len( state_vals)
    nA = len( act_vals)
    
    ############################
    #  fix perception channel  #
    ############################

    # fix percpetion channel: ecnode observation into state    
    # The basic idea of this handcrafted perception is
    # assuming the perception system can almost optimally encode the objective 
    # weith minor perturbation.  
    psi_s1o = psi_hand( obs_vals, state_vals, lamb)

    # inference the observation based on the mental state
    # ψ(o|s) = ψ(s|o)p(o)/p(s), the inference is calculated using Baye's rule 
    psi_o1s = p_o * psi_s1o + 1e-20    # nOx1 * nOxnS, p(o) will braodcast  
    psi_o1s = (psi_o1s / np.sum( psi_o1s, axis=0, keepdims=True)).T
    p_s =  (p_o.T @ psi_s1o).T

    # now we have p(o), ψ(o|s), what we need p(a|o,s) == π(a|s)
    # init the p(a|o,s) as a uniform distribution 
    pi_a1s = np.ones( [ nS, nA]) 
    pi_a1s = pi_a1s / np.sum( pi_a1s, axis=-1, keepdims=True)

    # before that we need to find the bel util matrix
    # because the policy channel is only about mental states and actions
    # belU(s,a) = ∑_o ψ(o|s)U(o,a)
    bel_util = psi_o1s @ util_mat
    # use BA iteration to find the optimal policy channel
    results = BA_algs( bel_util, p_s, pi_a1s, 
                       beta2,
                       tol, max_iter)
    
    # unpack the optimized reults and 
    # calcute observational policy π(a|o)
    pi_a1s, p_a = results 
    # calculate the observation policy π(a|o) = ∑_s ψ(s|o)π(a|s)
    pi_a1o = np.sum( psi_s1o[ :, :, np.newaxis] * pi_a1s[ np.newaxis, :, :], axis=1)

    # store fix perception results for visualization
    fix_psi = dict()
    fix_psi['p(o)']     = p_o
    fix_psi['psi(s|o)'] = psi_s1o
    fix_psi['p(s)']     = p_s
    fix_psi['pi(a|s)']  = pi_a1s
    fix_psi['pi(a|o)']  = pi_a1o
    fix_psi['p(a)']     = p_a
    fix_psi['EU']       = np.sum( p_o * pi_a1o * util_mat)
    fix_psi['I(o;s)']   = I( p_o, psi_s1o)
    fix_psi['I(s;a)']   = I( p_s, pi_a1s)
    fix_psi['Jser']      = fix_psi['EU']  - 1/beta1 * fix_psi['I(o;s)'] \
                         - 1/beta2 * fix_psi['I(s;a)']

    ###########################
    #  RD perception channel  #
    ###########################

    # BA algorithm is a iterative algs, requiring initialization of some values
    # According to the channel rule p(o,s,a) = p(o)ψ(o|s)p(a|o,s)
    # once we know these three distributions, we can compute all other correspondence
    # among them, p(o) is given, ψ(o|s) I choose the handcrafed percpetion as init
    # what we only need is to asume p(a|o,s)
    
    # init the p(a|o,s) as a uniform distribution 
    p_a1os = np.ones( [ nO, nS, nA]) 
    p_a1os = p_a1os / np.sum( p_a1os, axis=-1, keepdims=True)

    # run the general blahut ariomoto algorithm to get the optimal
    # channel pairs 
    results = general_BA_algs( util_mat, p_o, psi_s1o, p_a1os, # utility & dist
                                beta1, beta2, beta3,           # price parameter 
                                tol, max_iter)                 # iteration hyperparameter
    # unpack the results 
    psi_s1o, pi_a1s, p_a1os, p_s, p_a = results
    # calculate the observation policy π(a|o) = ∑_s ψ(s|o)π(a|s)
    pi_a1o = np.sum( psi_s1o[ :, :, np.newaxis] * pi_a1s[ np.newaxis, :, :], axis=1)
    
    # store fix perception results for visualization
    RD_psi = dict()
    RD_psi['p(o)']     = p_o
    RD_psi['psi(s|o)'] = psi_s1o
    RD_psi['p(s)']     = p_s
    RD_psi['p(a|o,s)'] = p_a1os
    RD_psi['pi(a|s)']  = pi_a1s
    RD_psi['pi(a|o)']  = pi_a1o
    RD_psi['p(a)']     = p_a
    RD_psi['EU']       = np.sum( p_o * pi_a1o * util_mat)
    RD_psi['I(o;s)']   = I( p_o, psi_s1o)
    RD_psi['I(s;a)']   = I( p_s, pi_a1s)
    RD_psi['Jser']     = RD_psi['EU']  - 1/beta1 * RD_psi['I(o;s)'] \
                         - 1/beta2 * RD_psi['I(s;a)']

    ###################
    #  Visualization  #
    ###################

    plt.figure( figsize=( 21, 14))

    # Panel A: visualize handcrated perception channel ψ_λ(s|o)
    plt.subplot( 2, 3, 1)
    plt.imshow( fix_psi['psi(s|o)'].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
    plt.title('ψ_λ(s|o)')
    plt.xticks( np.arange(nO)+.5, obs_vars)
    plt.yticks( np.arange(nS)+.5, state_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    # note: share xlabel with panel D
    plt.ylabel( 'mental believed size')
    
    # Panel B: visualize the optimized policy π_λ(a|s)
    # with fix perception ψ_λ(s|o)
    plt.subplot( 2, 3, 2)
    plt.imshow( fix_psi['pi(a|o)'].T, cmap='Blues', origin='lower', vmin=0, vmax=1)
    plt.title('π_RD(a|o)')
    plt.xticks( np.arange(nO)+.5, obs_vars)
    plt.yticks( np.arange(nA)+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    # note: share xlabel with panel E
    plt.ylabel( 'action')

    # Panel C: viualize EU and Jser
    plt.subplot( 2, 3, 3)
    groups = [ 'rationality', 'resource-rationality']
    x = np.arange(len(groups))
    fix_group = [ fix_psi['EU'], fix_psi['Jser']]
    RD_group  = [  RD_psi['EU'],  RD_psi['Jser']]
    width = .35
    plt.bar( x-width/2, fix_group, width, label='fix ψ_λ(s|o)', color='salmon')
    plt.bar( x+width/2,  RD_group, width, label='learnt ψ_RD(s|o)', color='royalblue')
    plt.xticks( x, groups)
    plt.ylabel( 'values')
    plt.ylim([ 0, 4.2])
    plt.legend()

    # Panel D: visual the RD optimized percpetion channel ψ_RD(s|o)
    plt.subplot( 2, 3, 4)
    plt.imshow( RD_psi['psi(s|o)'].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
    plt.title('ψ_RD(s|o)')
    plt.xticks( np.arange(nO)+.5, obs_vars)
    plt.yticks( np.arange(nS)+.5, state_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    plt.xlabel( 'observed animal size')
    plt.ylabel( 'mental believed size')
    
    # Panel E: visualize the optimized policy π_RD(a|s)
    # with adaptive perception ψ_RD(s|o)
    plt.subplot( 2, 3, 5)
    plt.imshow( RD_psi['pi(a|o)'].T, cmap='Blues', origin='lower', vmin=0, vmax=1)
    plt.grid(alpha=.5)
    plt.title('π_RD(a|o)')
    plt.xticks( np.arange(nO)+.5, obs_vars)
    plt.yticks( np.arange(nA)+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    plt.xlabel( 'observed animal size')
    plt.ylabel( 'action')

    # Panel F: viualize mutual information 
    plt.subplot( 2, 3, 6)
    groups = [ 'I(o;s)', 'I(s;a)']
    x = np.arange(len(groups))
    fix_group = [ fix_psi['I(o;s)'], fix_psi['I(s;a)']]
    RD_group  = [  RD_psi['I(o;s)'],  RD_psi['I(s;a)']]
    width = .35
    plt.bar( x-width/2, fix_group, width, label='fix ψ_λ(s|o)', color='salmon')
    plt.bar( x+width/2,  RD_group, width, label='learnt ψ_RD(s|o)', color='royalblue')
    plt.xticks( x, groups)
    plt.ylabel( 'values')
    plt.ylim([ 0, 4.2])
    plt.legend()
    
    fig_name = f'{path}/figures/cascade_channel-lambda={lamb}-beta1={beta1}-beta2={beta2}.png'
    plt.savefig( fig_name)

def get_parl_results( beta1, beta2, beta3, 
                        is_uniform, tol, max_iter):
    # load env and perception 
    obs_vals, _, p_o, act_vals, _, util_mat = setup_medical_example(is_uniform)

    # init the model, as said in original paper in page 14,
    # we use same notation for model and state, becuase this benefit comparision. 
    state_vals  = np.arange( 1, 4)

    # obtain the cardinality of each variable
    nO = len( obs_vals)
    nS = len( state_vals)
    nA = len( act_vals)
    
    # BA algorithm is a iterative algs, requiring initialization of some values
    # According to the channel rule p(o,s,a) = p(o)ψ(o|s)p(a|o,s)
    # once we know these three distributions, we can compute all other correspondence
    # among them, p(o) is given, 
    # what we only need is to asume ψ(s|o) and p(a|o,s)

    # init the ψ(s|o) as an uniform distribution
    # this is not how they used in their document,
    # but I cannot understand their code
    psi_s1o = np.random.rand( nO, nS)
    psi_s1o = psi_s1o / np.sum( psi_s1o, axis=-1, keepdims=True)
    
    # init the p(a|o,s) as an uniform distribution 
    p_a1os = np.ones( [ nO, nS, nA]) 
    p_a1os = p_a1os / np.sum( p_a1os, axis=-1, keepdims=True)

    # run the general blahut ariomoto algorithm to get the optimal
    # channel pairs 
    results = general_BA_algs( util_mat, p_o, psi_s1o, p_a1os, # utility & dist
                                beta1, beta2, beta3,           # price parameter 
                                tol, max_iter)                 # iteration hyperparameter
    # unpack the results 
    psi_s1o, pi_a1s, p_a1os, p_s, p_a = results
    # calculate the observation policy π(a|o) = ∑_s ψ(s|o)π(a|s)
    pi_a1o = np.sum( psi_s1o[ :, :, np.newaxis] * p_a1os, axis=1)
    
    # store fix perception results for visualization
    p1 = dict()
    p1['p(o)']     = p_o
    p1['psi(s|o)'] = psi_s1o
    p1['p(s)']     = p_s
    p1['p(a|o,s)'] = p_a1os
    p1['pi(a|s)']  = pi_a1s
    p1['pi(a|o)']  = pi_a1o
    p1['p(a)']     = p_a
    # p1['EU']       = np.sum( p_o * pi_a1o * util_mat)
    # p1['I(o;s)']   = I( p_o, psi_s1o)
    # p1['I(o;a|s)'] = I( p_s, pi_a1s)
    # p1['Jser']     = p1['EU']  - 1/beta1 * p1['I(o;s)'] \
    #                      - 1/beta3 * p1['I(s;a)']

    return p1


def illustrate_parallel_channel( beta1, beta2, beta3):
    '''REPLICATE THE FIG10

    This function replicate fig to illustrate
    how parallel channel may work, the env is 
    the prey pradator environment.
    '''
    # set hyperparameters
    # lamb : precision of the hand-crafted perceptual model
    # beta1: price for I(O;S)
    # beta2: price for I(S;A)
    # beta3: price for I(A;S,O)
    tol   = 1e-3  # tolerance for convergence 
    max_iter = 10000 # maximum number of BA iterations

    # load vars for plot
    _, obs_vars, _, _, act_vars, _ = setup_medical_example()
    state_vals  = np.arange( 1, 4)
    state_vars  = [ f'm={s}' for s in state_vals]

    # obtain the cardinality of each variable
    nO = len( obs_vars)
    nS = len( state_vars)
    nA = len( act_vars)

    #################################
    #  RD begin with uniform prior  #
    #################################

    is_uniform = True 
    p1 = get_parl_results( beta1, beta2, beta3,
                           is_uniform, tol, max_iter)

    ################################
    #  RD begin with biased prior  #
    ################################

    is_uniform = False 
    p2 = get_parl_results( beta1, beta2, beta3,
                            is_uniform, tol, max_iter)
    
    ###################
    #  Visualization  #
    ###################

    plt.figure( figsize=( 21, 14))

    # Panel A: visualize higher-level model selector with uniform prior
    plt.subplot( 2, 3, 1)
    plt.imshow( p1['psi(s|o)'].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
    plt.title('ψ_1(s|o)')
    plt.xticks( np.arange(nO)+.5, obs_vars)
    plt.yticks( np.arange(nS)+.5, state_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    # note: share xlabel with panel D
    plt.ylabel( 'first diagnosis')
    
    # Panel B: visualize the model policy with uniform pior 
    plt.subplot( 2, 3, 2)
    plt.imshow( p1['pi(a|s)'].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
    plt.title('π_1(a|s)')
    plt.xticks( np.arange(nS)+.5, state_vars)
    plt.yticks( np.arange(nA)+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    # note: share xlabel with panel E
    plt.ylabel( 'treatment')

    # Panel C: visualize the obs policy with uniform pior 
    plt.subplot( 2, 3, 3)
    plt.imshow( p1['pi(a|o)'].T, cmap='Blues', origin='lower', vmin=0, vmax=1)
    plt.title('π_1(a|o)')
    plt.xticks( np.arange(nO)+.5, obs_vars)
    plt.yticks( np.arange(nA)+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    # note: share xlabel with panel E
    plt.ylabel( 'treatment')

    # Panel C: visualize higher-level model selector with biased prior
    plt.subplot( 2, 3, 4)
    plt.imshow( p2['psi(s|o)'].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
    plt.title('ψ_2(s|o)')
    plt.xticks( np.arange(nO)+.5, obs_vars)
    plt.yticks( np.arange(nS)+.5, state_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    plt.xlabel( 'disease type')
    plt.ylabel( 'first diagnosis')
    
    # Panel D: visualize the model policy with biased prior 
    plt.subplot( 2, 3, 5)
    plt.imshow( p2['pi(a|s)'].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
    plt.title('π_2(a|s)')
    plt.xticks( np.arange(nS)+.5, state_vars)
    plt.yticks( np.arange(nA)+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    plt.xlabel( 'first diagnosis')
    plt.ylabel( 'treatment')

    # Panel E: visualize the obs policy with biased prior
    plt.subplot( 2, 3, 6)
    plt.imshow( p2['pi(a|o)'].T, cmap='Blues', origin='lower', vmin=0, vmax=1)
    plt.title('π_2(a|o)')
    plt.xticks( np.arange(nO)+.5, obs_vars)
    plt.yticks( np.arange(nA)+.5, act_vars)
    plt.grid(linewidth=1.2)
    plt.colorbar()
    plt.xlabel( 'disease type')
    plt.ylabel( 'treatment')

    fig_name = f'{path}/figures/parallel_channel-beta1={beta1}-beta3={beta3}.png'
    plt.savefig( fig_name)

if __name__ == '__main__':

    #Predator prey
    show_predator_prey_util()   

    lambdas = [ 1.65, 1.65, .4]
    beta1s  = [  8, 8, 1 ]
    beta2s  = [ 10, 1, 1 ]
    beta3s  = [  0, 0, 0 ]
    for lamb, beta1, beta2, beta3 in zip( lambdas, beta1s, beta2s, beta3s):
        illustrate_cascade_channel( lamb, beta1, beta2, beta3)  

    # Medical
    show_medical_util()

    beta1  = 2.
    beta2  = np.inf
    beta3  = .9
    illustrate_parallel_channel( beta1, beta2, beta3)

    
    



        


            





    
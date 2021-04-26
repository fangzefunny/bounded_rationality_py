'''
This code is to understand the tradeoff parameter
as rejection rate in sampling method. 

I believed this is important becuase many believe human finite
sampling is the real bounded rationality.

Original repo: 
https://github.com/tgenewein/BoundedRationalityAbstractionAndHierarchicalDecisionMaking
'''

import os 
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.special import logsumexp # for partition function

# define the saving path
path = os.path.dirname(os.path.abspath(__file__))

'''
SEC1: Environment
'''

## Taxonomy environment
def setup_taxonomy_example():

    #### observations
    obs_vars = ['Laptop', 'Monitor', 'Gamepad',
                'Coffee machine', 'Vacuum cleaner', 'Electric toothbrush',
                'Grapes', 'Strawberries', 'Limes',
                'Pancake mix', 'Baking soda', 'Bakers yeast', 'Muffin cups']
    nO       = len(obs_vars)
    obs_vals = np.arange( 1, nO+1)
    p_o      = np.ones( [ nO, 1]) / nO

    #### action
    act_vars = ["Laptop sleeve","Monitor cable","Video game",
                "Coffee","Vacuum cleaner bags","Brush heads",
                "Cheese","Cream","Cane sugar",
                "Maple syrup","Vinegar","Flour","Chocolate chips",
                "COMPUTERS","APPLIANCES","FRUIT","BAKING","Electronics","Food"]
    nA       = len(act_vars)
    act_vals = np.arange( 1, nA+1)

    # utility function
    def utility_fn( obs, act):

        # all utilities 
        util_correct_obs = 3
        util_correct_cat = 2.2
        util_correct_supcat = 1.6

        # correct item
        if (act<14) and (act==obs):
            return util_correct_obs

        # flour is also fine for s=muffil cups
        if (obs==13) and (act==12):
            return util_correct_obs

        # packe mix both fruit and baking
        if (obs==10) and (act==16):
            return util_correct_cat

        # extra if-clause is required for muffin cups
        if (obs==13) and (act==17):
            return util_correct_cat

        # correct category
        if (act<18):
            cat = np.ceil( obs/3)
            if (act-13)==cat:
                return util_correct_cat

        # correct supercategory
        supcat = np.ceil( obs/6)
        if (act-17) == supcat:
            return util_correct_supcat

        # separate case for act==19
        if (act==19) and (obs==13):
            return util_correct_supcat

        return 0


    # make util table 
    util_mat = make_util_mat( utility_fn, obs_vals, act_vals)

    return obs_vals, obs_vars, p_o, act_vals, act_vars, util_mat

# from utility function to utility matrix 
def make_util_mat( utility_fn, obs_vals, act_vals):
    '''MAKE UTILITY MATRIX 
    '''
    util_mat = np.zeros( [len(obs_vals), len(act_vals)])
    for o_idx, o in enumerate(obs_vals):
        for a_idx, a in enumerate(act_vals):
            util_mat[ o_idx, a_idx] = utility_fn( o, a)
    return util_mat

'''
SEC2: sampling 
'''

def sampling_channel( util, beta, prop_dist, 
                      n_samples, max_iter):
    '''
    Goal: get a sample from target distribution f(y)

    Method: propose a sample y ~ q(y)
            and calcuate the acceptance rate: f(y)/Mq(y)

    In the free energy problem
    the target distribution: p(y|xi) ∝ f(y) = p_y(y)exp(βU(y,xi))  
    scale parameter M: M = exp(βUmax)
    Thus the acceptance rate: p_y(y)exp(βU(y,xi)) / exp(βUmax)p_y(y)
                              exp(β[U(y,xi)-Umax]) 
                              exp(-β Distort)
    '''
    # init
    y_samples = []
    acc_cnts  = 0

    for t in max_iter:
        
        # propose a sample, and decide if we accept it
        # according to acceptance rate
        u        = np.random.rand()
        prop_y   = np.random.choice(len(prop_dist), p=prop_dist)
        acc_rate = np.exp( beta * (util[prop_y] - util.max()))
        if u < acc_rate:
            acc_cnts += 1
            # turn the value to variables
            y_samples.append( prop_y)
        
        # check whether to stop sampling
        if acc_cnts >= n_samples:
            break
        
    if t == max_iter:
        print(f'''
               Maximum number of iter {max_iter} reached,
               the number of samples is potentially lower than expected
               ''')
    
    # store the sample and compute the total accpetance rate
    y_samples = np.array(y_samples)
    tot_acc_rate = acc_cnts / t

    return y_samples, tot_acc_rate

def sampling_marginal( y_samples, prop_cnts_mat, forgetting=True):

    # get the y cardinality 
    nY = len(prop_cnts_mat)

    # incremental or fully forgetting
    if forgetting:
        prop_cnts_mat = np.ones( [ nY,])/nY

    bins = np.arange( -.5, nY+.5)
    freq, _ = np.histogram( y_samples, bins)
    prop_cnts_mat += freq

    return prop_cnts_mat

##
def BA_sampling( util_mat, p_o, p_a,
                 beta, 
                 num_p_a, num_p_a1o,
                 burin_rate=.7, max_iter=200, forgetting=True):
    '''Rate-distortion rejection sampling
    
    Pesudocode:

        1. Draw a number of samples (a batch) from 
            π(a|s)= 1/Z(s) p(a)exp[βU(s,a)]
            using the rejection sampling scheme.

        2. Update p(a) with the accepted samples obtained
           in step 1.
           - either simply increase the counters for each
             accepted a and renormalize (no-forgetting)
           - reset the counters for a and use only the last 
             batch of accepted samples to estimate p(a) (fully forgetting)
           - use a parameteric model for p_θ(a) and use gradient-based 
             update 

        3. Reapeat util convergence

    '''
    # get the cardinality,
    nA = p_a.shape[0]
    nO = p_o.shape[0]

    # initialize the sampling iteration
    p_a1o_cnts     = np.ones([ nO, nA]) / nA
    p_a_cnts       = p_a.reshape[-1]
    start_sampling = False
    
    
    for i in range(num_p_a):
        
        # holds the samples from p(a|o) in the inner loop
        a_samples = np.zeros([num_p_a1o,])
        if start_sampling:
            p_a1o_cnts = np.ones([ nO, nA]) / nA
            avg_acc_rate   = 0 

        for j in range(num_p_a1o):

            # get an observation
            obs = np.random.choice(len(nO), p=p_o)

            # sampling to estimate p(a|ot)
            # based on U(ot, ad)
            util = util_mat[ obs, :]
            a1o_samples, acc_rate = sampling_channel( util, beta, p_a, 
                                                    1, max_iter)
            a_samples[j] = a1o_samples[0]

            # record the samples
            if start_sampling:
                p_a1o_cnts[ obs, a_samples] += 1
                avg_acc_rate += acc_rate

        # naive burn-in check
        if (i > num_p_a * burin_rate):
            start_sampling = True

        # update
        p_a_cnts = sampling_marginal( a_samples, p_a_cnts, forgetting)  

        # normlaize to esimate probaility distribution
        # note that due to our initlaization, we do not need 
        # an epsilon here
        p_a1o = p_a1o_cnts / np.sum( p_a1o_cnts, axis=1, keepdims=True)
        p_a = p_a_cnts / np.sum( p_a_cnts)

    # get the average acceptance rate after sampling
    avg_acc_rate /= ( num_p_a * (1-burin_rate) * num_p_a1o)

    return p_a1o, p_a, avg_acc_rate


'''
SEC3: Visualization
'''

def illustrate_sampling_BA( ):

    # hyperameters
    max_iter  = 10000
    beta      = 1.2
    num_p_a1o = 700
    num_p_a   = 500

    # load env and perception 
    obs_vals, obs_vars, p_o, act_vals, act_vars, util_mat = setup_taxonomy_example()
    nA = len(act_vals)
    nO = len(obs_vals)

    # initialize p(a) as uniform
    p_a = np.ones([ nA, 1]) / nA 

    # find the best channel π(a|o)
    # use the sampling method
    p_a1o, p_a, avg_acc_rate = BA_sampling( util_mat, p_o, p_a, 
                                            beta, 
                                            num_p_a, num_p_a1o, 
                                            forgetting=False)


def show_taxonomy_util() :
    _, obs_vars, _, _, act_vars, util_mat = setup_taxonomy_example()
    
    plt.figure( figsize=( 21, 7))
    plt.subplot( 1,2,1)
    plt.imshow( util_mat.T, cmap='Blues', origin='lower')
    plt.title('Utility of taxonomy')
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

def sim( ):

    # init for the iteration
    nA = 3 
    T = 10
    V = np.ones( [nA,]) / nA
    R = np.array( [ 0, 1, 0])
    p_a = np.ones( [nA,]) / nA
    beta_sweeps = np.linspace( 0.1, 3000, 1000)
    lr_v = .1
    n_target = 2.5
    history = np.zeros([T,])

    for t in range(T):
        # update V 
        V += lr_v * (R - V)

    
        for beta in beta_sweeps:

            est_n = 1 / np.sum(p_a * np.exp( beta * (V - V.max())))
            
            if (est_n - n_target) > 0:
                history[t] = beta 
                break 
    
    plt.plot( history)
    plt.show()

if __name__  == '__main__':

    sim()






    
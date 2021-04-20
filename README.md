# Multi channels Rate-distortion theory

## Introduction

This code is created to replicate the results of a publication: 
https://www.frontiersin.org/articles/10.3389/frobt.2015.00027/full

The paper was published with its own code repo written in Julia: 
https://github.com/tgenewein/BoundedRationalityAbstractionAndHierarchicalDecisionMaking

I do not speak Julia and cannot run those codes, so I translate them into Python.
During the translation, I change the notation a bit to make sure the code is consistent with
my own projects. The map between published nontation and my notation are:

* World states: W (published) --> O (mine)
* Perceived representations: X (paper) or O (code) --> S (mine)
* Actions: A (publised) --> A (mine)

## Prerequisite packages

What you need are:

* numpy 
* matplotlib.pyplot
* scipy.special

I believed these are all basic packages and we should be able to run the code with any version of these packages. 

## How to run 

This is a python 3.8 script and ideally, it can be run with any python3 interpreter using the command:

    python main.py 
    
You can tune the hyperparameter λ, β1, β2, β3 as you want while running the code. (start from line 1004)

The execution takes less than 1 min, and you will find in the current folder the following figures:

* Cascade channel example: predator and prey task
    * utility matrix and optimal policy 
    * learned rate-distortion based state encoder ψ(s|o) and policy π(a|s) (varies with input hyperparameters)
* Parallel channel example: medical task
    * utility matrix and optimal policy
    * learned rate-distortion based model selector ψ(s|o), model policy π(a|s), and policy π(a|o,s) (varies with input hyperparameters)

## Do I replicate the publication?

No. I do not reproduce the publised results.

Probably, I made some mistakes when coding, and I will proofread my codes later. 

I also had difficulty with understanding the following original codes:

   * The initialization of the distribution: I cannot understand how the author sparsely initialized the state encoder and model selector ψ(s|o) (original repo, "ThreeVariableBlahutArimoto.jl" line 200-221). What I used is random initialization. As mentioned by the author in the paper (page 21, second paragraph):
   "We found sometimes that the solutions can be sensitive to the initialization. This hints at the problem being non-convex..." My failure to reproduce the results might be caused by different initialization implementation
   * The update equations in general Blahut-Arimoto algorithm: As far as I know, when implementing the alternative optimization method, we usually follow: 
        *  xt = argmin f(x, yt-1)
        *  yt = argmin f(xt, y)
      
     However, in the orginal code (original repo, "ThreeVariableBlahutArimoto.jl" line 323-342), they mentioned of using the old distribution, in another word, they followed a different alternative update schema:
        *  xt = argmin f(x, yt-1)
        *  yt = argmin f(xt-1, y)
     
     I know the second schema is often used in neural network trainning, but I prefer the first method. Will this different implement the cause of my reproduction failures?
     
I will analyze the results and back to this section later.


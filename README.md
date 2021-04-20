# Multi channels Rate-distortion theory

## Introduction

This code is created to replicate the results of a publication: 
https://www.frontiersin.org/articles/10.3389/frobt.2015.00027/full

The paper was published with its own code repo written in Julia: 
https://github.com/tgenewein/BoundedRationalityAbstractionAndHierarchicalDecisionMaking

I do not speak Julia and cannot run those codes, so I translate them into Python.

## How to run 

This is a python 3.8 script and ideally, it can be run with any python3 interpreter using the command:

    python main.py 
    
You can tune the hyperparameter λ, β1, β2, β3 as you want while running the code. (locate to line 999)

The execution takes less than 1 min, and you will find in the current folder the following figures:

* Cascade channel example: predator and prey task
    * utility matrix and optimal policy 
    * learned rate-distortion based state encoder ψ(s|o) and policy π(a|s) (varies with input hyperparameters)
* Parallel channel example: medical task
    * utility matrix and optimal policy
    * learned rate-distortion based model selector ψ(s|o), model policy π(a|s), and policy π(a|o,s) (varies with input hyperparameters)

## Do I replicate the publication?

No. I found some issues with the original codes. And probably, I also made some mistakes when coding. 
I will be back and update this section after I thoroughly analyze both codes. 

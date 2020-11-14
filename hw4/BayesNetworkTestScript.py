#!/usr/bin/env python3

from BayesianNetworks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
base_dir = os.path.dirname(__file__)
'''
#############################
## Example Tests from Bishop Pattern recognition textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors 
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, 'fuel', '1')
evidenceUpdateNet(carNet, ['fuel', 'battery'], ['1', '0'])

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.

marginalizeNetworkVariables(carNet, 'battery') ## this returns back a list
marginalizeNetworkVariables(carNet, 'fuel') ## this returns back a list
marginalizeNetworkVariables(carNet, ['battery', 'fuel'])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
'''
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv(os.path.join(base_dir,'RiskFactorsData.csv'))

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit    = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up     = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

## you need to create more factor tables

risk_net = [income, smoke, long_sit, stay_up, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2,long_sit=1)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise','long_sit'})
obsVars  = ['smoke', 'exercise','long_sit']
obsVals  = [1, 2, 1]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)


### Please write your own test scrip similar to  the previous example 
###########################################################################
#HW4 test scrripts start from here
###########################################################################

# create all factors
_income = readFactorTablefromData(riskFactorNet, ['income'])
_exercise = readFactorTablefromData(riskFactorNet, ['exercise','income'])
_long_sit = readFactorTablefromData(riskFactorNet, ['long_sit','income'])
_stay_up = readFactorTablefromData(riskFactorNet, ['stay_up','income'])
_smoke = readFactorTablefromData(riskFactorNet, ['smoke','income'])
_bmi = readFactorTablefromData(riskFactorNet, ['bmi','exercise','income','long_sit'])
_bp = readFactorTablefromData(riskFactorNet, ['bp','exercise','long_sit','income','stay_up','smoke'])
_cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol','exercise','stay_up','income','smoke'])
_diabetes = readFactorTablefromData(riskFactorNet, ['diabetes','bmi'])
_stroke = readFactorTablefromData(riskFactorNet, ['stroke','bmi','bp','cholesterol'])
_attack = readFactorTablefromData(riskFactorNet, ['attack','bmi','bp','cholesterol'])
_angina = readFactorTablefromData(riskFactorNet, ['angina','bmi','bp','cholesterol'])

# create factor tables
full_net=[_income,_exercise,_long_sit,_stay_up,_smoke,_bmi,_bp,_cholesterol,_diabetes,_stroke,_attack,_angina]

'''
# question 1
print('-----------------question 1-------------------')

# calculate size of network 
net_size=0
for i in full_net:
    net_size+=len(i)
print('size of the network:',net_size)

# calculate size of full joint distribution
joint_dis=inference(full_net,[],[],[])
print('total number of probabilities needed to store the full joint distribution:',len(joint_dis))

print('-----------------question 1-------------------')
'''
print('-----------------question 2a-------------------')

diabetes_marg=list(set(factors)-{'diabetes','smoke','exercise','long_sit','stay_up'})
stroke_marg=list(set(factors)-{'stroke','smoke','exercise','long_sit','stay_up'})
attack_marg=list(set(factors)-{'attack','smoke','exercise','long_sit','stay_up'})
angina_marg=list(set(factors)-{'angina','smoke','exercise','long_sit','stay_up'})

habitVars=['smoke','exercise','long_sit','stay_up']
badVals=[1,2,1,1]
goodVals=[2,1,2,2]
'''
# calculate probs with bad habits
diabetes_bad=inference(full_net,diabetes_marg,habitVars,badVals)
print('probability of diabetes if I have bad habits:\n',diabetes_bad)
stroke_bad=inference(full_net,stroke_marg,habitVars,badVals)
print('probability of stroke if I have bad habits:\n',stroke_bad)
attack_bad=inference(full_net,attack_marg,habitVars,badVals)
print('probability of attack if I have bad habits:\n',attack_bad)
angina_bad=inference(full_net,angina_marg,habitVars,badVals)
print('probability of angina if I have bad habits:\n',angina_bad)

# calculate probs with good habits
diabetes_good=inference(full_net,diabetes_marg,habitVars,goodVals)
print('probability of diabetes if I have good habits:\n',diabetes_good)
stroke_good=inference(full_net,stroke_marg,habitVars,goodVals)
print('probability of stroke if I have good habits:\n',stroke_good)
attack_good=inference(full_net,attack_marg,habitVars,goodVals)
print('probability of attack if I have good habits:\n',attack_good)
angina_good=inference(full_net,angina_marg,habitVars,goodVals)
print('probability of angina if I have good habits:\n',angina_good)

print('-----------------question 2a-------------------')

print('-----------------question 2b-------------------')
'''
diabetes_marg2=list(set(factors)-{'diabetes','bp','cholesterol','bmi'})
stroke_marg2=list(set(factors)-{'stroke','bp','cholesterol','bmi'})
attack_marg2=list(set(factors)-{'attack','bp','cholesterol','bmi'})
angina_marg2=list(set(factors)-{'angina','bp','cholesterol','bmi'})

healthVars=['bp','cholesterol','bmi']
badVals2=[1,1,3]
goodVals2=[3,2,2]
'''
# calculate probs with poor health
diabetes_bad2=inference(full_net,diabetes_marg2,healthVars,badVals2)
print('probability of diabetes if I have poor health:\n',diabetes_bad2)
stroke_bad2=inference(full_net,stroke_marg2,healthVars,badVals2)
print('probability of stroke if I have poor health:\n',stroke_bad2)
attack_bad2=inference(full_net,attack_marg2,healthVars,badVals2)
print('probability of attack if I have poor health:\n',attack_bad2)
angina_bad2=inference(full_net,angina_marg2,healthVars,badVals2)
print('probability of angina if I have poor health:\n',angina_bad2)

# calculate probs with good health
diabetes_good2=inference(full_net,diabetes_marg2,healthVars,goodVals2)
print('probability of diabetes if I have good health:\n',diabetes_good2)
stroke_good2=inference(full_net,stroke_marg2,healthVars,goodVals2)
print('probability of stroke if I have good health:\n',stroke_good2)
attack_good2=inference(full_net,attack_marg2,healthVars,goodVals2)
print('probability of attack if I have good health:\n',attack_good2)
angina_good2=inference(full_net,angina_marg2,healthVars,goodVals2)
print('probability of angina if I have good health:\n',angina_good2)

print('-----------------question 2b-------------------')

print('-----------------question 3-------------------')

outcomes=['diabetes','stroke','attack','angina']
Vars=['income']
x=list(range(1,9))
prob_dict={}

# calculate and plot the probs
for out in outcomes:
    marg=list(set(factors)-{out,'income'})
    prob_dict[out]=[]
    for i in range(1,9):
        probs=inference(full_net,marg,Vars,[i])
        prob_dict[out].append(list(probs[probs[out]==1]['probs'])[0])
        print('prob of '+out+' when income is '+str(i)+': ',prob_dict[out][-1])
    plt.plot(x,prob_dict[out],label=out)
plt.title('relationship between probability of diseases and income level')
plt.legend()
plt.xlabel('income level')
plt.ylabel('probability of disease')
plt.grid()
plt.savefig('disease_and_income.png')
plt.show()

print('-----------------question 3-------------------')
'''
print('-----------------question 4-------------------')

_diabetes2 = readFactorTablefromData(riskFactorNet, ['diabetes','bmi','smoke','exercise'])
_stroke2 = readFactorTablefromData(riskFactorNet, ['stroke','bmi','bp','cholesterol','smoke','exercise'])
_attack2 = readFactorTablefromData(riskFactorNet, ['attack','bmi','bp','cholesterol','smoke','exercise'])
_angina2 = readFactorTablefromData(riskFactorNet, ['angina','bmi','bp','cholesterol','smoke','exercise'])

full_net2=[_income,_exercise,_long_sit,_stay_up,_smoke,_bmi,_bp,_cholesterol,_diabetes2,_stroke2,_attack2,_angina2]

# calculate probs with bad habits
diabetes_bad3=inference(full_net2,diabetes_marg,habitVars,badVals)
print('probability of diabetes if I have bad habits:\n',diabetes_bad3)
stroke_bad3=inference(full_net2,stroke_marg,habitVars,badVals)
print('probability of stroke if I have bad habits:\n',stroke_bad3)
attack_bad3=inference(full_net2,attack_marg,habitVars,badVals)
print('probability of attack if I have bad habits:\n',attack_bad3)
angina_bad3=inference(full_net2,angina_marg,habitVars,badVals)
print('probability of angina if I have bad habits:\n',angina_bad3)

# calculate probs with good habits
diabetes_good3=inference(full_net2,diabetes_marg,habitVars,goodVals)
print('probability of diabetes if I have good habits:\n',diabetes_good3)
stroke_good3=inference(full_net2,stroke_marg,habitVars,goodVals)
print('probability of stroke if I have good habits:\n',stroke_good3)
attack_good3=inference(full_net2,attack_marg,habitVars,goodVals)
print('probability of attack if I have good habits:\n',attack_good3)
angina_good3=inference(full_net2,angina_marg,habitVars,goodVals)
print('probability of angina if I have good habits:\n',angina_good3)

# calculate probs with poor health
diabetes_bad4=inference(full_net2,diabetes_marg2,healthVars,badVals2)
print('probability of diabetes if I have poor health:\n',diabetes_bad4)
stroke_bad4=inference(full_net2,stroke_marg2,healthVars,badVals2)
print('probability of stroke if I have poor health:\n',stroke_bad4)
attack_bad4=inference(full_net2,attack_marg2,healthVars,badVals2)
print('probability of attack if I have poor health:\n',attack_bad4)
angina_bad4=inference(full_net2,angina_marg2,healthVars,badVals2)
print('probability of angina if I have poor health:\n',angina_bad4)

# calculate probs with good health
diabetes_good4=inference(full_net2,diabetes_marg2,healthVars,goodVals2)
print('probability of diabetes if I have good health:\n',diabetes_good4)
stroke_good4=inference(full_net2,stroke_marg2,healthVars,goodVals2)
print('probability of stroke if I have good health:\n',stroke_good4)
attack_good4=inference(full_net2,attack_marg2,healthVars,goodVals2)
print('probability of attack if I have good health:\n',attack_good4)
angina_good4=inference(full_net2,angina_marg2,healthVars,goodVals2)
print('probability of angina if I have good health:\n',angina_good4)

print('-----------------question 4-------------------')
'''
print('-----------------question 5-------------------')

_stroke3 = readFactorTablefromData(riskFactorNet, ['stroke','bmi','bp','cholesterol','smoke','exercise','diabetes'])

full_net3=[_income,_exercise,_long_sit,_stay_up,_smoke,_bmi,_bp,_cholesterol,_diabetes2,_stroke3,_attack2,_angina2]

marg=list(set(factors)-{'diabetes','stroke'})
# second network
d1_2=inference(full_net2,marg,['diabetes'],[1])
d3_2=inference(full_net2,marg,['diabetes'],[3])
print('For the second Bayesian network:\n')
print(d1_2)
print('P(stroke=1|diabetes=1) = ',list(d1_2[d1_2['stroke']==1]['probs'])[0])
print(d3_2)
print('P(stroke=1|diabetes=3) = ',list(d3_2[d3_2['stroke']==1]['probs'])[0])

# third network
d1_3=inference(full_net3,marg,['diabetes'],[1])
d3_3=inference(full_net3,marg,['diabetes'],[3])
print('\n')
print('For the third Bayesian network:\n')
print(d1_3)
print('P(stroke=1|diabetes=1) = ',list(d1_3[d1_3['stroke']==1]['probs'])[0])
print(d3_3)
print('P(stroke=1|diabetes=3) = ',list(d3_3[d3_3['stroke']==1]['probs'])[0])

print('-----------------question 5-------------------')
'''
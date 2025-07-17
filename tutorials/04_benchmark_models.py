from alpaca.models.model_library import *
import sympy as sp

# Models from 1610.07593 to check we obtain the same results

charge=sp.symbols('X')

R1=HeavyFermion('3','1',-1/3,charge)
R2=HeavyFermion('3','1',2/3,charge)
R3=HeavyFermion('3','2',1/6,charge)
R4=HeavyFermion('3','2',-5/6,charge)
R5=HeavyFermion('3','2',7/6,charge)
R6=HeavyFermion('3','3',-1/3,charge)
R7=HeavyFermion('3','3',2/3,charge)
R8=HeavyFermion('3','3',-4/3,charge)
R9=HeavyFermion('6_bar','1',-1/3,charge)
R10=HeavyFermion('6_bar','1',2/3,charge)
R11=HeavyFermion('6_bar','2',1/6,charge)
R12=HeavyFermion('8','1',-1,charge)
R13=HeavyFermion('8','2',-1/2,charge)
R14=HeavyFermion('15','1',-1/3,charge)
R15=HeavyFermion('15','1',2/3,charge)

list_of_fermions=[R1,R2,R3,R4,R5,R6,R7, R8, R9, R10, R11, R12, R13, R14, R15]


list_of_KSVZ_models = []
i=1
for HeavyFermion in list_of_fermions:
    name = 'R' + str(i)
    list_of_KSVZ_models.append(KSVZ_model(name, [HeavyFermion]))
    i+=1
for models in list_of_KSVZ_models:
   print(models.model_name,'E/N= ', models.E_over_N())


# Models from 1705.05370 to check we obtain the same results
Chargeu=sp.symbols('Xu')
Charged=sp.symbols('Xd')


DFSZ1=PQChargedModel('DFSZ-I', {'eR': charge, 'uR': charge, 'dR': charge})
DFSZ2=PQChargedModel('DFSZ-II', {'eR': -charge, 'uR': charge, 'dR': charge})
DFSZ3a=PQChargedModel('DFSZ-IIIa', {'eR': -(2*Chargeu+Charged), 'uR': Chargeu, 'dR': Charged})
DFSZ3b=PQChargedModel('DFSZ-IIIb', {'eR': Charged,  'uR': Chargeu, 'dR': Charged})
DFSZ3c=PQChargedModel('DFSZ-IIIc', {'eR': Chargeu+2*Charged, 'uR': Chargeu, 'dR': Charged})
DFSZ3d=PQChargedModel('DFSZ-IIId', {'eR': -Chargeu, 'uR': Chargeu, 'dR': Charged})

list_of_DFSZ_models=[DFSZ1, DFSZ2, DFSZ3a, DFSZ3b, DFSZ3c, DFSZ3d]
for models in list_of_DFSZ_models:
   print(models.model_name,'E/N= ', models.E_over_N())





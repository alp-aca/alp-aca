from alpaca.models.model_library import *
import sympy as sp

# Models from 1610.07593 to check we obtain the same results

charge=sp.symbols('X')

R1=fermion('3','1',-1/3,charge)
R2=fermion('3','1',2/3,charge)
R3=fermion('3','2',1/6,charge)
R4=fermion('3','2',-5/6,charge)
R5=fermion('3','2',7/6,charge)
R6=fermion('3','3',-1/3,charge)
R7=fermion('3','3',2/3,charge)
R8=fermion('3','3',-4/3,charge)
R9=fermion('6_bar','1',-1/3,charge)
R10=fermion('6_bar','1',2/3,charge)
R11=fermion('6_bar','2',1/6,charge)
R12=fermion('8','1',-1,charge)
R13=fermion('8','2',-1/2,charge)
R14=fermion('15','1',-1/3,charge)
R15=fermion('15','1',2/3,charge)

list_of_fermions=[R1,R2,R3,R4,R5,R6,R7, R8, R9, R10, R11, R12, R13, R14, R15]


list_of_KSVZ_models = []
i=1
for fermion in list_of_fermions:
    name = 'R' + str(i)
    list_of_KSVZ_models.append(KSVZ_model(name, [fermion]))
    i+=1
for models in list_of_KSVZ_models:
   print(models.model_name,'E/N= ',sp.Rational(models.couplings['cgamma']/models.couplings['cg']).limit_denominator())


# Models from 1705.05370 to check we obtain the same results
Chargeu=sp.symbols('Xu')
Charged=sp.symbols('Xd')


DFSZ1=model('DFSZ-I', {'lL': 0, 'eR': charge, 'qL': 0, 'uR': charge, 'dR': charge})
DFSZ2=model('DFSZ-II', {'lL': 0, 'eR': -charge, 'qL': 0, 'uR': charge, 'dR': charge})
DFSZ3a=model('DFSZ-IIIa', {'lL': 0, 'eR': -(2*Chargeu+Charged), 'qL': 0, 'uR': Chargeu, 'dR': Charged})
DFSZ3b=model('DFSZ-IIIb', {'lL': 0, 'eR': Charged, 'qL': 0, 'uR': Chargeu, 'dR': Charged})
DFSZ3c=model('DFSZ-IIIc', {'lL': 0, 'eR': Chargeu+2*Charged, 'qL': 0, 'uR': Chargeu, 'dR': Charged})
DFSZ3d=model('DFSZ-IIId', {'lL': 0, 'eR': -Chargeu, 'qL': 0, 'uR': Chargeu, 'dR': Charged})

list_of_DFSZ_models=[DFSZ1, DFSZ2, DFSZ3a, DFSZ3b, DFSZ3c, DFSZ3d]
for models in list_of_DFSZ_models:
   print(models.model_name,'E/N= ',sp.Rational(sp.simplify(models.couplings['cgamma']/models.couplings['cg'])).limit_denominator())





# This file was automatically created by ALPaca

from object_library import all_propagators, Propagator


# define only once the denominator since this is always the same
denominator = "P('mu', id) * P('mu', id) - Mass(id) * Mass(id) + complex(0,1) * Mass(id) * Width(id)"

# propagator for the scalar
S = Propagator(name = "S",
               numerator = "complex(0,1)",
               denominator = denominator
               )

# propagator for the incoming fermion # the one for the outcomming is computed on the flight
F = Propagator(name = "F",
                numerator = "complex(0,1) * (Gamma('mu', 1, 2) * P('mu', id) + Mass(id) * Identity(1, 2))",
                denominator = denominator
               )

# massive vector in the unitary gauge, can't be use for massless particles
V1 = Propagator(name = "V1",
                numerator = "complex(0,1) * (-1 * Metric(1, 2) + Metric(1,'mu')* P('mu', id) * P(2, id) / Mass(id)**2)",
                denominator = denominator
               )

# massless vector and massive vector in unitary gauge
V2 = Propagator(name = "V2",
                numerator = "complex(0,-1) * Metric(1, 2)",
                denominator =  "P('mu', id) * P('mu', id)"
               )



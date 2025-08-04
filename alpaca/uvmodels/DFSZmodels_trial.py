from ..constants import GF, mW, mu, mc, mt
class DFSZSpecificModel(ModelBase):
    """A class to define a model of the type DFSZ given the PQ charges of the SM fermions.

    """
    def X1(mq, mHc, mW):
        aux = 2 + (mHc**2)/(mHc**2-mq**2) - 3*mW**2/(mq**2-mW**2) +\
        3*mW**4*(mHc**2+mW**2-2*mq**2)/((mHc**2-mW**2)(mq**2-mW**2)**2)*np.log(mq**2/mW**2) +\
        mHc**2/(mHc**2-mq**2)*(mHc**2/(mHc**2-mq**2)-6*mW**2/(mHc**2-mW**2))*np.log(mq**2/mHc**2)
        return aux
    
    def X2(mq, mHc):
        aux = -2*mq**2/(mHc**2-mq**2)*(1+mHc**2/(mHc**2-mq**2)*np.log(mq**2/mHc**2))
        return aux
    
    def __init__(self, model_name: str, charges: dict[str, sp.Expr], masses: dict[str, sp.Expr], vevs: dict[str, sp.Expr]):
        """Initialize the model with the given name and PQ charges.

        Arguments
        ---------
        model_name : str
            The name of the model.
        charges : dict[str, sp.Expr]
            A dictionary with the PQ charges of the SM fermions. The keys are the names of the fermions in the unbroken phase: 'cqL', 'cuR', 'cdR', 'clL', 'ceR'.
        masses : dict[str, sp.Expr]
            Mass of the charged Higgses in 2HDM. The keys are the names of Higgses.
        vevs : dict[str, sp.Expr]
            Vevs of the charged Higgses in 2HDM. The keys are the names of the 2 Higgs doublets.

        Raises
        ------
        NotImplementedError
            If nonuniversal is True.
        """
        super().__init__(model_name)
        masses = {f: 0 for f in ['mH+']} | masses # initialize to zero all missing masses
        self.masses = {key: sp.sympify(value) for key, value in masses.items()}  # Convert all values to sympy objects
        
        masses_np = {key: np.broadcast_to(value, 1) for key, value in masses.items()}  # Convert all values to numpy arrays

        vevs = {f: 0 for f in ['vu', 'vd']} | vevs # initialize to zero all missing vevs
        self.vevs = {key: sp.sympify(value) for key, value in vevs.items()}  # Convert all values to sympy objects
        
        vevs_np = {key: np.broadcast_to(value, 1) for key, value in vevs.items()}  # Convert all values to numpy arrays

        charges = {f: 0 for f in ['lL', 'eR', 'qL', 'uR', 'dR']} | charges # initialize to zero all missing charges
        self.charges = {key: sp.sympify(value) for key, value in charges.items()}  # Convert all values to sympy objects
        
        charges_np = {key: np.broadcast_to(value, 3) for key, value in charges.items()}  # Convert all values to numpy arrays
        for f in ['qL', 'uR', 'dR', 'lL', 'eR']:
            if np.array(self.charges[f]).shape == ():
                self.couplings[f'c{f}'] = -self.charges[f]
            else:
                self.couplings[f'c{f}'] = - sp.diag(charges_np[f].tolist(), unpack=True)
        
        mquark = [mu, mc, mt]
        self.couplings['cuR'][1, 2] = self.couplings['cuR'][1, 2] + \
            - GF/(16*np.pi**2)*(vevs_np['vd']/np.sqrt(vevs_np['vd']**2 + vevs_np['vu']))**2 *1/3*(np.sum([np.conjugate(VCKM[i,0])*VCKM[i,1]*mquark[i]*(X1(mquark[i], masses_np, mW)+X2(mquark[i], masses_np) * (vevs_np['vd']/vevs_np['vu'])**2) for i in range(len(mquark))]))

        self.couplings['cG'] = -sp.Rational(1,2) * sp.simplify(np.sum(
            2 * charges_np['qL'] - charges_np['dR'] - charges_np['uR']
        ))
        self.couplings['cW'] = -sp.Rational(1,2) * sp.simplify(np.sum(
            3 * charges_np['qL'] + charges_np['lL']
        ))
        self.couplings['cB'] = -sp.Rational(1,6) * sp.simplify(np.sum(
            charges_np['qL'] - 8 * charges_np['uR'] - 2 * charges_np['dR'] + 3 * charges_np['lL'] - 6 * charges_np['eR']
        ))


    def initialize(self):
        citations.register_inspire('DiLuzio:2020wdo')
        citations.register_inspire('Alonso-Alvarez:2021ett')
    
    def _repr_markdown_(self):
        md = super()._repr_markdown_()
        md += "<details><summary><b>PQ charges:</b></summary>\n\n"
        for f, c in self.charges.items():
            md += f"- $\\mathcal{{X}}{couplings_latex['c'+f][1:]} = {sp.latex(c)}$\n"
        md += "\n\n</details>"
        return md
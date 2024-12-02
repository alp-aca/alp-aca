// m: mass [MeV]
// tau: lifetime [ps]
// type: 0 = relative, 1 = absolute limits
// returns UL at 95% CL, or -1 if no limit is set where requested
double getUL(double m, double tau, int type){  
  // veto regions
  if(m < 214) return -1;
  if(m > 4350) return -1;
  if(m > 1000 && m < 1040 && tau < 1) return -1;
  if(m > 742 && m < 822 && tau < 1) return -1;
  if(m > 2913 && m < 3211) return -1;
  if(m > 3609 && m < 3875) return -1;

  TFile f("lhcb_2015_036.root");
  TGraph *g = (TGraph*)f.Get("g_masses");
  // find best mass
  int n = g->GetN(),mmin=-1;
  double min=1e6;
  for(int i=0; i<n; i++){
    if(fabs(m-g->GetX()[i]) < min){
      min = fabs(m-g->GetX()[i]);
      mmin = int(g->GetX()[i]);
    }
  }
  char name[100];
  if(type == 0) sprintf(name,"rel_ul_spline_%d",mmin);
  else if(type == 1) sprintf(name,"abs_ul_spline_%d",mmin);
  else {
    cout << "Only types are 0 and 1.  Please try again." << endl; 
    return -666;
  }
  TSpline3 *spl = (TSpline3*)f.Get(name);
  if(tau < 1000){
    return spl->Eval(tau);
  }
  double ul1000 = spl->Eval(1000);
  return ul1000/(1-exp(-1000/tau))*(1-exp(-1));
}

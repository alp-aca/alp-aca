// Simple test macro that produces a plot similar to Fig 2 in Supplemental 
// Material of the Letter.
// tau: lifetime [ps]
// type: 0 = relative, 1 = absolute limits
void test(double tau, int type){
  gROOT->ProcessLine(".L getUL.C");

  TFile f("lhcb_2015_036.root");
  TGraph *gm = (TGraph*)f.Get("g_masses");
  char name[100];
  int nm = gm->GetN();
  double ul[2000];
  for(int i=0; i<nm-1; i++){
    ul[i] = getUL(gm->GetX()[i],tau,type);
  }
 
  TCanvas *c1 = new TCanvas("c1","",25,25,1400,500); 
  c1->cd();
  c1->SetBorderMode(0);
  c1->SetTopMargin(0.05);
  c1->SetBottomMargin(0.15);
  c1->SetLeftMargin(0.1);
  c1->SetRightMargin(0.03);  
  gPad->SetLogy();

  TH1F h("h","",1000,214,4350);
  h.GetXaxis()->SetTitleFont(132);
  h.GetXaxis()->SetTitleSize(0.06);
  h.GetYaxis()->SetTitleFont(132);
  h.GetYaxis()->SetTitleSize(0.06);
  h.GetXaxis()->SetTitle("#it{m(#mu^{+}#mu^{-})} [MeV]");
  h.GetXaxis()->SetLabelSize(0.05);
  h.GetYaxis()->SetLabelSize(0.05);
  h.GetXaxis()->SetLabelFont(132);
  h.GetYaxis()->SetLabelFont(132);
  h.GetYaxis()->SetTitleOffset(0.8);
  if(type == 0) h.GetYaxis()->SetTitle("relative UL at 95% CL");
  else h.GetYaxis()->SetTitle("absolute UL at 95% CL");
  if(type == 0){
    h.SetMinimum(2e-3);
    h.SetMaximum(1);
  }
  else{
    h.SetMinimum((2e-3)*(1.6e-7));
    h.SetMaximum(1.6e-7);
  }
  h.DrawCopy();
  TGraph *gul = new TGraph(nm-1,gm->GetX(),ul);
  gul->SetMarkerStyle(20);
  gul->Draw("psame");
}

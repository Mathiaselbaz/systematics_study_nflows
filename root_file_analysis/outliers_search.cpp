#include "utils_root.hxx"

void outliers_search(string filename = "../Dataset/ROOT/allParams/test200k.root"){

  // Open TFile
  TFile *f = new TFile(filename.c_str());
  // Open TTree
  TTree *t = (TTree*)f->Get("margThrow");

  // Get number of toys
  int Toys = t->GetEntries();

  // variables for TTree branches
  double NLL,priorSum;
  double ePrior=0, LH;
  double gNLL=0;
  std::vector<double> *prior{0};
  std::vector<double> *params{0};
  std::vector<bool> *marginalise{0};
  std::vector<double> *gLLH_vector{0};

  t->SetBranchAddress("priorSum", &priorSum);
  t->SetBranchAddress("LLH", &NLL);
  t->SetBranchAddress("Parameters", &params);
  t->SetBranchAddress("Marginalise", &marginalise);
  t->SetBranchAddress("prior", &prior);
  t->SetBranchAddress("weightsChiSquare", &gLLH_vector);
  t->SetBranchAddress("gLLH", &gNLL);

  t->GetEntry(0);
  // Get number of params
  int Nparams = params->size();

  // bins for showing the marginalised variable
  int bins = 200; //200
  // 2D L-g vs. logg (or logL)
  TH2D* Lminusg_vs_logg = new TH2D("Lminusg_vs_logg", ";-log(LH);-log(LH)-[-log(g)]", 1000, 900, 1150, 1000,-100,100);

  // Load mean parameters vector
  TVectorD* mean_params = root_utils::getBestFitParametersVector(f);
  TVectorD* prior_sigmas = root_utils::getPriorSigmasVector(f);
  // Load covariance matrix
  TH2D* postFitCovarianceMatrix = root_utils::getPostFitCovarianceMatrix(f);
  // Get sigmas from covariance matrix
  TVectorD* postFitSigmas = root_utils::getPostFitSigmasVector(f);
  // Convert to vector
  vector<double> mean_params_vector(mean_params->GetMatrixArray(), mean_params->GetMatrixArray() + mean_params->GetNrows());
  vector<double> prior_sigmas_vector(prior_sigmas->GetMatrixArray(), prior_sigmas->GetMatrixArray() + prior_sigmas->GetNrows());
  vector<double> postFitSigmas_vector(postFitSigmas->GetMatrixArray(), postFitSigmas->GetMatrixArray() + postFitSigmas->GetNrows());
  // Get parameter titles
  vector<string>* parameter_titles = root_utils::getParametersFullTitles(f);


  vector<TH1D*> distributions;
  vector<TH1D*> distributions_outliers;
  for(int iParam=0; iParam<Nparams; iParam++){
    distributions.push_back(new TH1D(Form("dist_%d",iParam),"",bins,mean_params_vector[iParam]-4*postFitSigmas_vector[iParam],mean_params_vector[iParam]+4*postFitSigmas_vector[iParam]));
    distributions_outliers.push_back(new TH1D(Form("dist_outliers_%d",iParam),"",bins,mean_params_vector[iParam]-4*postFitSigmas_vector[iParam],mean_params_vector[iParam]+4*postFitSigmas_vector[iParam]));
  }

  double log_weightcap = 20;

  ProgressBar progressBar(Toys);

  double weightSum = 0;
  double squaredWeightSum = 0;
  /////////////////// main loop
  cout<<"Main loop: filling histograms..."<<endl;
  for (int iToy=0; iToy<Toys; iToy++){
    t->GetEntry(iToy);
    if(gLLH_vector->size()!=Nparams){
      cerr<<"ERROR: chi square size is not equal to Nparams"<<endl;
      continue;
    }
    double log_weight = (- NLL + gNLL );

    for(int iParam=0; iParam<Nparams; iParam++){
      distributions[iParam]->Fill((*params)[iParam]);
      if(log_weight>log_weightcap || log_weight<-log_weightcap){
        distributions_outliers[iParam]->Fill((*params)[iParam]);
      }
    }

    Lminusg_vs_logg->Fill(NLL,NLL-gNLL);
    // update the progress bar
    progressBar.update();
  }// end of main loop on throws

  double average_weights = weightSum/Toys;
  double average_weights2 = squaredWeightSum/Toys;
  double variance_weights = average_weights2 - average_weights*average_weights;

  // compute the correlation (or covariance) matrices of the TTree
  TMatrixD* corrMatrix_gaussian = new TMatrixD(Nparams,Nparams);
  TMatrixD* corrMatrix_LH = new TMatrixD(Nparams,Nparams);

  progressBar.reset();

  // Draw the 2D histograms
  TCanvas *c1 = new TCanvas("c1","c1",800,800);
  Lminusg_vs_logg->Draw("colz");
  // draw a line at 0
  TLine *line = new TLine(0,0,3*Nparams,0);
  line->SetLineColor(kRed);
  line->SetLineStyle(2);
  line->Draw();
  // draw lines at +- log weight
  TLine *line1 = new TLine(0,log_weightcap,3*Nparams,log_weightcap);
  line1->SetLineColor(kRed);
  line1->Draw();
  TLine *line2 = new TLine(0,-log_weightcap,3*Nparams,-log_weightcap);
  line2->SetLineColor(kRed);
  line2->Draw();

  gROOT->SetBatch(kTRUE);
  // Draw and save the distributions
  for(int iParam=0; iParam<Nparams; iParam++){
    distributions[iParam]->Scale(1./distributions[iParam]->GetEntries());
    distributions_outliers[iParam]->Scale(1./distributions_outliers[iParam]->GetEntries());
    TCanvas *c2 = new TCanvas(Form("c2_%d",iParam),Form("c2_%d",iParam),800,800);
    distributions[iParam]->SetLineWidth(2);
    distributions[iParam]->Draw("hist");
    distributions_outliers[iParam]->SetLineColor(kRed);
    distributions_outliers[iParam]->SetFillColor(kRed);
    distributions_outliers[iParam]->SetFillStyle(3305);
    distributions_outliers[iParam]->Draw("hist same");
    distributions[iParam]->SetTitle(parameter_titles->at(iParam).c_str());
    distributions_outliers[iParam]->SetTitle(parameter_titles->at(iParam).c_str());
    // Overlay the postfit expected gaussian
    TF1 *f1 = new TF1("f1","gaus",mean_params_vector[iParam]-4*postFitSigmas_vector[iParam],mean_params_vector[iParam]+4*postFitSigmas_vector[iParam]);
    f1->SetParameter(0,distributions[iParam]->GetMaximum());
    f1->SetParameter(1,mean_params_vector[iParam]);
    f1->SetParameter(2,postFitSigmas_vector[iParam]);
    f1->SetLineColor(kBlue);
    f1->Draw("same");
    if(iParam==0) {
      c2->Print("outliers.pdf[");
    }else if(iParam==Nparams-1){
      c2->Print("outliers.pdf]");
    }else{
      c2->Print("outliers.pdf",Form("Title:par_%d",iParam));
    }

//    c2->SaveAs(Form("distributions_outliers/distributions_%d.png",iParam));
  }




}

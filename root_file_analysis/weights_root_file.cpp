#include <TApplication.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include <TLegend.h>

#include "ProgressBar.h"

using namespace std;

int weights_root_file(string filename){

    cout<<"Opening file: "<<filename<<endl;

    // Open TFile
    TFile *f = new TFile(filename.c_str());
    if(not f){
      cerr<<"ERROR: file not found"<<endl;
      return 1;
    }
    if(f->IsZombie()){
      cerr<<"ERROR: file not found"<<endl;
      return 1;
    }
    // Open TTree
    TTree *t = (TTree*)f->Get("margThrow");
    if(not t){
      cerr<<"ERROR: TTree not found"<<endl;
      return 1;
    }


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
    // -logLH and gaussian approximation histograms
    TH1D* NLL_histo = new TH1D("NLL_histo", "NLL/2", 10*Nparams, 0, 3*Nparams);
    TH1D* gNLL_histo = new TH1D("gNLL_histo", "gNLL/2", 10*Nparams, 0, 3*Nparams);
    // L/g histogram (distribution of weights for the variance)
    TH1D* Loverg_histo = new TH1D("Loverg_histo", "L/g", 200000, 0, 20000);
    // L-g histogram
    TH1D* Lminusg_histo = new TH1D("Lminusg_histo", "L-g", 1000, -100, 300);
    // 2D L-g vs. logg (or logL)
    TH2D* Lminusg_vs_logg = new TH2D("Lminusg_vs_logg", "log(g) vs. log(L)-log(g);log(L)-log(g);log(g)", 1000, -30, 30, Nparams, 0, 2*Nparams);
    // 2D log(L) vs. log(g)
    TH2D* logL_vs_logg = new TH2D("logL_vs_logg", "-log(g) vs. -log(L);-log(L);-log(g)", Nparams*4, 0, 4*Nparams, 4*Nparams, 0, 4*Nparams);


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
        double weight = (- NLL + gNLL );
        weight = exp(weight);
        weightSum += weight;
        squaredWeightSum += weight*weight;

//        if(weight>1000){
//          cout<<"---------------------------------------------------\n";
//          cout<<"Event: "<<iToy<<" | negative-log-likelihood: "<<NLL<<" | negative-log-gaussian: "<<gNLL<<" weight(LH/g): "<<weight<<endl;
//        }

        NLL_histo->Fill(NLL);
        gNLL_histo->Fill(gNLL);
        // also plot the L/g
        Loverg_histo->Fill(weight);
        Lminusg_histo->Fill(NLL-gNLL);
        Lminusg_vs_logg->Fill(NLL-gNLL,gNLL);
        logL_vs_logg->Fill(NLL,gNLL);
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

    //draw L/g and print out its variance on the legend
    TCanvas* c2 = new TCanvas("c2","c2",1600,1000);
    c2->cd();
    Loverg_histo->Draw("hist");
    //Lminusg_vs_logg->Draw("colz");
    cout<<"\nVariance of weights: "<<variance_weights<<endl;
    Loverg_histo->SetTitle("Distribution of weights;L/g;counts");
    gPad->SetLogy();
    gPad->SetLogx();
    c2->SaveAs("weightsOutput/weights_distribution.pdf");
    c2->SaveAs("weightsOutput/weights_distribution.root");

    // draw log(L) vs. log(g)
    TCanvas* c5 = new TCanvas("c5","c5",1600,1000);
    c5->cd();
    logL_vs_logg->Draw("colz");
    c5->SaveAs("weightsOutput/logL_vs_logg.pdf");
    c5->SaveAs("weightsOutput/logL_vs_logg.root");

    // draw L-g and print out its variance on the legend
    TCanvas* c4 = new TCanvas("c4","c4",1600,1000);
    c4->cd();
    Lminusg_histo->Draw("hist");
    Lminusg_histo->SetTitle("log(L)-log(g);log(L)-log(g);counts");
    c4->SaveAs("weightsOutput/Lminusg_distribution.pdf");
    c4->SaveAs("weightsOutput/Lminusg_distribution.root");
    

    //draw NLL and gNLL
    TCanvas* c1 = new TCanvas("c1","c1",1600,1000);
    c1->cd();
    NLL_histo->SetLineColor(kBlack);
    NLL_histo->SetLineWidth(2);
    NLL_histo->Draw("hist");
    gNLL_histo->SetLineColor(kRed);
    gNLL_histo->SetLineWidth(2);
    gNLL_histo->Draw("hist same");
    TLegend* leg3 = new TLegend(0.6,0.4,0.99,0.7);
    leg3->AddEntry(NLL_histo,"-log(LH)","l");
    leg3->AddEntry(gNLL_histo,"-log(g) (gaussian approx.)","l");
    NLL_histo->SetTitle("-log(LH) and -log(g) ;; counts");
    leg3->Draw("same");
    c1->SaveAs("weightsOutput/NLL_gNLL_distribution.pdf");
    c1->SaveAs("weightsOutput/NLL_gNLL_distribution.root");

    // draw the marg vs profiled res


    // Effective sample size
    double ess = weightSum*weightSum/squaredWeightSum;
    cout<<"Γ---------------------------------------\n";
    cout<<"|     Real sample size: "<<Toys<<endl;
    cout<<"|Effective sample size: "<<ess<<endl;
    cout<<"L---------------------------------------\n";

    return 0;
}


int main(int argc, char** argv){
  // Create TApplication
  //TApplication app("weights", &argc, argv);

  cout<<"argc: "<<argc<<endl;
  if(argc==2){
    weights_root_file(argv[1]);
  }else{
    std::cout<<"Usage: ./weightsApp <filename>"<<std::endl;
  }
  return 0;
}

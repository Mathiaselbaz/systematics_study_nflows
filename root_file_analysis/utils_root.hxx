//
// Created by Lorenzo Giannessi on 22.01.2025.
//

#ifndef SYSTEMATICS_STUDY_NFLOWS_UTILS_ROOT_HXX
#define SYSTEMATICS_STUDY_NFLOWS_UTILS_ROOT_HXX

#include "ProgressBar.h"

namespace root_utils {

  TString color_map(int color_int){
    switch (color_int)
    {
      case 1:
        return "kBlack";
        break;
      case 632:
        return "kRed";
        break;
      case 416:
        return "kGreen";
        break;
      case 600:
        return "kBlue";
        break;
      case 602:
        return "kBlue";
        break;
      case 616:
        return "kMagenta";
        break;
      default:
        return TString(to_string(color_int))+" (unknown color)";
        break;
    }
  }

  TVectorD* getBestFitParametersVector(TFile *f){
    TDirectory* postFitDir = (TDirectory*)f->Get("postFitInfo");
    if(not postFitDir){
      std::cout << "ERROR: postFitInfo not found in file" << std::endl;
      return nullptr;
    }
    // Extract prior and postFit information (best fit values and convaraince matrix)
    TVectorD* bestFit_TVectorD = dynamic_cast<TVectorD*>(f->Get("bestFitParameters_TVectorD"));
    if (not bestFit_TVectorD) {
      std::cout << "Best fit TVectorD not found in file. Looking in the TDirectory postFitInfo... ";
      postFitDir->cd();
      bestFit_TVectorD = dynamic_cast<TVectorD*>(postFitDir->Get("bestFitParameters_TVectorD"));
      f->cd();
      if (not bestFit_TVectorD) {
        std::cout << "ERROR: best fit TVectorD not found!" << std::endl;
        return nullptr;
      }
    }

    return bestFit_TVectorD;
  }

  TVectorD* getPriorParametersVector(TFile *f){
    TVectorD* priorParameters_TVectorD = dynamic_cast<TVectorD*>(f->Get("priorParameters_TVectorD"));
    if (not priorParameters_TVectorD) {
      std::cout << "WARNING: prior values TVectorD not found in file. Missing prior info." << std::endl;
      return nullptr;
    }
    return priorParameters_TVectorD;
  }

  TVectorD* getPriorSigmasVector(TFile *f) {
    TVectorD *priorSigmas_TVectorD = dynamic_cast<TVectorD *>(f->Get("priorSigmas_TVectorD"));
    if (not priorSigmas_TVectorD) {
    std::cout << "WARNING: prior sigmas TVectorD not found in file. Missing prior info." <<
    std::endl;
    return nullptr;
    }
    return priorSigmas_TVectorD;
  }

  std::vector<std::string>* getParametersFullTitles(TFile *f){
    std::vector<std::string> *parameterFullTitles;
    string nn = "noname";
    f->GetObject("parameterFullTitles",parameterFullTitles);
    if (not parameterFullTitles) {
    std::cout << "ERROR: parameterFullTitles not found in postFitInfo" << std::endl;
    return nullptr;
    }
    return parameterFullTitles;
  }

  TNamed* getBestFitParameters_TNamed(TFile* f){
    TDirectory* postFitDir = (TDirectory*)f->Get("postFitInfo");
    if(not postFitDir){
      std::cout << "ERROR: postFitInfo not found in file" << std::endl;
      return nullptr;
    }
    postFitDir->cd();
    TNamed* bestFit_TNamed = dynamic_cast<TNamed*>(postFitDir->Get("postFitParameters_TNamed"));
    if (not bestFit_TNamed) {
    std::cout << "ERROR: bestFit_TNamed not found in postFitInfo" << std::endl;
    return nullptr;
    }
    return bestFit_TNamed;
  }

  TH2D* getPostFitCovarianceMatrix(TFile *f){
    TDirectory* postFitDir = (TDirectory*)f->Get("postFitInfo");
    if(not postFitDir){
      std::cout << "ERROR: postFitInfo not found in file" << std::endl;
      return nullptr;
    }
    TH2D* covMatrix = dynamic_cast<TH2D*>(postFitDir->Get("postFitCovarianceOriginal_TH2D"));
    if (not covMatrix) {
      std::cout << "ERROR: covMatrix not found in postFitInfo" << std::endl;
      return nullptr;
    }
    // HACK
    double hadd_factor = covMatrix->GetEntries()/(covMatrix->GetNbinsX()*covMatrix->GetNbinsY());
    for(int i=0;i<covMatrix->GetNbinsX();i++){
      for(int j=0;j<covMatrix->GetNbinsY();j++){
        covMatrix->SetBinContent(i+1,j+1,covMatrix->GetBinContent(i+1,j+1)/sqrt(hadd_factor));
      }
    }

    return covMatrix;
  }

  TH2D* getPostFitCorrelationMatrix(TFile *f){
    TH2D* covMatrix = getPostFitCovarianceMatrix(f);

    TH2D* corrMatrix = (TH2D*)covMatrix->Clone("corrMatrix");
    for(int i=0;i<corrMatrix->GetNbinsX();i++){
      for(int j=0;j<corrMatrix->GetNbinsY();j++){
        corrMatrix->SetBinContent(i+1,j+1,covMatrix->GetBinContent(i+1,j+1)/sqrt(covMatrix->GetBinContent(i+1,i+1)*covMatrix->GetBinContent(j+1,j+1)));
      }
    }
    return corrMatrix;
  }

  TVectorD* getPostFitSigmasVector(TFile *f){
    TH2D* covMatrix = getPostFitCovarianceMatrix(f);

    vector<double> postFitSigmas_vector;
    for(int i=0;i<covMatrix->GetNbinsX();i++){
      postFitSigmas_vector.push_back(sqrt(covMatrix->GetBinContent(i+1,i+1)));
    }
    TVectorD* postFitSigmas = new TVectorD(postFitSigmas_vector.size(),postFitSigmas_vector.data());

    return postFitSigmas;
  }

  void printCovarianceCorrelationMatrix(TH2D* covMatrix, TH2D* corrMatrix, bool drawText = false){

    // print the Covariance matrix as Th2D
    TCanvas* c_cov = new TCanvas("c_cov","c_cov",1600,800);
    c_cov->Divide(2,1);
    c_cov->cd(1);
    covMatrix->SetTitle("Covariance matrix");
    covMatrix->Draw("colz");
    c_cov->cd(2);
    corrMatrix->SetTitle("Correlation matrix");
    corrMatrix->Draw("colz");
    for(int i=1;i<=covMatrix->GetNbinsX();i++){
      for(int j=1;j<=covMatrix->GetNbinsY();j++){
        double value = covMatrix->GetBinContent(i, j);
        TText *text = new TText(covMatrix->GetXaxis()->GetBinCenter(i), covMatrix->GetYaxis()->GetBinCenter(j), Form("%.4f", value));
        text->SetTextSize(0.02);
        text->SetTextAlign(22); // Centered
        c_cov->cd(1);
        if(drawText) text->Draw();
        value = corrMatrix->GetBinContent(i, j);
        text = new TText(corrMatrix->GetXaxis()->GetBinCenter(i), corrMatrix->GetYaxis()->GetBinCenter(j), Form("%.4f", value));
        text->SetTextSize(0.02);
        text->SetTextAlign(22); // Centered
        c_cov->cd(2);
        if(drawText) text->Draw();
      }
    }
    c_cov->Update();

  }

//  void makeMatrixOfHistograms(TTree* t, TVectorD* bestFit_TVectorD, TH2D* covMatrix, std::vector<std::string>* params) {
//    // you DON'T want to do this if the dimensionality is too large
//    if (params->size() > 100){
//      return;
//    }
//    // make a matrix of histograms
//    vector < TH1D * > histo_parameter;
//    vector < TH1D * > histo_parameter_gaussian;
//    vector < TH2D * > histo_parameter2D;
//    vector < TH2D * > histo_parameter2D_gaussian;
//    int index = 0;
//      for (int iCol = 0; iCol < params->size(); iCol++) {
//        for (int iRow = 0; iRow < params->size(); iRow++) {
//          if (iRow == iCol) {
//            histo_parameter.push_back(
//                    new TH1D("histo_" + TString(to_string(iCol)) + "_" + TString(to_string(iRow)), "", 100,
//                             bestFit_TVectorD->GetMatrixArray()[iCol] -
//                             4 * sqrt(covMatrix->GetBinContent(iCol + 1, iCol + 1)),
//                             bestFit_TVectorD->GetMatrixArray()[iCol] +
//                             4 * sqrt(covMatrix->GetBinContent(iCol + 1, iCol + 1))
//                    ));
//            histo_parameter_gaussian.push_back(
//                    new TH1D("histo_gaussian_" + TString(to_string(iCol)) + "_" + TString(to_string(iRow)), "", 100,
//                             bestFit_TVectorD->GetMatrixArray()[iCol] -
//                             4 * sqrt(covMatrix->GetBinContent(iCol + 1, iCol + 1)),
//                             bestFit_TVectorD->GetMatrixArray()[iCol] +
//                             4 * sqrt(covMatrix->GetBinContent(iCol + 1, iCol + 1))
//                    ));
//          } else if (iRow > iCol) {
//            histo_parameter2D.push_back(
//                    new TH2D("histo_" + TString(to_string(iCol)) + "_" + TString(to_string(iRow)), "", 100,
//                             bestFit_TVectorD->GetMatrixArray()[iRow] -
//                             4 * sqrt(covMatrix->GetBinContent(iRow + 1, iRow + 1)),
//                             bestFit_TVectorD->GetMatrixArray()[iRow] +
//                             4 * sqrt(covMatrix->GetBinContent(iRow + 1, iRow + 1)),
//                             100,
//                             bestFit_TVectorD->GetMatrixArray()[iCol] -
//                             4 * sqrt(covMatrix->GetBinContent(iCol + 1, iCol + 1)),
//                             bestFit_TVectorD->GetMatrixArray()[iCol] +
//                             4 * sqrt(covMatrix->GetBinContent(iCol + 1, iCol + 1))
//                    ));
//            histo_parameter2D_gaussian.push_back(
//                    new TH2D("histo_gaussian_" + TString(to_string(iCol)) + "_" + TString(to_string(iRow)), "", 100,
//                             bestFit_TVectorD->GetMatrixArray()[iRow] -
//                             4 * sqrt(covMatrix->GetBinContent(iRow + 1, iRow + 1)),
//                             bestFit_TVectorD->GetMatrixArray()[iRow] +
//                             4 * sqrt(covMatrix->GetBinContent(iRow + 1, iRow + 1)),
//                             100,
//                             bestFit_TVectorD->GetMatrixArray()[iCol] -
//                             4 * sqrt(covMatrix->GetBinContent(iCol + 1, iCol + 1)),
//                             bestFit_TVectorD->GetMatrixArray()[iCol] +
//                             4 * sqrt(covMatrix->GetBinContent(iCol + 1, iCol + 1))
//                    ));
//            index++;
//          }
//        } // iRow
//      } // iCol
//
//
//    cout<<"Computing covariance matrices..."<<endl;
//    for(int i_Entry=0;i_Entry<t->GetEntries();i_Entry++){
//      t->GetEntry(i_Entry);
//      index = 0;
//      for(int iCol = 0 ; iCol < params->size() ; iCol++){
//        for(int iRow = 0 ; iRow < params->size() ; iRow++){
//          (*corrMatrix_gaussian)[iCol][iRow] +=
//                  (params->at(iCol)-meanVector_gaussian[iCol]) *
//                  (params->at(iRow)-meanVector_gaussian[iRow]) /
//                  (double)Toys;
//          (*corrMatrix_LH)[iCol][iRow] +=
//                  (params->at(iCol)-meanVector[iCol]) *
//                  (params->at(iRow)-meanVector[iRow]) *
//                  exp(2*(- NLL + gNLL)) /
//                  (double)Toys;
//          // fill histograms representing all covariances ("matrix of histograms")
//          if(Nparams<100){
//            if(iRow==iCol){
//              if(true) { //exp(- NLL + gNLL)>100
//                histo_parameter[iCol]->Fill(params->at(iCol), exp(-NLL + gNLL));
//                histo_parameter_gaussian[iCol]->Fill(params->at(iCol));
//              }
//            }
//            else if (iRow>iCol) {
//              if(true) { //exp(- NLL + gNLL)>100
//                histo_parameter2D[index]->Fill(params->at(iRow), params->at(iCol), exp(-NLL + gNLL));
//                histo_parameter2D_gaussian[index]->Fill(params->at(iRow), params->at(iCol));
//                //if(i_Entry==0) cout<<index<<" histo_"+TString(to_string(iCol))+"_"+TString(to_string(iRow))<<endl;
//              }
//              index++;
//            }
//          }
//        } // end of loop on rows
//      } // end of loop on columns
//      progressBar.update();
//    } // end of loop on entries
//
//    // print big canvas with the matrix of histograms
//    TCanvas* c_histo_matrix = new TCanvas("c_histo_matrix","c_histo_matrix",1600,1000);
//    c_histo_matrix->Divide(params->size(),params->size());
//    index=0;
//    for(int iCol = 0 ; iCol < params->size() ; iCol++){
//      for(int iRow = 0 ; iRow < params->size() ; iRow++){
//        c_histo_matrix->cd(iCol*params->size()+iRow+1);
//        // set pad margins to 0
//        gPad->SetLeftMargin(0); gPad->SetRightMargin(0); gPad->SetTopMargin(0); gPad->SetBottomMargin(0);
//        if(iRow==iCol) {
//          histo_parameter[iCol]->Draw("hist");
////            gPad->SetLogy();
//          gPad->Update();
//        }
//        else if (iRow>iCol) {
//          histo_parameter2D[index]->Draw("colz");
//          gPad->Update();
//          index++;
//
//        }
//      } // iRow
//    } // iCol
//    gStyle->SetOptStat(0);
//    // repeat for the gaussian histograms
//    TCanvas* c_histo_matrix_gaussian = new TCanvas("c_histo_matrix_gaussian","c_histo_matrix_gaussian",1600,1000);
//    c_histo_matrix_gaussian->Divide(params->size(),params->size());
//    index=0;
//    for(int iCol = 0 ; iCol < params->size() ; iCol++){
//      for(int iRow = 0 ; iRow < params->size() ; iRow++){
//        c_histo_matrix_gaussian->cd(iCol*params->size()+iRow+1);
//        // set pad margins to 0
//        gPad->SetLeftMargin(0); gPad->SetRightMargin(0); gPad->SetTopMargin(0); gPad->SetBottomMargin(0);
//        if(iRow==iCol) histo_parameter_gaussian[iCol]->Draw("colz");
//        else if (iRow>iCol) {
//          histo_parameter2D_gaussian[index]->Draw("colz");
//          index++;
//        }
//      } // iRow
//    } // iCol
//    gStyle->SetOptStat(0);
//
//  }
//}

};

#endif //SYSTEMATICS_STUDY_NFLOWS_UTILS_ROOT_HXX

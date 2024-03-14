#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "TList.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
//#include "TriggerInfo.h"
#include "TDirectoryFile.h"
#include "TString.h"
using namespace std;

//58 is enough to store full pathname
const char* OPEN_PATH = "/Users/benkroul/Documents/CS/final_229/roots/";
const char* SAVE_PATH = "/Users/benkroul/Documents/CS/final_229/csvs/";
const char* fileNames[11] = {"AliVSD_MasterClass_1","AliVSD_Masterclass_1","AliVSD_Masterclass_2","AliVSD_Masterclass_3","AliVSD_Masterclass_4","AliVSD_Masterclass_5","AliVSD_Masterclass_6","AliVSD_Masterclass_7","AliVSD_Masterclass_8","AliVSD_Masterclass_9","AliVSD_Masterclass_10"};

// store tree into new filename
void storeTree(const char* tree_name, TDirectoryFile* f) {
   const char* fname = f->GetName();
   printf("name found %s\n",fname);
   int len = strlen(SAVE_PATH)+strlen(fname)+strlen(tree_name)+6;
   char newfname[len];
   strcpy(newfname, SAVE_PATH);
   strcat(newfname, fname);
   strcat(newfname, tree_name);
   strcat(newfname, ".root\0");

   // retrieve tree and clone it
   printf("retrieving tree %s...\n",tree_name);
   TTree* oldtree = f->Get<TTree>(tree_name);
   //oldtree = (TTree *)oldtree;
   //TTree * oldtree = (TTree *)obj;
   printf("found tree %s...\n",tree_name);
   printf("pointer to tree %p\n",oldtree);
   oldtree->Print();
   printf("pointer to tree dereferenced %p\n",&oldtree);
   int numentries = oldtree->GetEntries();

   printf("tree has %d entries\n", numentries);
   TObjArray* branchesList = oldtree->GetListOfBranches();
   printf("accessed list of branches\n");
   for (TObject * obj: *branchesList) {
      const char* name = obj->GetName();
      printf("name is %s\n",name);
   }
   oldtree->SetBranchStatus("*", 1);
   printf("set branch status...\n");
   TFile newfile(newfname, "recreate");
   printf("now created %s\n",newfname);
   printf("made file %s...\n",newfname);
   
   auto newtree = oldtree->CloneTree();
   printf("cloned tree...\n");
   newtree->Print();
   newfile.Write();
}

// read tree and store in .txt file "csv"
void storeCSV(const char* tree_name, TDirectoryFile* f) {
   const char* fname = f->GetName();
   int len = strlen(SAVE_PATH)+strlen(fname)+strlen(tree_name)+5;
   char newfname[len];
   strcpy(newfname, SAVE_PATH);
   strcat(newfname, fname);
   strcat(newfname, tree_name);
   strcat(newfname, ".txt\0");
   
   fstream myfile; // make output file
   myfile.open(newfname);

   TTree *tree = (TTree*)f->Get(tree_name);
   tree->Print(); // print tree branches to screen
   //tree->Scan(); pritn all entries to screen
   
   // implement TTreeReader class
   TTreeReader reader(tree);
   TTreeReaderValue<UShort_t> fId0s(reader, "fDetId");
   TTreeReaderValue<UShort_t> fId1s(reader, "fSubdetId");
   TTreeReaderValue<Int_t>    fId2s(reader, "fLabel[3]");
   TTreeReaderValue<Float_t>  fXs(reader, "fV.fX");
   TTreeReaderValue<Float_t>  fYs(reader, "fV.fY");
   TTreeReaderValue<Float_t>  fZs(reader, "fV.fZ");
   
   myfile << "fDetId;fSubdetId;fLabel[3];fV.fX;fV.fY;fV.fZ\n";
   bool firstEntry = true;
   while (reader.Next()) {
      // convert from the wierd Int_t types to regular C types
      unsigned short int fId0 = (unsigned short int)(*fId0s);
      unsigned short int fId1 = (unsigned short int)(*fId1s);
      int fId2 = (int)(*fId2s);
      float fX = (float)(*fXs);
      float fY = (float)(*fYs);
      float fZ = (float)(*fZs);
      int length = snprintf(NULL,0,"%hu;%hu;%d;%f;%f:%f",fId0,fId1,fId2,fX,fY,fZ);
      char* str = (char *)malloc( length + 5 );
      snprintf(str,length + 1,"%hu;%hu;%d;%f;%f:%f",fId0,fId1,fId2,fX,fY,fZ);
      strcat(str, "\n\0");
      myfile << str;
   }
   if (reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd) {
      printf("reader did not read all entries successfully");
   }

   myfile.close();
}

void treeToCSV() {
   // make pathname from indexed root file
   int index = 1;
   const char* fname = fileNames[index];
   char pathname[strlen(OPEN_PATH)+30];
   strcpy(pathname, OPEN_PATH);
   strcat(pathname, fname);
   strcat(pathname,".root");
   printf("%s\n", pathname);
   // open path and get list of events (34 for index 1)
   TFile *fd = TFile::Open(pathname);
   TList *events= fd->GetListOfKeys();

   //TDirectoryFile *f = fd->GetEntry(fname);
   printf("found %d entries\n", events->GetEntries());
   for(int i=0; i<events->GetEntries(); i++) {
      if (i == 0) {
         printf("at entry %d\n",i);
         TObject* obj = events->At(i);
         TDirectoryFile* f = (TDirectoryFile *)obj;
         const char* tree_name = "Clusters";
         storeTree(tree_name, f);
         storeTree("RecTracks", f);
         storeCSV("Clusters", f);
         storeCSV("RecTracks", f);
      }
   }
}

/*
void storeCSV0(const char* tree_name, TDirectoryFile* f) {
   const char* fname = f->GetName();
   char newfname[43+strlen(fname)+strlen(tree_name)+5];
   strcpy(newfname, SAVE_PATH);
   strcat(newfname, fname);
   strcat(newfname, tree_name);
   strcat(newfname, ".txt\0");
   
   fstream myfile; // make output file
   myfile.open(newfname);

   TTree *tree = (TTree*)f->Get(tree_name);
   // tree->Scan(); // print content to screen
   float *addr; // create variables of the same type as the branches you want to access
   events->SetBranchAddress("branch_name",&addr); // for all the TTree branches you need this
   //Event *event = 0;  //event must be null or point to a valid object
                     //it must be initialized
   //tree.SetBranchAddress("event",&event);
   // loop over the tree
   for (int i=0;i<tree->GetEntries();i++){
      tree->GetEntry(i);
      //char* a = tree->Something;
      //myfile << a << "\n";
   }
   myfile.close();
}
*/

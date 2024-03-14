#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
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
const char* DIRNAME = "/Users/benkroul/Documents/CS/final_229/";
const char* fileNames[11] = {"AliVSD_MasterClass_1","AliVSD_Masterclass_1","AliVSD_Masterclass_2","AliVSD_Masterclass_3","AliVSD_Masterclass_4","AliVSD_Masterclass_5","AliVSD_Masterclass_6","AliVSD_Masterclass_7","AliVSD_Masterclass_8","AliVSD_Masterclass_9","AliVSD_Masterclass_10"};

// store tree into new filename
void storeTree(const char* tree_name, TDirectoryFile* f, const int masterclass_index) {
   const char* fname = f->GetName();
   int len = strlen(DIRNAME)+strlen(fname)+17+2*strlen(tree_name);
   if (masterclass_index > 9) {
      len++;
   }
   char newfname[len];
   snprintf(newfname, len, "%sroots/%s/%d_%s_%s.root",DIRNAME,tree_name,masterclass_index,fname,tree_name);

   TTree* oldtree = f->Get<TTree>(tree_name);
   oldtree->SetBranchStatus("*", 1);
   TFile newfile(newfname, "recreate");
   auto newtree = oldtree->CloneTree();
   newtree->Print();
   newfile.Write();
}

typedef struct readerWithType {
      const char* type;
      void* reader;
} readerWithType;

readerWithType* makeTreeReaderFromType(const char* type, const char* name, TTreeReader reader) {
   // initialize a TTreeReaderValue() of the correct type
   void* ret = nullptr;
   if (!strcmp(type, "Int_t")) {
      TTreeReaderValue<Int_t>      ret(reader, name);
   } else if (!strcmp(type, "Float_t")) {
      TTreeReaderValue<Float_t>    ret(reader, name);
   } else if (!strcmp(type, "Double32_t")) {
      TTreeReaderValue<Double32_t> ret(reader, name);
   } else if (!strcmp(type, "UShort_t")) {
      TTreeReaderValue<UShort_t>   ret(reader, name);
   } else {
      printf("new type found in %s: %s",name,type);
   }
   readerWithType* newThing;
   newThing->type = type;
   newThing->reader = (void *)ret;
   return newThing;
}

void addReaderValueToFile(ofstream* fileptr, const readerWithType* thing) {
   const char* type = thing->type;
   char* str = NULL;
   if (!strcmp(type, "Int_t")) {
      //TTreeReaderValue<Int_t>* reader = (TTreeReaderValue<Int_t>*)thing->reader;
      int val = *(int *)thing->reader;
      int len = snprintf(NULL,0,"%d",val)+1;
      char str[len];
      snprintf(str,len,"%d",val);
   } else if (!strcmp(type, "Float_t")) {
      //TTreeReaderValue<Float_t>* reader = (TTreeReaderValue<Float_t>*)thing->reader;
      float val = *(float *)thing->reader;
      int len = snprintf(NULL,0,"%f",val)+1;
      char str[len];
      snprintf(str,len,"%f",val);
   } else if (!strcmp(type, "Double32_t")) {
      //TTreeReaderValue<Double_t>* reader = (TTreeReaderValue<Double_t>*)thing->reader;
      double val = *(double *)thing->reader;
      int len = snprintf(NULL,0,"%lf",val)+1;
      char str[len];
      snprintf(str,len,"%lf",val);
   } else if (!strcmp(type, "UShort_t")) {
      //TTreeReaderValue<UShort_t>* reader = (TTreeReaderValue<UShort_t>*)thing->reader;
      unsigned short int val = *(unsigned short int *)thing->reader;
      int len = snprintf(NULL,0,"%hu",val)+1;
      char str[len];
      snprintf(str,len,"%hu",val);
   }
   *fileptr << str;
}

// read tree and store in .txt file "csv"
void storeCSV(const char* tree_name, TDirectoryFile* f, const int masterclass_index) {
   const char* fname = f->GetName();
   int len = strlen(DIRNAME)+strlen(fname)+15+2*strlen(tree_name);
   if (masterclass_index > 9) {
      len++;
   }
   char newfname[len];
   snprintf(newfname, len, "%scsvs/%s/%d_%s_%s.csv",DIRNAME,tree_name,masterclass_index,fname,tree_name);
   
   ofstream myfile; // make output file
   myfile.open(newfname);
   printf("%d_%s_%s.csv\n",masterclass_index,fname,tree_name);
   TTree *tree = f->Get<TTree>(tree_name);
   
   // implement TTreeReader class
   TTreeReader reader(tree);
   
   // dynamic TTreeReaderValue s, basically for columns
   vector<readerWithType*> column_readers;

   TObjArray* leavesList = tree->GetListOfLeaves();
   for (TObject* obj: *leavesList) {
      TLeaf* leaf = (TLeaf *)obj;
      const char* name = leaf->GetName();
      const char* type = leaf->GetTypeName();
      readerWithType* thing = makeTreeReaderFromType(type, name, reader);
      column_readers.push_back(thing);
      myfile << name;
   }
   myfile << "\n";
   if (!myfile.is_open()) { 
      printf("we not guuci :(\n");
   }

   bool firstEntry = true;
   int i = 0;
   while (reader.Next()) {
      for (int i = 0; i < column_readers.size(); i++) {
         readerWithType* col_reader = column_readers[i];
         if (col_reader->reader) {
            addReaderValueToFile(&myfile, col_reader);
         }
      }
      myfile << "\n";
      i++;
   }
   if (reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd) {
      printf("reader did not read all entries successfully");
   }
   printf("processed %d entries\n",i);
   myfile.close();
}

void treeToCSV() {
   // make pathname from indexed root file
   int index = 1;
   const char* fname = fileNames[index];
   int len = strlen(DIRNAME)+40;
   char pathname[len];
   snprintf(pathname, len, "%sroots/%s.root", DIRNAME, fname);
   printf("%s\n", pathname);
   // open path and get list of events (34 for index 1)
   TFile *fd = TFile::Open(pathname);
   TList *events= fd->GetListOfKeys();
   //TDirectoryFile *f = fd->GetEntry(fname);
   printf("found %d entries\n", events->GetEntries());
   for(int i=0; i<events->GetEntries(); i++) {
      TDirectoryFile* f = (TDirectoryFile *)events->At(i);
      const char* fname = f->GetName();
      TDirectoryFile* event = (TDirectoryFile*)fd->Get(fname);
      //storeTree("Clusters", event, index);
      //storeTree("RecTracks", event, index);
      storeCSV("Clusters", event, index);
      //storeCSV("RecTracks", event, index);
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
